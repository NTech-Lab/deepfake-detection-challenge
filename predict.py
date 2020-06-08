import os
import yaml
import glob

import numpy as np
import cv2

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision.models.detection.transform import GeneralizedRCNNTransform

from albumentations import Compose, SmallestMaxSize, CenterCrop, Normalize, PadIfNeeded
from albumentations.pytorch import ToTensor

from dsfacedetector.face_ssd_infer import SSD
from tracker.iou_tracker import track_iou
from efficientnet_pytorch.model import EfficientNet, MBConvBlock

DETECTOR_WEIGHTS_PATH = 'WIDERFace_DSFD_RES152.fp16.pth'
DETECTOR_THRESHOLD = 0.3
DETECTOR_MIN_SIZE = 512
DETECTOR_MAX_SIZE = 512
DETECTOR_MEAN = (104.0, 117.0, 123.0)
DETECTOR_STD = (1.0, 1.0, 1.0)
DETECTOR_BATCH_SIZE = 16
DETECTOR_STEP = 3

TRACKER_SIGMA_L = 0.3
TRACKER_SIGMA_H = 0.9
TRACKER_SIGMA_IOU = 0.3
TRACKER_T_MIN = 7

VIDEO_MODEL_BBOX_MULT = 1.5
VIDEO_MODEL_MIN_SIZE = 224
VIDEO_MODEL_CROP_HEIGHT = 224
VIDEO_MODEL_CROP_WIDTH = 192
VIDEO_FACE_MODEL_TRACK_STEP = 2
VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH = 7
VIDEO_SEQUENCE_MODEL_TRACK_STEP = 14

VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH = 'efficientnet-b7_ns_seq_aa-original-mstd0.5_100k_v4_cad79a/snapshot_100000.fp16.pth'
FIRST_VIDEO_FACE_MODEL_WEIGHTS_PATH = 'efficientnet-b7_ns_aa-original-mstd0.5_large_crop_100k_v4_cad79a/snapshot_100000.fp16.pth'
SECOND_VIDEO_FACE_MODEL_WEIGHTS_PATH = 'efficientnet-b7_ns_aa-original-mstd0.5_re_100k_v4_cad79a/snapshot_100000.fp16.pth'

VIDEO_BATCH_SIZE = 1
VIDEO_TARGET_FPS = 15
VIDEO_NUM_WORKERS = 0


class UnlabeledVideoDataset(Dataset):
    def __init__(self, root_dir, content=None):
        self.root_dir = os.path.normpath(root_dir)
        if content is not None:
            self.content = content
        else:
            self.content = []
            for path in glob.iglob(os.path.join(self.root_dir, '**', '*.mp4'), recursive=True):
                rel_path = path[len(self.root_dir) + 1:]
                self.content.append(rel_path)
            self.content = sorted(self.content)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rel_path = self.content[idx]
        path = os.path.join(self.root_dir, rel_path)

        sample = {
            'frames': [],
            'index': idx
        }

        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            return sample

        fps = int(capture.get(cv2.CAP_PROP_FPS))
        video_step = round(fps / VIDEO_TARGET_FPS)
        if video_step == 0:
            return sample

        for i in range(frame_count):
            capture.grab()
            if i % video_step != 0:
                continue
            ret, frame = capture.retrieve()
            if not ret:
                continue

            sample['frames'].append(frame)

        return sample


class Detector(object):
    def __init__(self, weights_path):
        self.model = SSD('test')
        self.model.cuda().eval()

        state = torch.load(weights_path, map_location=lambda storage, loc: storage)
        state = {key: value.float() for key, value in state.items()}
        self.model.load_state_dict(state)

        self.transform = GeneralizedRCNNTransform(DETECTOR_MIN_SIZE, DETECTOR_MAX_SIZE, DETECTOR_MEAN, DETECTOR_STD)
        self.transform.eval()

    def detect(self, images):
        images = torch.stack([torch.from_numpy(image).cuda() for image in images])
        images = images.transpose(1, 3).transpose(2, 3).float()
        original_image_sizes = [img.shape[-2:] for img in images]
        images, _ = self.transform(images, None)
        with torch.no_grad():
            detections_batch = self.model(images.tensors).cpu().numpy()
        result = []
        for detections, image_size in zip(detections_batch, images.image_sizes):
            scores = detections[1, :, 0]
            keep_idxs = scores > DETECTOR_THRESHOLD
            detections = detections[1, keep_idxs, :]
            detections = detections[:, [1, 2, 3, 4, 0]]
            detections[:, 0] *= image_size[1]
            detections[:, 1] *= image_size[0]
            detections[:, 2] *= image_size[1]
            detections[:, 3] *= image_size[0]
            result.append({
                'scores': torch.from_numpy(detections[:, 4]),
                'boxes': torch.from_numpy(detections[:, :4])
            })

        result = self.transform.postprocess(result, images.image_sizes, original_image_sizes)
        return result


def get_tracks(detections):
    if len(detections) == 0:
        return []

    converted_detections = []
    frame_bbox_to_face_idx = {}
    for i, detections_per_frame in enumerate(detections):
        converted_detections_per_frame = []
        for j, (bbox, score) in enumerate(zip(detections_per_frame['boxes'], detections_per_frame['scores'])):
            bbox = tuple(bbox.tolist())
            frame_bbox_to_face_idx[(i, bbox)] = j
            converted_detections_per_frame.append({'bbox': bbox, 'score': score})
        converted_detections.append(converted_detections_per_frame)

    tracks = track_iou(converted_detections, TRACKER_SIGMA_L, TRACKER_SIGMA_H, TRACKER_SIGMA_IOU, TRACKER_T_MIN)
    tracks_converted = []
    for track in tracks:
        start_frame = track['start_frame'] - 1
        bboxes = np.array(track['bboxes'], dtype=np.float32)
        frame_indices = np.arange(start_frame, start_frame + len(bboxes)) * DETECTOR_STEP
        interp_frame_indices = np.arange(frame_indices[0], frame_indices[-1] + 1)
        interp_bboxes = np.zeros((len(interp_frame_indices), 4), dtype=np.float32)
        for i in range(4):
            interp_bboxes[:, i] = np.interp(interp_frame_indices, frame_indices, bboxes[:, i])

        track_converted = []
        for frame_idx, bbox in zip(interp_frame_indices, interp_bboxes):
            track_converted.append((frame_idx, bbox))
        tracks_converted.append(track_converted)

    return tracks_converted


class SeqExpandConv(nn.Module):
    def __init__(self, in_channels, out_channels, seq_length):
        super(SeqExpandConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.seq_length = seq_length

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        x = x.view(batch_size // self.seq_length, self.seq_length, in_channels, height, width)
        x = self.conv(x.transpose(1, 2).contiguous()).transpose(2, 1).contiguous()
        x = x.flatten(0, 1)
        return x


class TrackSequencesClassifier(object):
    def __init__(self, weights_path):
        model = EfficientNet.from_name('efficientnet-b7', override_params={'num_classes': 1})

        for module in model.modules():
            if isinstance(module, MBConvBlock):
                if module._block_args.expand_ratio != 1:
                    expand_conv = module._expand_conv
                    seq_expand_conv = SeqExpandConv(expand_conv.in_channels, expand_conv.out_channels,
                                                    VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH)
                    module._expand_conv = seq_expand_conv
        self.model = model.cuda().eval()

        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = Compose(
            [SmallestMaxSize(VIDEO_MODEL_MIN_SIZE), CenterCrop(VIDEO_MODEL_CROP_HEIGHT, VIDEO_MODEL_CROP_WIDTH),
             normalize, ToTensor()])

        state = torch.load(weights_path, map_location=lambda storage, loc: storage)
        state = {key: value.float() for key, value in state.items()}
        self.model.load_state_dict(state)

    def classify(self, track_sequences):
        track_sequences = [torch.stack([self.transform(image=face)['image'] for face in sequence]) for sequence in
                           track_sequences]
        track_sequences = torch.cat(track_sequences).cuda()
        with torch.no_grad():
            track_probs = torch.sigmoid(self.model(track_sequences)).flatten().cpu().numpy()

        return track_probs


class TrackFacesClassifier(object):
    def __init__(self, first_weights_path, second_weights_path):
        first_model = EfficientNet.from_name('efficientnet-b7', override_params={'num_classes': 1})
        self.first_model = first_model.cuda().eval()
        second_model = EfficientNet.from_name('efficientnet-b7', override_params={'num_classes': 1})
        self.second_model = second_model.cuda().eval()

        first_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.first_transform = Compose(
            [SmallestMaxSize(VIDEO_MODEL_CROP_WIDTH), PadIfNeeded(VIDEO_MODEL_CROP_HEIGHT, VIDEO_MODEL_CROP_WIDTH),
             CenterCrop(VIDEO_MODEL_CROP_HEIGHT, VIDEO_MODEL_CROP_WIDTH), first_normalize, ToTensor()])

        second_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.second_transform = Compose(
            [SmallestMaxSize(VIDEO_MODEL_MIN_SIZE), CenterCrop(VIDEO_MODEL_CROP_HEIGHT, VIDEO_MODEL_CROP_WIDTH),
             second_normalize, ToTensor()])

        first_state = torch.load(first_weights_path, map_location=lambda storage, loc: storage)
        first_state = {key: value.float() for key, value in first_state.items()}
        self.first_model.load_state_dict(first_state)

        second_state = torch.load(second_weights_path, map_location=lambda storage, loc: storage)
        second_state = {key: value.float() for key, value in second_state.items()}
        self.second_model.load_state_dict(second_state)

    def classify(self, track_faces):
        first_track_faces = []
        second_track_faces = []
        for i, face in enumerate(track_faces):
            if i % 4 < 2:
                first_track_faces.append(self.first_transform(image=face)['image'])
            else:
                second_track_faces.append(self.second_transform(image=face)['image'])
        first_track_faces = torch.stack(first_track_faces).cuda()
        second_track_faces = torch.stack(second_track_faces).cuda()
        with torch.no_grad():
            first_track_probs = torch.sigmoid(self.first_model(first_track_faces)).flatten().cpu().numpy()
            second_track_probs = torch.sigmoid(self.second_model(second_track_faces)).flatten().cpu().numpy()
            track_probs = np.concatenate((first_track_probs, second_track_probs))

        return track_probs


def extract_sequence(frames, start_idx, bbox, flip):
    frame_height, frame_width, _ = frames[start_idx].shape
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    xcenter = xmin + width / 2
    ycenter = ymin + height / 2
    width = width * VIDEO_MODEL_BBOX_MULT
    height = height * VIDEO_MODEL_BBOX_MULT
    xmin = xcenter - width / 2
    ymin = ycenter - height / 2
    xmax = xmin + width
    ymax = ymin + height

    xmin = max(int(xmin), 0)
    xmax = min(int(xmax), frame_width)
    ymin = max(int(ymin), 0)
    ymax = min(int(ymax), frame_height)

    sequence = []
    for i in range(VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH):
        face = cv2.cvtColor(frames[start_idx + i][ymin:ymax, xmin:xmax], cv2.COLOR_BGR2RGB)
        sequence.append(face)

    if flip:
        sequence = [face[:, ::-1] for face in sequence]

    return sequence


def extract_face(frame, bbox, flip):
    frame_height, frame_width, _ = frame.shape
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    xcenter = xmin + width / 2
    ycenter = ymin + height / 2
    width = width * VIDEO_MODEL_BBOX_MULT
    height = height * VIDEO_MODEL_BBOX_MULT
    xmin = xcenter - width / 2
    ymin = ycenter - height / 2
    xmax = xmin + width
    ymax = ymin + height

    xmin = max(int(xmin), 0)
    xmax = min(int(xmax), frame_width)
    ymin = max(int(ymin), 0)
    ymax = min(int(ymax), frame_height)

    face = cv2.cvtColor(frame[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2RGB)
    if flip:
        face = face[:, ::-1].copy()

    return face


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    detector = Detector(os.path.join(config['MODELS_PATH'], DETECTOR_WEIGHTS_PATH))
    track_sequences_classifier = TrackSequencesClassifier(os.path.join(config['MODELS_PATH'], VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH))
    track_faces_classifier = TrackFacesClassifier(os.path.join(config['MODELS_PATH'], FIRST_VIDEO_FACE_MODEL_WEIGHTS_PATH),
                                                  os.path.join(config['MODELS_PATH'], SECOND_VIDEO_FACE_MODEL_WEIGHTS_PATH))

    dataset = UnlabeledVideoDataset(os.path.join(config['DFDC_DATA_PATH'], 'test_videos'))
    print('Total number of videos: {}'.format(len(dataset)))

    loader = DataLoader(dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=False, num_workers=VIDEO_NUM_WORKERS,
                        collate_fn=lambda X: X,
                        drop_last=False)

    video_name_to_score = {}

    for video_sample in loader:
        frames = video_sample[0]['frames']
        detector_frames = frames[::DETECTOR_STEP]
        video_idx = video_sample[0]['index']
        video_rel_path = dataset.content[video_idx]
        video_name = os.path.basename(video_rel_path)

        if len(frames) == 0:
            video_name_to_score[video_name] = 0.5
            continue

        detections = []
        for start in range(0, len(detector_frames), DETECTOR_BATCH_SIZE):
            end = min(len(detector_frames), start + DETECTOR_BATCH_SIZE)
            detections_batch = detector.detect(detector_frames[start:end])
            for detections_per_frame in detections_batch:
                detections.append({key: value.cpu().numpy() for key, value in detections_per_frame.items()})

        tracks = get_tracks(detections)
        if len(tracks) == 0:
            video_name_to_score[video_name] = 0.5
            continue

        sequence_track_scores = []
        for track in tracks:
            track_sequences = []
            for i, (start_idx, _) in enumerate(
                    track[:-VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH + 1:VIDEO_SEQUENCE_MODEL_TRACK_STEP]):
                assert start_idx >= 0 and start_idx + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH <= len(frames)
                _, bbox = track[i * VIDEO_SEQUENCE_MODEL_TRACK_STEP + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH // 2]
                track_sequences.append(extract_sequence(frames, start_idx, bbox, i % 2 == 0))
            sequence_track_scores.append(track_sequences_classifier.classify(track_sequences))

        face_track_scores = []
        for track in tracks:
            track_faces = []
            for i, (frame_idx, bbox) in enumerate(track[::VIDEO_FACE_MODEL_TRACK_STEP]):
                face = extract_face(frames[frame_idx], bbox, i % 2 == 0)
                track_faces.append(face)
            face_track_scores.append(track_faces_classifier.classify(track_faces))

        sequence_track_scores = np.concatenate(sequence_track_scores)
        face_track_scores = np.concatenate(face_track_scores)
        track_probs = np.concatenate((sequence_track_scores, face_track_scores))

        delta = track_probs - 0.5
        sign = np.sign(delta)
        pos_delta = delta > 0
        neg_delta = delta < 0
        track_probs[pos_delta] = np.clip(0.5 + sign[pos_delta] * np.power(abs(delta[pos_delta]), 0.65), 0.01, 0.99)
        track_probs[neg_delta] = np.clip(0.5 + sign[neg_delta] * np.power(abs(delta[neg_delta]), 0.65), 0.01, 0.99)
        weights = np.power(abs(delta), 1.0) + 1e-4
        video_score = float((track_probs * weights).sum() / weights.sum())

        video_name_to_score[video_name] = video_score
        print('NUM DETECTION FRAMES: {}, VIDEO SCORE: {}. {}'.format(len(detections), video_name_to_score[video_name],
                                                                     video_rel_path))

    os.makedirs(os.path.dirname(config['SUBMISSION_PATH']), exist_ok=True)
    with open(config['SUBMISSION_PATH'], 'w') as f:
        f.write('filename,label\n')
        for video_name in sorted(video_name_to_score):
            score = video_name_to_score[video_name]
            f.write('{},{}\n'.format(video_name, score))


main()