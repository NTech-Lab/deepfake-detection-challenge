import os
import random
import glob

import cv2
import numpy as np

from torch.utils.data import Dataset


class UnlabeledVideoDataset(Dataset):
    def __init__(self, root_dir, content=None, transform=None):
        self.root_dir = os.path.normpath(root_dir)
        self.transform = transform

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

        capture = cv2.VideoCapture(path)

        frames = []
        if capture.isOpened():
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                if self.transform is not None:
                    frame = self.transform(frame)

                frames.append(frame)

        sample = {
            'frames': frames,
            'index': idx
        }

        return sample


class FaceDataset(Dataset):
    def __init__(self, root_dir, content, labels=None, transform=None):
        self.root_dir = os.path.normpath(root_dir)
        self.content = content
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rel_path = self.content[idx]
        path = os.path.join(self.root_dir, rel_path)

        face = cv2.imread(path, cv2.IMREAD_COLOR)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            face = self.transform(image=face)['image']

        sample = {
            'face': face,
            'index': idx
        }

        if self.labels is not None:
            sample['label'] = self.labels[idx]

        return sample


class TrackPairDataset(Dataset):
    FPS = 30

    def __init__(self, tracks_root, pairs_path, indices, track_length, track_transform=None, image_transform=None,
                 sequence_mode=True):
        self.tracks_root = os.path.normpath(tracks_root)
        self.track_transform = track_transform
        self.image_transform = image_transform
        self.indices = np.asarray(indices, dtype=np.int32)
        self.track_length = track_length
        self.sequence_mode = sequence_mode

        self.pairs = []
        with open(pairs_path, 'r') as f:
            for line in f:
                real_track, fake_track = line.strip().split(',')
                self.pairs.append((real_track, fake_track))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        real_track_path, fake_track_path = self.pairs[idx]

        real_track_path = os.path.join(self.tracks_root, real_track_path)
        fake_track_path = os.path.join(self.tracks_root, fake_track_path)

        if self.track_transform is not None:
            img = self.load_img(real_track_path, 0)
            src_height, src_width = img.shape[:2]
            track_transform_params = self.track_transform.get_params(self.FPS, src_height, src_width)
        else:
            track_transform_params = None

        real_track = self.load_track(real_track_path, self.indices, track_transform_params)
        fake_track = self.load_track(fake_track_path, self.indices, track_transform_params)

        if self.image_transform is not None:
            prev_state = random.getstate()
            transformed_real_track = []
            for img in real_track:
                if self.sequence_mode:
                    random.setstate(prev_state)
                transformed_real_track.append(self.image_transform(image=img)['image'])

            real_track = transformed_real_track

            random.setstate(prev_state)
            transformed_fake_track = []
            for img in fake_track:
                if self.sequence_mode:
                    random.setstate(prev_state)
                transformed_fake_track.append(self.image_transform(image=img)['image'])
            fake_track = transformed_fake_track

        sample = {
            'real': real_track,
            'fake': fake_track
        }

        return sample

    def load_img(self, track_path, idx):
        img = cv2.imread(os.path.join(track_path, '{}.png'.format(idx)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def load_track(self, track_path, indices, transform_params):
        if transform_params is None:
            track = np.stack([self.load_img(track_path, idx) for idx in indices])
        else:
            track = self.track_transform(track_path, self.FPS, *transform_params)
            indices = (indices.astype(np.float32) / self.track_length) * len(track)
            indices = np.round(indices).astype(np.int32).clip(0, len(track) - 1)
            track = track[indices]

        return track
