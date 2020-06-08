import argparse
import os
import glob
import yaml
import pickle
import tqdm

import torch
from torch.utils.data import DataLoader

from dsfacedetector.face_ssd_infer import SSD
from datasets import UnlabeledVideoDataset

DETECTOR_WEIGHTS_PATH = 'external_data/WIDERFace_DSFD_RES152.pth'
DETECTOR_THRESHOLD = 0.3
DETECTOR_STEP = 6
DETECTOR_TARGET_SIZE = (512, 512)

BATCH_SIZE = 1
NUM_WORKERS = 0

DETECTIONS_ROOT = 'detections'
DETECTIONS_FILE_NAME = 'detections.pkl'


def main():
    parser = argparse.ArgumentParser(description='Detects faces on videos')
    parser.add_argument('--num_parts', type=int, default=1, help='Number of parts')
    parser.add_argument('--part', type=int, default=0, help='Part index')

    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    content = []
    for path in glob.iglob(os.path.join(config['DFDC_DATA_PATH'], 'dfdc_train_part_*', '*.mp4')):
        parts = path.split('/')
        content.append('/'.join(parts[-2:]))
    content = sorted(content)

    print('Total number of videos: {}'.format(len(content)))

    part_size = len(content) // args.num_parts + 1
    assert part_size * args.num_parts >= len(content)
    part_start = part_size * args.part
    part_end = min(part_start + part_size, len(content))
    print('Part {} ({}, {})'.format(args.part, part_start, part_end))

    dataset = UnlabeledVideoDataset(config['DFDC_DATA_PATH'], content[part_start:part_end])

    detector = SSD('test')
    state = torch.load(DETECTOR_WEIGHTS_PATH, map_location=lambda storage, loc: storage)
    detector.load_state_dict(state)
    device = torch.device('cuda')
    detector = detector.eval().to(device)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=lambda X: X,
                        drop_last=False)

    dst_root = os.path.join(config['ARTIFACTS_PATH'], DETECTIONS_ROOT)
    os.makedirs(dst_root, exist_ok=True)

    for video_sample in tqdm.tqdm(loader):
        frames = video_sample[0]['frames']
        video_idx = video_sample[0]['index']
        video_rel_path = dataset.content[video_idx]

        detections = []
        for frame in frames[::DETECTOR_STEP]:
            with torch.no_grad():
                detections_per_frame = detector.detect_on_image(frame, DETECTOR_TARGET_SIZE, device, is_pad=False,
                                                                keep_thresh=DETECTOR_THRESHOLD)
                detections.append({'boxes': detections_per_frame[:, :4], 'scores': detections_per_frame[:, 4]})

        os.makedirs(os.path.join(dst_root, video_rel_path), exist_ok=True)
        with open(os.path.join(dst_root, video_rel_path, DETECTIONS_FILE_NAME), 'wb') as f:
            pickle.dump(detections, f)


if __name__ == '__main__':
    main()
