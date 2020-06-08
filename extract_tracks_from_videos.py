import argparse
import os
import yaml
import random
import pickle
import tqdm

import cv2
import numpy as np

from generate_aligned_tracks import ALIGNED_TRACKS_FILE_NAME

SEED = 0xDEADFACE
TRACK_LENGTH = 50
DETECTOR_STEP = 6
BOX_MULT = 1.5

TRACKS_ROOT = 'tracks'
BOXES_FILE_NAME = 'boxes.float32'


def main():
    parser = argparse.ArgumentParser(description='Extracts tracks from videos')
    parser.add_argument('--num_parts', type=int, default=1, help='Number of parts')
    parser.add_argument('--part', type=int, default=0, help='Part')

    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    with open(os.path.join(config['ARTIFACTS_PATH'], ALIGNED_TRACKS_FILE_NAME), 'rb') as f:
        aligned_tracks = pickle.load(f)

    part_size = len(aligned_tracks) // args.num_parts + 1
    assert part_size * args.num_parts >= len(aligned_tracks)
    part_start = part_size * args.part
    part_end = min(part_start + part_size, len(aligned_tracks))
    print('Part {} ({}, {})'.format(args.part, part_start, part_end))

    random.seed(SEED)
    for real_video, fake_video, aligned_track in tqdm.tqdm(aligned_tracks[part_start:part_end]):
        if len(aligned_track) < TRACK_LENGTH // DETECTOR_STEP:
            continue
        real_boxes = [item[1] for item in aligned_track]
        fake_boxes = [item[2] for item in aligned_track]
        start_idx = random.randint(0, len(aligned_track) - TRACK_LENGTH // DETECTOR_STEP)
        start_frame = aligned_track[start_idx][0] * DETECTOR_STEP
        middle_idx = start_idx + TRACK_LENGTH // DETECTOR_STEP // 2

        if random.choice([False, True]):
            xmin, ymin, xmax, ymax = real_boxes[middle_idx]
        else:
            xmin, ymin, xmax, ymax = fake_boxes[middle_idx]

        width = xmax - xmin
        height = ymax - ymin
        xcenter = xmin + width / 2
        ycenter = ymin + height / 2
        width = width * BOX_MULT
        height = height * BOX_MULT
        xmin = xcenter - width / 2
        ymin = ycenter - height / 2
        xmax = xmin + width
        ymax = ymin + height

        for video, boxes in [(real_video, real_boxes), (fake_video, fake_boxes)]:
            capture = cv2.VideoCapture(os.path.join(config['DFDC_DATA_PATH'], video))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0:
                continue
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

            xmin = max(int(xmin), 0)
            xmax = min(int(xmax), frame_width)
            ymin = max(int(ymin), 0)
            ymax = min(int(ymax), frame_height)

            dst_root = os.path.join(config['ARTIFACTS_PATH'], TRACKS_ROOT,
                                    video + '_{}_{}_{}'.format(start_frame, xmin, ymin))
            if os.path.exists(dst_root):
                continue
            os.makedirs(dst_root)
            for i in range(start_frame + TRACK_LENGTH):
                capture.grab()
                if i < start_frame:
                    continue
                ret, frame = capture.retrieve()
                if not ret:
                    continue
                face = frame[ymin:ymax, xmin:xmax]
                dst_path = os.path.join(dst_root, '{}.png'.format(i - start_frame))
                cv2.imwrite(dst_path, face)

            boxes = np.array(boxes, dtype=np.float32)
            boxes[:, 0] -= xmin
            boxes[:, 1] -= ymin
            boxes[:, 2] -= xmin
            boxes[:, 3] -= ymin
            boxes.tofile(os.path.join(dst_root, BOXES_FILE_NAME))


if __name__ == '__main__':
    main()
