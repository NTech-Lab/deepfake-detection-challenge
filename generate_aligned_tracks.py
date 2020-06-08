import glob
import os
import yaml
import json
from collections import defaultdict
import tqdm
import pickle

from tracker.utils import iou

from generate_tracks import TRACKS_FILE_NAME

MIN_TRACK_LENGTH = 5
IOU_THRESHOLD = 0.5
METADATA_FILE_NAME = 'metadata.json'

ALIGNED_TRACKS_FILE_NAME = 'aligned_tracks.pkl'


def get_track(tracks, min_track_length):
    good_tracks = [track for track in tracks if len(track) >= min_track_length]
    if len(good_tracks) == 1:
        return good_tracks[0]
    else:
        return None


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    video_to_meta = {}

    for path in glob.iglob(os.path.join(config['DFDC_DATA_PATH'], '**', METADATA_FILE_NAME), recursive=True):
        root = os.path.basename(os.path.dirname(path))
        with open(path, 'r') as f:
            for video, meta in json.load(f).items():
                video_to_meta[os.path.join(root, video)] = meta

    real_video_to_fake_videos = defaultdict(list)
    for video in video_to_meta:
        root = os.path.dirname(video)
        meta = video_to_meta[video]
        if meta['label'] == 'FAKE':
            original_video = os.path.join(root, meta['original'])
            real_video_to_fake_videos[original_video].append(video)

    print('Total number of real videos: {}'.format(len(real_video_to_fake_videos)))
    print('Total number of fake videos: {}'.format(sum([len(fake_videos) for fake_videos in real_video_to_fake_videos.items()])))

    with open(os.path.join(config['ARTIFACTS_PATH'], TRACKS_FILE_NAME), 'rb') as f:
        video_to_tracks = pickle.load(f)

    real_fake_aligned_tracks = []
    real_videos = sorted(real_video_to_fake_videos)
    for real_video in tqdm.tqdm(real_videos):
        if real_video not in video_to_tracks:
            continue
        real_tracks = [track for track in video_to_tracks[real_video] if len(track) >= MIN_TRACK_LENGTH]

        for fake_video in real_video_to_fake_videos[real_video]:
            if fake_video not in video_to_tracks:
                continue
            fake_tracks = [track for track in video_to_tracks[fake_video] if len(track) >= MIN_TRACK_LENGTH]

            for real_track in real_tracks:
                real_frame_idx_to_bbox = {}
                for real_frame_idx, real_bbox in real_track:
                    real_frame_idx_to_bbox[real_frame_idx] = real_bbox

                for fake_track in fake_tracks:
                    fake_frame_idx_to_bbox = {}
                    ious = []
                    for fake_frame_idx, fake_bbox in fake_track:
                        fake_frame_idx_to_bbox[fake_frame_idx] = fake_bbox
                        if fake_frame_idx in real_frame_idx_to_bbox:
                            real_bbox = real_frame_idx_to_bbox[fake_frame_idx]
                            ious.append(iou(real_bbox, fake_bbox))
                    if len(ious) > 0 and min(ious) > IOU_THRESHOLD:
                        start_frame_idx = max(min(real_frame_idx_to_bbox), min(fake_frame_idx_to_bbox))
                        end_frame_idx = min(max(real_frame_idx_to_bbox), max(fake_frame_idx_to_bbox)) + 1
                        assert start_frame_idx < end_frame_idx
                        real_fake_aligned_track = []
                        for frame_idx in range(start_frame_idx, end_frame_idx):
                            real_bbox = real_frame_idx_to_bbox[frame_idx]
                            fake_bbox = fake_frame_idx_to_bbox[frame_idx]
                            assert iou(real_bbox, fake_bbox) > IOU_THRESHOLD
                            real_fake_aligned_track.append((frame_idx, real_bbox, fake_bbox))
                        real_fake_aligned_tracks.append((real_video, fake_video, real_fake_aligned_track))
                        break

    print('Total number of tracks: {}'.format(len(real_fake_aligned_tracks)))

    with open(os.path.join(config['ARTIFACTS_PATH'], ALIGNED_TRACKS_FILE_NAME), 'wb') as f:
        pickle.dump(real_fake_aligned_tracks, f)


if __name__ == '__main__':
    main()
