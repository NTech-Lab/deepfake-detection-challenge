import yaml
import os
import json
from collections import defaultdict
import glob

from generate_aligned_tracks import METADATA_FILE_NAME
from extract_tracks_from_videos import TRACKS_ROOT

TRACK_PAIRS_FILE_NAME = 'track_pairs.txt'


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    video_to_tracks = defaultdict(list)

    for path in glob.iglob(os.path.join(config['ARTIFACTS_PATH'], TRACKS_ROOT, 'dfdc_train_part_*', '*.mp4_*')):
        parts = path.split('/')
        rel_path = '/'.join(parts[-2:])
        video = '_'.join(rel_path.split('_')[:-3])
        video_to_tracks[video].append(rel_path)

    video_to_meta = {}

    for path in glob.iglob(os.path.join(config['DFDC_DATA_PATH'], '**', METADATA_FILE_NAME), recursive=True):
        root = os.path.basename(os.path.dirname(path))
        with open(path, 'r') as f:
            for video, meta in json.load(f).items():
                video_to_meta[os.path.join(root, video)] = meta

    fake_video_to_real_video = {}
    for video in video_to_meta:
        root = os.path.dirname(video)
        meta = video_to_meta[video]
        if meta['label'] == 'FAKE':
            original_video = os.path.join(root, meta['original'])
            fake_video_to_real_video[video] = original_video

    print('Total number of fake videos: {}'.format(len(fake_video_to_real_video)))

    track_pairs = []

    fake_videos = sorted(fake_video_to_real_video)
    for fake_video in fake_videos:
        real_video = fake_video_to_real_video[fake_video]
        fake_tracks = video_to_tracks[fake_video]
        real_tracks = video_to_tracks[real_video]

        for fake_track in fake_tracks:
            if not os.path.exists(os.path.join(config['ARTIFACTS_PATH'], TRACKS_ROOT, fake_track, '0.png')):
                continue
            suffix = fake_track[len(fake_video):]
            for real_track in real_tracks:
                if not os.path.exists(os.path.join(config['ARTIFACTS_PATH'], TRACKS_ROOT, real_track, '0.png')):
                    continue
                if real_track.endswith(suffix):
                    track_pairs.append((real_track, fake_track))
                    break

    print('Total number of track pairs: {}'.format(len(track_pairs)))

    with open(os.path.join(config['ARTIFACTS_PATH'], TRACK_PAIRS_FILE_NAME), 'w') as f:
        for real_track, fake_track in track_pairs:
            f.write('{},{}\n'.format(real_track, fake_track))


if __name__ == '__main__':
    main()
