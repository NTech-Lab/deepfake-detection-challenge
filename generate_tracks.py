import os
import yaml
import tqdm
import glob
import pickle

from tracker.iou_tracker import track_iou
from detect_faces_on_videos import DETECTIONS_FILE_NAME, DETECTIONS_ROOT

SIGMA_L = 0.3
SIGMA_H = 0.9
SIGMA_IOU = 0.3
T_MIN = 1

TRACKS_FILE_NAME = 'tracks.pkl'


def get_tracks(detections):
    if len(detections) == 0:
        return []

    converted_detections = []
    for i, detections_per_frame in enumerate(detections):
        converted_detections_per_frame = []
        for j, (bbox, score) in enumerate(zip(detections_per_frame['boxes'], detections_per_frame['scores'])):
            bbox = tuple(bbox.tolist())
            converted_detections_per_frame.append({'bbox': bbox, 'score': score})
        converted_detections.append(converted_detections_per_frame)

    tracks = track_iou(converted_detections, SIGMA_L, SIGMA_H, SIGMA_IOU, T_MIN)
    tracks_converted = []
    for track in tracks:
        track_converted = []
        start_frame = track['start_frame'] - 1
        for i, bbox in enumerate(track['bboxes']):
            track_converted.append((start_frame + i, bbox))
        tracks_converted.append(track_converted)

    return tracks_converted


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    root_dir = os.path.join(config['ARTIFACTS_PATH'], DETECTIONS_ROOT)
    detections_content = []
    for path in glob.iglob(os.path.join(root_dir, '**', DETECTIONS_FILE_NAME), recursive=True):
        rel_path = path[len(root_dir) + 1:]
        detections_content.append(rel_path)

    detections_content = sorted(detections_content)
    print('Total number of videos: {}'.format(len(detections_content)))

    video_to_tracks = {}
    for rel_path in tqdm.tqdm(detections_content):
        video = os.path.dirname(rel_path)
        with open(os.path.join(root_dir, rel_path), 'rb') as f:
            detections = pickle.load(f)
        video_to_tracks[video] = get_tracks(detections)

    track_count = sum([len(tracks) for tracks in video_to_tracks.values()])
    print('Total number of tracks: {}'.format(track_count))

    with open(os.path.join(config['ARTIFACTS_PATH'], TRACKS_FILE_NAME), 'wb') as f:
        pickle.dump(video_to_tracks, f)


if __name__ == '__main__':
    main()
