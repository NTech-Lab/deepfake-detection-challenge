import os
import yaml

import torch

WEIGHTS_MAPPING = {
    'snapshots/efficientnet-b7_ns_aa-original-mstd0.5_large_crop_100k/snapshot_100000.pth': 'efficientnet-b7_ns_aa-original-mstd0.5_large_crop_100k_v4_cad79a/snapshot_100000.fp16.pth',
    'snapshots/efficientnet-b7_ns_aa-original-mstd0.5_re_100k/snapshot_100000.pth': 'efficientnet-b7_ns_aa-original-mstd0.5_re_100k_v4_cad79a/snapshot_100000.fp16.pth',
    'snapshots/efficientnet-b7_ns_seq_aa-original-mstd0.5_100k/snapshot_100000.pth': 'efficientnet-b7_ns_seq_aa-original-mstd0.5_100k_v4_cad79a/snapshot_100000.fp16.pth'
}

SRC_DETECTOR_WEIGHTS = 'external_data/WIDERFace_DSFD_RES152.pth'
DST_DETECTOR_WEIGHTS = 'WIDERFace_DSFD_RES152.fp16.pth'


def copy_weights(src_path, dst_path):
    state = torch.load(src_path, map_location=lambda storage, loc: storage)
    state = {key: value.half() for key, value in state.items()}
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(state, dst_path)


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    for src_rel_path, dst_rel_path in WEIGHTS_MAPPING.items():
        src_path = os.path.join(config['ARTIFACTS_PATH'], src_rel_path)
        dst_path = os.path.join(config['MODELS_PATH'], dst_rel_path)
        copy_weights(src_path, dst_path)

    copy_weights(SRC_DETECTOR_WEIGHTS, os.path.join(config['MODELS_PATH'], DST_DETECTOR_WEIGHTS))


if __name__ == '__main__':
    main()