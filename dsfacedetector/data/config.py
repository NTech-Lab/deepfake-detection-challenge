import numpy as np


def test_base_transform(image, mean):
    x = image.astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class TestBaseTransform:
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        return test_base_transform(image, self.mean)


widerface_640 = {
    'num_classes': 2,

    'feature_maps': [160, 80, 40, 20, 10, 5],
    'min_dim': 640,

    'steps': [4, 8, 16, 32, 64, 128],  # stride

    'variance': [0.1, 0.2],
    'clip': True,  # make default box in [0,1]
    'name': 'WIDERFace',
    'l2norm_scale': [10, 8, 5],
    'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    'extras': [256, 'S', 512, 128, 'S', 256],

    'mbox': [1, 1, 1, 1, 1, 1],
    'min_sizes': [16, 32, 64, 128, 256, 512],
    'max_sizes': [],
    'aspect_ratios': [[1.5], [1.5], [1.5], [1.5], [1.5], [1.5]],  # [1,2]  default 1

    'backbone': 'resnet152',
    'feature_pyramid_network': True,
    'bottom_up_path': False,
    'feature_enhance_module': True,
    'max_in_out': True,
    'focal_loss': False,
    'progressive_anchor': True,
    'refinedet': False,
    'max_out': False,
    'anchor_compensation': False,
    'data_anchor_sampling': False,

    'overlap_thresh': [0.4],
    'negpos_ratio': 3,
    # test
    'nms_thresh': 0.3,
    'conf_thresh': 0.01,
    'num_thresh': 5000,
}
