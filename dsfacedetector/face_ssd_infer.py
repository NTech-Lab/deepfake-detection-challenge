# Source: https://github.com/vlad3996/FaceDetection-DSFD

import torch
import torchvision
import torch.nn as nn

from .data.config import TestBaseTransform, widerface_640 as cfg
from .layers import Detect, get_prior_boxes, FEM, pa_multibox, mio_module, upsample_product
from .utils import resize_image


class SSD(nn.Module):

    def __init__(self, phase, nms_thresh=0.3, nms_conf_thresh=0.01):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = 2
        self.cfg = cfg

        resnet = torchvision.models.resnet152(pretrained=False)

        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        self.layer5 = nn.Sequential(
            *[nn.Conv2d(2048, 512, kernel_size=1),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True),
              nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True)]
        )
        self.layer6 = nn.Sequential(
            *[nn.Conv2d(512, 128, kernel_size=1, ),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True)]
        )

        output_channels = [256, 512, 1024, 2048, 512, 256]

        # FPN
        fpn_in = output_channels

        self.latlayer3 = nn.Conv2d(fpn_in[3], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(fpn_in[2], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(fpn_in[1], fpn_in[0], kernel_size=1, stride=1, padding=0)

        self.smooth3 = nn.Conv2d(fpn_in[2], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(fpn_in[1], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d(fpn_in[0], fpn_in[0], kernel_size=1, stride=1, padding=0)

        # FEM
        cpm_in = output_channels

        self.cpm3_3 = FEM(cpm_in[0])
        self.cpm4_3 = FEM(cpm_in[1])
        self.cpm5_3 = FEM(cpm_in[2])
        self.cpm7 = FEM(cpm_in[3])
        self.cpm6_2 = FEM(cpm_in[4])
        self.cpm7_2 = FEM(cpm_in[5])

        # head
        head = pa_multibox(output_channels)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)

        if self.phase != 'onnx_export':
            self.detect = Detect(self.num_classes, 0, cfg['num_thresh'], nms_conf_thresh, nms_thresh,
                                 cfg['variance'])
            self.last_image_size = None
            self.last_feature_maps = None

        if self.phase == 'test':
            self.test_transform = TestBaseTransform((104, 117, 123))

    def forward(self, x):

        image_size = [x.shape[2], x.shape[3]]
        loc = list()
        conf = list()

        conv3_3_x = self.layer1(x)
        conv4_3_x = self.layer2(conv3_3_x)
        conv5_3_x = self.layer3(conv4_3_x)
        fc7_x = self.layer4(conv5_3_x)
        conv6_2_x = self.layer5(fc7_x)
        conv7_2_x = self.layer6(conv6_2_x)

        lfpn3 = upsample_product(self.latlayer3(fc7_x), self.smooth3(conv5_3_x))
        lfpn2 = upsample_product(self.latlayer2(lfpn3), self.smooth2(conv4_3_x))
        lfpn1 = upsample_product(self.latlayer1(lfpn2), self.smooth1(conv3_3_x))

        conv5_3_x = lfpn3
        conv4_3_x = lfpn2
        conv3_3_x = lfpn1

        sources = [conv3_3_x, conv4_3_x, conv5_3_x, fc7_x, conv6_2_x, conv7_2_x]

        sources[0] = self.cpm3_3(sources[0])
        sources[1] = self.cpm4_3(sources[1])
        sources[2] = self.cpm5_3(sources[2])
        sources[3] = self.cpm7(sources[3])
        sources[4] = self.cpm6_2(sources[4])
        sources[5] = self.cpm7_2(sources[5])

        # apply multibox head to source layers
        featuremap_size = []
        for (x, l, c) in zip(sources, self.loc, self.conf):
            featuremap_size.append([x.shape[2], x.shape[3]])
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            len_conf = len(conf)
            cls = mio_module(c(x), len_conf)
            conf.append(cls.permute(0, 2, 3, 1).contiguous())

        face_loc = torch.cat([o[:, :, :, :4].contiguous().view(o.size(0), -1) for o in loc], 1)
        face_loc = face_loc.view(face_loc.size(0), -1, 4)
        face_conf = torch.cat([o[:, :, :, :2].contiguous().view(o.size(0), -1) for o in conf], 1)
        face_conf = self.softmax(face_conf.view(face_conf.size(0), -1, self.num_classes))

        if self.phase != 'onnx_export':

            if self.last_image_size is None or self.last_image_size != image_size or self.last_feature_maps != featuremap_size:
                self.priors = get_prior_boxes(self.cfg, featuremap_size, image_size).to(face_loc.device)
                self.last_image_size = image_size
                self.last_feature_maps = featuremap_size
            with torch.no_grad():
                output = self.detect(face_loc, face_conf, self.priors)
        else:
            output = torch.cat((face_loc, face_conf), 2)
        return output

    def detect_on_image(self, source_image, target_size, device, is_pad=False, keep_thresh=0.3):

        image, shift_h_scaled, shift_w_scaled, scale = resize_image(source_image, target_size, is_pad=is_pad)

        x = torch.from_numpy(self.test_transform(image)).permute(2, 0, 1).to(device)
        x.unsqueeze_(0)

        detections = self.forward(x).cpu().numpy()

        scores = detections[0, 1, :, 0]
        keep_idxs = scores > keep_thresh  # find keeping indexes
        detections = detections[0, 1, keep_idxs, :]  # select detections over threshold
        detections = detections[:, [1, 2, 3, 4, 0]]  # reorder

        detections[:, [0, 2]] -= shift_w_scaled  # 0 or pad percent from left corner
        detections[:, [1, 3]] -= shift_h_scaled  # 0 or pad percent from top
        detections[:, :4] *= scale

        return detections
