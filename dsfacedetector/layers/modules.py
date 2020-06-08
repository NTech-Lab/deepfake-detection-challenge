import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepHeadModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DeepHeadModule, self).__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._mid_channels = min(self._input_channels, 256)

        self.conv1 = nn.Conv2d(self._input_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self._mid_channels, self._output_channels, kernel_size=1, dilation=1, stride=1,
                               padding=0)

    def forward(self, x):
        return self.conv4(
            F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x), inplace=True)), inplace=True)), inplace=True))


class FEM(nn.Module):
    def __init__(self, channel_size):
        super(FEM, self).__init__()
        self.cs = channel_size
        self.cpm1 = nn.Conv2d(self.cs, 256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm2 = nn.Conv2d(self.cs, 256, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm3 = nn.Conv2d(256, 128, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm4 = nn.Conv2d(256, 128, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm5 = nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1)

    def forward(self, x):
        x1_1 = F.relu(self.cpm1(x), inplace=True)
        x1_2 = F.relu(self.cpm2(x), inplace=True)
        x2_1 = F.relu(self.cpm3(x1_2), inplace=True)
        x2_2 = F.relu(self.cpm4(x1_2), inplace=True)
        x3_1 = F.relu(self.cpm5(x2_2), inplace=True)
        return torch.cat((x1_1, x2_1, x3_1), 1)


def upsample_product(x, y):
    '''Upsample and add two feature maps.
       Args:
         x: (Variable) top feature map to be upsampled.
         y: (Variable) lateral feature map.
       Returns:
         (Variable) added feature map.
       Note in PyTorch, when input size is odd, the upsampled feature map
       with `F.upsample(..., scale_factor=2, mode='nearest')`
       maybe not equal to the lateral feature map size.
       e.g.
       original input size: [N,_,15,15] ->
       conv2d feature map size: [N,_,8,8] ->
       upsampled feature map size: [N,_,16,16]
       So we choose bilinear upsample which supports arbitrary output sizes.
       '''
    _, _, H, W = y.size()

    # FOR ONNX CONVERSION
    # return F.interpolate(x, scale_factor=2, mode='nearest') * y
    return F.interpolate(x, size=(int(H), int(W)), mode='bilinear', align_corners=False) * y


def pa_multibox(output_channels):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        if k == 0:
            loc_output = 4
            conf_output = 2
        elif k == 1:
            loc_output = 8
            conf_output = 4
        else:
            loc_output = 12
            conf_output = 6
        loc_layers += [DeepHeadModule(512, loc_output)]
        conf_layers += [DeepHeadModule(512, (2 + conf_output))]
    return (loc_layers, conf_layers)


def mio_module(each_mmbox, len_conf, your_mind_state='peasant'):
    # chunk = torch.split(each_mmbox, 1, 1) - !!!!! failed to export on PyTorch v1.0.1 (ONNX version 1.3)
    chunk = torch.chunk(each_mmbox, int(each_mmbox.shape[1]), 1)

    # some hacks for ONNX and Inference Engine export
    if your_mind_state == 'peasant':
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
    elif your_mind_state == 'advanced':
        bmax = torch.max(each_mmbox[:, :3], 1)[0].unsqueeze(0)
    else: # supermind
        bmax = torch.nn.functional.max_pool3d(each_mmbox[:, :3], kernel_size=(3, 1, 1))

    cls = (torch.cat((bmax, chunk[3]), dim=1) if len_conf == 0 else torch.cat((chunk[3], bmax), dim=1))
    cls = torch.cat((cls, *list(chunk[4:])), dim=1)
    return cls