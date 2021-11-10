import torch
import torch.nn as nn
import torchvision
from itertools import chain
import cv2
import numpy as np

def Conv2D(in_channel, out_channel, filter_size, stride, pad, is_relu=True, groups=1):
    if is_relu:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, filter_size, stride, padding=pad, bias=False, groups=groups),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, filter_size, stride, padding=pad, bias=False, groups=groups),
            nn.BatchNorm2d(out_channel)
        )

def depthwise_conv(in_channel, hidden_channel, out_channel, stride1=1, stride2=1, relu=False):
    return nn.Sequential(
        Conv2D(in_channel, hidden_channel, 3, stride1, 1, is_relu=True, groups=hidden_channel),
        Conv2D(hidden_channel, out_channel, 1, stride2, 0, is_relu=relu)
    )

class yoloface(nn.Module):
    def __init__(self):
        super(yoloface, self).__init__()
        self.conv1 = Conv2D(3, 8, 3, 2, 1, is_relu=True)# 0
        self.conv2 = depthwise_conv(8, 8, 4)# 1-2
        self.conv3 = Conv2D(4, 18, 1, 1, 0, is_relu=True)# 3
        self.conv4 = depthwise_conv(18, 18, 6, stride1=2)# 4-5
        self.conv5 = Conv2D(6, 36, 1, 1, 0, is_relu=True)# 6
        self.conv6 = depthwise_conv(36, 36, 6)# 7-8
        # shortcut1 from conv4
        self.conv7 = Conv2D(6, 18, 1, 1, 0, is_relu=True)# 10
        # route conv3
        self.maxpool1 = nn.MaxPool2d(kernel_size=8, stride=2, padding=int((8 - 1) // 2))# 12
        # route maxpool1 + conv7
        self.conv8 = Conv2D(36, 24, 1, 1, 0, is_relu=True)# 14
        self.conv9 = depthwise_conv(24, 24, 8, stride1=2)# 15-16
        self.conv10 = Conv2D(8, 40, 1, 1, 0, is_relu=True)# 17
        self.conv11 = depthwise_conv(40, 40, 8)# 18-19
        # shortcut2 from conv9
        self.conv12 = Conv2D(8, 40, 1, 1, 0, is_relu=True)# 21
        self.conv13 = depthwise_conv(40, 40, 8)# 22-23
        # shortcut3 from shortcut2
        self.conv14 = Conv2D(8, 24, 1, 1, 0, is_relu=True)# 25
        # route conv8
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=int((4 - 1) // 2))# 27
        # route maxpool2 + conv14
        self.conv15 = Conv2D(48, 40, 1, 1, 0, is_relu=True)# 29
        self.conv16 = depthwise_conv(40, 40, 32, relu=True)# 30-31
        self.conv17 = nn.Conv2d(32, 18, 1, 1, padding=0, bias=True)# 32
        self.detector = yolo_layer([[9,14], [12,17], [22,21]])
    
    def forward(self, input):
        conv3 = self.conv3(self.conv2(self.conv1(input)))
        
        conv4 = self.conv4(conv3)
        conv6 = self.conv6(self.conv5(conv4))
        conv6 = conv4 + conv6
        conv7 = self.conv7(conv6)
        maxpool1 = self.maxpool1(conv3)
        route1 = torch.cat([maxpool1, conv7], axis=1)
        conv8 = self.conv8(route1)
        conv9 = self.conv9(conv8)
        conv11 = self.conv11(self.conv10(conv9))
        conv11 = conv9 + conv11
        conv13 = self.conv13(self.conv12(conv11))
        
        conv13 = conv11 + conv13
        conv14 = self.conv14(conv13)
        maxpool2 = self.maxpool2(conv8)
        route2 = torch.cat([maxpool2, conv14], axis=1)
        conv17 = self.conv17(self.conv16(self.conv15(route2)))
        # tmp_input = conv17.numpy()
        # print(tmp_input.shape)
        # for i in range(8):
        #     tmp = np.abs(tmp_input[0,i,:,:])
        #     tmp *= 255/np.max(tmp)
        #     cv2.imshow('imgg%d'%i, tmp.astype(np.uint8))
        #     cv2.waitKey(0)
        # return conv17
        return self.detector(conv17, input.shape[2])
    
    def load_conv_bn_weights(self, weights, ptr, conv_layer, bn_layer):
        num_b = bn_layer.bias.numel()  # Number of biases
        # Bias
        bn_b = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
        bn_layer.bias.data.copy_(bn_b)
        ptr += num_b
        # Weight
        bn_w = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
        bn_layer.weight.data.copy_(bn_w)
        ptr += num_b
        # Running Mean
        bn_rm = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
        bn_layer.running_mean.data.copy_(bn_rm)
        ptr += num_b
        # Running Var
        bn_rv = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
        bn_layer.running_var.data.copy_(bn_rv)
        ptr += num_b

        # Load conv. weights
        num_w = conv_layer.weight.numel()
        conv_w = torch.from_numpy(
            weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
        conv_layer.weight.data.copy_(conv_w)
        ptr += num_w

        return ptr
    
    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        # 开始加载权重
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv1[0], self.conv1[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv2[0][0], self.conv2[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv2[1][0], self.conv2[1][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv3[0], self.conv3[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv4[0][0], self.conv4[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv4[1][0], self.conv4[1][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv5[0], self.conv5[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv6[0][0], self.conv6[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv6[1][0], self.conv6[1][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv7[0], self.conv7[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv8[0], self.conv8[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv9[0][0], self.conv9[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv9[1][0], self.conv9[1][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv10[0], self.conv10[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv11[0][0], self.conv11[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv11[1][0], self.conv11[1][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv12[0], self.conv12[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv13[0][0], self.conv13[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv13[1][0], self.conv13[1][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv14[0], self.conv14[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv15[0], self.conv15[1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv16[0][0], self.conv16[0][1])
        ptr = self.load_conv_bn_weights(weights, ptr, self.conv16[1][0], self.conv16[1][1])
        
        # Load conv. bias
        num_b = self.conv17.bias.numel()
        conv_b = torch.from_numpy(
            weights[ptr: ptr + num_b]).view_as(self.conv17.bias)
        self.conv17.bias.data.copy_(conv_b)
        ptr += num_b
        # Load conv. weights
        num_w = self.conv17.weight.numel()
        conv_w = torch.from_numpy(
            weights[ptr: ptr + num_w]).view_as(self.conv17.weight)
        self.conv17.weight.data.copy_(conv_w)
        ptr += num_w
        

class yolo_layer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors):
        super(yolo_layer, self).__init__()
        self.num_anchors = len(anchors)
        self.no = 6  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(-1, 1, 1, 2))
        self.stride = None

    def forward(self, x, img_size):
        # x.shape: 18*7*7
        x = x.squeeze(0)
        stride = img_size // x.shape[1]
        self.stride = stride
        _, ny, nx = x.shape  # x(18, 7, 7) to x(3, 7, 7, 6)
        x = x.view(self.num_anchors, self.no, ny, nx).permute(0, 2, 3, 1).contiguous()

        self.grid = self._make_grid(nx, ny).to(x.device)

        x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
        x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
        x[..., 4:] = x[..., 4:].sigmoid()
        x = x.view(-1, self.no)

        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, ny, nx, 2)).float()

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.25):
    x = prediction[prediction[..., 4] > conf_thres]

    if not x.shape[0]:
        return
    
    box = xywh2xyxy(x[:, :4])

    return box

model = yoloface()
model.load_darknet_weights('yoloface-50k.weights')
img = cv2.imread('small_dataset/img_82.jpg')
input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W, _ = img.shape
w_scale = W/56.
h_scale = H/56.
input = cv2.resize(input, (56, 56))
input = torch.from_numpy(input).permute(2, 0, 1).float().unsqueeze(0)
input = input/255.
with torch.no_grad():
    output = model(input)
# torch.onnx.export(model, input, 'yoloface.onnx',
#                     export_params=True,
#                     verbose=True,
#                     input_names=['input'],
#                     output_names=["output"],
#                     # opset_version=11,
#                     training=False)
output = non_max_suppression(output, 0.7)
if output != None:
    for detect in output:
        detect[[0,2]] *= w_scale
        detect[[1,3]] *= h_scale
        detect = detect.numpy().astype(np.int32)
        cv2.rectangle(img, (detect[0], detect[1]), (detect[2], detect[3]), (0,0,255), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
