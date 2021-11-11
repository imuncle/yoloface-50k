# yoloface-50k

本仓库包含已量化的yoloface tflite模型以及未量化的onnx模型，h5模型和pb模型，另外还有使用pytorch解析运行yolo `cfg`和`weight`的小工具

本仓库所使用的网络模型来自[dog-qiuqiu/MobileNet-Yolo](https://github.com/dog-qiuqiu/MobileNet-Yolo)，感谢这位大佬

```txt
ncnn: yoloface使用ncnn推理后的工程，可以在CPU上实时运行。其中libncnn.a是在Ubuntu 20.04上编译的，如是不同的操作系统，请下载[ncnn](https://github.com/Tencent/ncnn)自行编译替换

tensorflow: 内含yolo转h5、h5转pb的代码

tflite: pb转tflite并量化的代码
```

> 注意：代码中的`nms`是虚假的`nms`，并没有进行非极大值抑制，只是提取出了置信度较高的目标，使用时可自己添加