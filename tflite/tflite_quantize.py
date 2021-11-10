import tensorflow.compat.v1 as tf
from tensorflow import keras
import numpy as np
import os
import cv2

img_lists = []
for root, dirs, files in os.walk("small_dataset"):
    img_lists = files

def representative_dataset_gen():
    for img in img_lists:
        img = cv2.imread('small_dataset/'+img)
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape
        w_scale = W/56.
        h_scale = H/56.
        input = cv2.resize(input, (56, 56))
        input = input[np.newaxis,:,:,:]
        input = input/255.
        # data = np.random.rand(1, 56, 56, 3)
        yield [input.astype(np.float32)]

# 定义转换器
converter = tf.lite.TFLiteConverter.from_frozen_graph("model.pb", ["Input"], ["Identity"], {"Input":[1,56,56,3]})
# 定义量化配置
converter.representative_dataset = representative_dataset_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
# converter.allow_custom_ops=True
 
# converter.inference_type = tf.int8    #tf.lite.constants.QUANTIZED_UINT8
# input_arrays = converter.get_input_arrays()
# converter.quantized_input_stats = {input_arrays[0]: (0, 0)} # mean, std_dev
# converter.default_ranges_stats = (-128, 127)


# 转换并保存
quantize_model = converter.convert()
open("yoloface_int8.tflite", "wb").write(quantize_model)

# data = np.random.rand(1, 56, 56, 3).astype(np.float32)
# output = loaded_model(data)
# print(output.shape)