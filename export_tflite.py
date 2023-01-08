import torch
import torchvision
import torch.nn as nn
from torchsummary import summary

import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

from collections import OrderedDict
import numpy as np
import mobiledet
import subprocess
import os


#pytorch - main

#tf_115
#netron -b runs/tflite/best.tflite -p 6009 --host 192.168.0.235

MODEL_BEST = "/last.pt"
MODEL_FILE = "2022-12-18_8-26-28"
TFLITE_FOLDER = "./runs/tflite/"

MODEL_PATH = "./runs/" + MODEL_FILE + MODEL_BEST
ONNX_PATH = "./runs/tflite/" + MODEL_FILE + ".onnx"
ONNX_OPT_PATH = "./runs/tflite/" + MODEL_FILE + "_opt.onnx"
MO_PATH = "./runs/tflite/" + MODEL_FILE + "_opt"
MO_PATH_XML = "./runs/tflite/" + MODEL_FILE + "_opt" + "/" + MODEL_FILE + "_opt.xml"
TF_PATH = "./runs/tflite/" + MODEL_FILE
TFLITE_PATH = "./runs/tflite/" + MODEL_FILE + ".tflite"


batch_size = 1
channels = 3
height = 320
width = 320
int8 = True
nms=False

classifier = 0
detector = 1


def export_onnx_saved_model_tflite_old():
    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
    #model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')).eval()

    """state_dict = torch.load(MODEL_PATH, map_location='cpu')
    new_state_dict = model.state_dict()
    for k, v in state_dict.items():
        if k=="classifier.1.weight":
            k = "classifier.0.weight"
        if k=="classifier.1.bias":
            k = "classifier.0.bias"

        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    """

    sample_input = torch.rand((batch_size, channels, height, width))

    train=False

    torch.onnx.export(
        model,                  # PyTorch Model
        sample_input,                    # Input tensor
        ONNX_PATH,        # Output file (eg. 'output_model.onnx')
        opset_version=12,       # Operator support version
        training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
        input_names=['input'],   # Input tensor name (arbitary)
        output_names=['output'] # Output tensor name (arbitary)
    )

    #convert to TF savedmodel
    onnx_model = onnx.load(ONNX_PATH)

    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(TF_PATH)

    #convert to tflite
    #converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
    #tflite_model = converter.convert()

    #convert uint8
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    #converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if int8:
        #dataset = LoadImages(check_dataset(data)['train'], img_size=imgsz, auto=False)  # representative data
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        converter.experimental_new_quantizer = True
    if nms:
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    tflite_model = converter.convert()

    # Save the model
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)

def representative_data_gen():
    #(x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()
    #mnist_train, _ = keras.datasets.mnist.load_data()

    x_train = np.ones((batch_size,height,width,3))
    #x_val = np.ones((1,CNN_INPUT,CNN_INPUT,3))

    #y_train = np.zeros((4,10))
    #y_val = np.zeros((1,10))

    #print("Images shape", mnist_train[0].shape) #Images shape (60000, 28, 28)
    print("Images shape", x_train[0].shape)#Images shape (28, 28)

    #x_train = np.expand_dims(x_train, -1)

    images = tf.cast(x_train, tf.float32)/255.0
    
    mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
    #print("REPR DAT SHAPE", mnist_ds)
    
    for input_value in mnist_ds.take(1):
        print(input_value.shape)
        yield [input_value]

def export_onnx_opt_openvino_tflite():
   #model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)

    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.ssdlite.SSDLite320_MobileNet_V3_Large_Weights)

    #model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    #model = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    #model = mobiledet.MobileDetTPU("detector", 6)
    #model.load_state_dict(torch.load(MODEL_PATH))
    #print(model)

    sample_input = torch.rand((batch_size, channels, height, width))

    train=False

    if not os.path.isdir(TFLITE_FOLDER):
        os.mkdir(TFLITE_FOLDER)

    torch.onnx.export(
        model,                  # PyTorch Model
        sample_input,                    # Input tensor
        ONNX_PATH,        # Output file (eg. 'output_model.onnx')
        opset_version=11,       # Operator support version 11-12 the best!!!
        training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
        input_names=['input'],   # Input tensor name (arbitary)
        output_names=['output'], # Output tensor name (arbitary)
        do_constant_folding = False
        )

    #sudo python3 -m onnxsim runs/best.onnx runs/best_opt.onnx
    subprocess.run(["sudo", "python3", "-m", "onnxsim", ONNX_PATH, ONNX_OPT_PATH])

    #mo --input_model runs/best_opt.onnx --output_dir runs/best
    subprocess.run(["mo", "--input_model", ONNX_OPT_PATH, "--output_dir", MO_PATH])

    #openvino2tensorflow --model_path runs/best/best_opt.xml --model_output_path runs/best_openvino --output_saved_model
    subprocess.run(["openvino2tensorflow", "--model_path", MO_PATH_XML, "--model_output_path", TF_PATH, "--output_saved_model"])

    #convert uint8

    if classifier:
        converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        #converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if int8:
            #dataset = LoadImages(check_dataset(data)['train'], img_size=imgsz, auto=False)  # representative data
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.uint8  # or tf.int8
            converter.inference_output_type = tf.uint8  # or tf.int8
            converter.experimental_new_quantizer = True # was True
            #converter.input_arrays = "input"
            #converter.output_arrays = "Identity"
            converter.allow_custom_ops = True
        if nms:
            converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

        tflite_model = converter.convert()
    
    if detector:
        converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        #converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if int8:
            #dataset = LoadImages(check_dataset(data)['train'], img_size=imgsz, auto=False)  # representative data
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.uint8  # or tf.int8
            converter.inference_output_type = tf.uint8  # or tf.int8
            converter.experimental_new_quantizer = True # was True
            converter.allow_custom_ops = True
            converter.input_arrays = "normalized_input_image_tensor"
            converter.output_arrays = "TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3"
        tflite_model = converter.convert()
    # Save the model
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)

    #run edgetpu compiler
    subprocess.run(["edgetpu_compiler", TFLITE_PATH, "-o", TFLITE_FOLDER, "-s"])


export_onnx_opt_openvino_tflite()
