# __Tutorial 1 : Classification__

QQuant provides multiple quantizers for quantization-aware fine-tuning. This tutorial provides several light demos, which are designed to introduce the overall style of QQuant and make you get started quickly. We assume that the reader has basic concepts of deep learning. 

QQuant currently supports PyTorch models. 

## __Structure of lowbit classification__

The directory structure of lowbit classification is as fllows:

```
├── demo                   //Jupyter DEMO 
├── doc                    //Tutorial
├── lbitcls                //Core Module
│   ├── apis               //Train,Test,Inference API
│   ├── core               //Eval, Fp16, and etc
│   ├── datasets           //Dataset and Dataloader
│   ├── __init__.py         
│   ├── models             //Models: Backbone, Neck, Loss, Head
│   ├── utils              //Tools
│   ├── VERSION            //Version Info
│   └── version.py
├── README.md
├── requirements           //Requirements
│   ├── build.txt
│   ├── docs.txt
│   ├── optional.txt
│   ├── readthedocs.txt
│   ├── runtime.txt
│   └── tests.txt
├── setup.py               //Install Python Script
├── thirdparty             //Thirdparty
│   └── configs            //Running Configure
├── tools
│   ├── dist_train.sh      //Distribution Training On Brain++
│   └── train.py           //Starting Training Script
└── work_dirs              //Your Working directory
    └── DSQ
```


<!--
 - __lbitcls__ contains core modules for training and evaluation like __datasets__ and __models__. For dataset format, we suggest to convert the data format into existing format(ImageNet). __Model__ module is disassembled into backbone, neck, classifier, loss, etc. To support new formats for datasets and models, you could define new classes under their corresponding directory.
-->

## __Turorial 1.1 : Inference with Quantization Model__

This section introduces how to convert a floating-point model to a quantization version and make inferences on a given image. 

### __Specify the model configuration file__

The configuration file determines all arguments related to the experiment, including model structure, dataset, quantization method, etc. These arguments are modularized so that we can customize them respectively.


```python
import mmcv
from lbitcls import __version__
config = 'config.py'
config = mmcv.Config.fromfile(config)
```


### __Build quantized model from the configuration file__

Building a network from scratch is usually a tedious process because it consists of several nested layers/modules. When quantifying an existing model, we will not repeat above steps. QQuant provides model transformer which recursively traverse the network structure and replace them with specified quantization layers. In addition to directly using the existing quantification methods, we could also modify them, or customize a new one.


```python
from thirdparty.mtransformer import build_mtransformer
from mmcv.runner import load_checkpoint
import warnings
from lbitcls.models import build_classifier

model = build_classifier(config.model)
# Quantize the floating-point model
if hasattr(config, "quant_transformer"):
    # Create a quantizer
    model_transformer = build_mtransformer(config.quant_transformer)
    # Quantize the floating-point model 
    model = model_transformer(model)
# Choose cpu or gpu device for inference 
device = 'cpu'
# Load the checkpoint
if config.load_from is not None:
    map_loc = 'cpu' if device == 'cpu' else None
    checkpoint = load_checkpoint(model, config.load_from, map_location=map_loc)
    if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        from lbitcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                        'meta data, use imagenet by default.')
        model.CLASSES = ImageNet.CLASSES
model.to(device)
model.cfg = config
model.eval()
```

### __Prepare data__ 

Before the data is fed to the network, a pipeline of pre-processing like cropping and normalization are required. These operations are recorded in the configuration file as a dict. In some cases, we may only want to use a single image for debugging instead of the entire dataset. The data pipeline and dataset loader are decoupled for better modularity.

```python
from lbitcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import numpy as np
import torch

img_file = './test.jpg'
cfg = model.cfg
data = dict(img_info=dict(filename=img_file), img_prefix=None)
# Extract the test transformation pipline from the config file
test_pipeline = Compose(cfg.data.test.pipeline)
# Transform raw data to the specified format
data = test_pipeline(data)
data = collate([data], samples_per_gpu=1)
device = next(model.parameters()).device  # model device
if next(model.parameters()).is_cuda:
    # scatter to specified GPU
    data = scatter(data, [device])[0]
```

### __Forward the quantized model__

Regardless of whether the model is quantified or not, the process of inference on images is unchanged, as same as other standard operation.

```python
with torch.no_grad():
    scores = model(return_loss=False, **data)
    pred_score = np.max(scores, axis=1)[0]
    pred_label = np.argmax(scores, axis=1)[0]
    result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
result['pred_class'] = model.CLASSES[result['pred_label']]
# Show the predicted results
print(result)
```



## __Turorial 1.2 : Start Training and Evaluation__

**QAT** (Quantization aware training) simulates the process of low-bit model training by inserting quantization nodes in some network modules. To practice a complete QAT approach, there are the following steps:

-  Confirm the configuration file
-  Train and test quantization model 
-  Analyze quantization performance

### __Confirm the config file__

If you just want to reproduce the existing quantization methods, you only need to learn the composition of the config file and be able to set it correctly. Generally, we don’t need to write a new config file from scratch, but inherit the existing standard template and modify some of the components. 

Next, we will establish a preliminary understanding about config file by reading a config example of Differentiable Soft Quantization (DSQ).


#### __Config Name__

The file path is: ```"./lowbit_classification/thirdparty/cls_quant_config/DSQ/res50/config2_res50_dsq_m1_16_2w2f.py"```.
The directory where the config file is located has a two-level structure, the first level is named by QAT method, and the second layer is named by backbone. The name of config file is determined by a naming rules:

```shell
config(number)_res50(backbone)_DSQ(qat_method)_m1(machine number)_16(sample_per_gpu)_2w2f(quant_bit).py
```

  - ```(number)```: experiment number.
  - ```(backbone)```: backbone type, like ```res50```, ```mobilenet```.
  - ```(qat_method)```: quantization method type like ```LSQ```, ```DSQ```, ```DoReFa```.
  - ```(machine number)```: machine number.
  - ```(sample_per_gpu)```:  batch size per GPU, like ```64```, ```128```.
  - ```(quant_bit)```: quantization bit of weights and activations, like ```2w2f```, ```4w4f```.

We selectively give components that may be frequently modified in the configuration file, see [MMClassification_Tutorial.ipynb](MMClassification_Tutorial.ipynb) for the complete source.


#### __Model Structure__

```python
model = dict(
    type='ImageClassifier',  # Type of classifier
    backbone=dict(
        type='ResNet',  #  Type of backbone
        depth=50,  # Depth of backbone
        num_stages=4,  # Number of stages of the backbone
        out_indices=(3, ),  # The index of output feature maps produced in each stages
        style='pytorch'),  # The style of backbone
    neck=dict(type='GlobalAveragePooling'),  # Type of neck of model 
    head=dict(
        type='LinearClsHead',  # Type of head used for classification
        num_classes=1000,  # The number of classes 
        in_channels=2048,  # Input channels for head
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),  # Type of loss for classification
        topk=(1, 5),  # Top-k accuracy
    ))
```


#### __Quantization Setting__
```python
quant_transformer = dict(  
    type = "mTransformerV2",  # Type of quantization transformer 
    quan_policy=dict(  
        Conv2d=dict(type='DSQConv',  # DSQ quant layer used to replace standard conv layer
        num_bit_w=2,  # Bit number of weight
        num_bit_a=2,  # Bit number of activation
        bSetQ=True),  # Switch of quantization
        Linear=dict(type='DSQLinear',  # DSQ quant layer used to replace standard linear layer
        num_bit_w=2,  # Bit number of weight
        num_bit_a=2)  # Bit number of activation
        ),
    special_layers = dict(  # Special layers that adopt different quant policy
        layers_name = [  # Names of special layers 
            'backbone.conv1',  
            'head.fc'],
        convert_type = [dict(
        type='DSQConv', # DSQ quant layer used to replace first conv layer of backbone
        num_bit_w=8, num_bit_a=8, bSetQ=True, quant_activation=False),
        dict(type='DSQLinear', # DSQ quant layer used to replace fc layer of head
        num_bit_w=8, num_bit_a=8)]
        )
)
```

#### __Optimizer__
```python
num_nodes = 1  # Number of machine
optimizer = dict(type='SGD',  # Type of optimizers
    lr=0.001 * num_nodes,  # Learning rate of optimizers
    momentum=0.9,  # Momentum
    weight_decay=0.0001)  # Weight decay 
optimizer_config = dict(grad_clip=None)  # Config used to build the optimizer hook
```

### __Train and test quantization model__

After completing the above step, all the details about the experiment have been determined. To train a model with this config file, we could simply run

```shell=
cd lowbit_classification
python tools/train.py  thirdparty/configs/DSQ/res50/config2_res50_dsq_m1_16_2w2f.py
```

Next, the program will output log information with the following format, which can be viewed under ```work_dir``` directory.

```shell
2021-07-21 21:21:52,381 - lbitcls - INFO - load checkpoint from /thirdparty/modelzoo/res50.pth
2021-07-21 21:21:52,382 - lbitcls - INFO - Use load_from_local loader
2021-07-21 21:21:52,959 - lbitcls - INFO - Start running, host: ***, work_dir: /data/workspace/lowbit_classification/workdirs/DSQ/res18/config2_res50_dsq_m1_16_2w2f
2021-07-21 21:21:52,959 - lbitcls - INFO - workflow: [('train', 1)], max: 100 epochs
2021-07-21 21:26:47,622 - lbitcls - INFO - Epoch [1][200/626]	lr: 1.000e-03, eta: 1 day, 1:32:11, time: 1.473, data_time: 0.449, memory: 10283, loss: 1.3465, top-1: 68.2227, top-5: 86.7495
2021-07-21 21:30:14,855 - lbitcls - INFO - Epoch [1][400/626]	lr: 1.000e-03, eta: 21:40:43, time: 1.036, data_time: 0.024, memory: 10283, loss: 1.3443, top-1: 68.2712, top-5: 86.7048
```

### __Analyze quantization performance__

QQuant provides model analysis API for further experiments. To explore more about the properties of the quantization model and how it differs from the standard model, we could use this script

```shell
python tools/model_analysis_tool.py \
    ${IMAGE_FILE} \
    ${FLOAT_CONFIG_FILE} \
    ${INT_CONFIG_FILE} \
    ${FLOAT_CHECKPOINT_FILE} \
    ${INT_CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--save-path ${HTML_SAVE_PATH}]
```

Examples:

```shell
python tools/model_analysis_tool.py \
    doc/tutorials/test.jpg \
    thirdparty/configs/DSQ/res50/config1_res50_dsq_m1_16_32w32f.py \
    thirdparty/configs/DSQ/res50/config2_res50_dsq_m1_16_2w2f.py \
    ./thirdparty/modelzoo/res50.pth \
    /data/workspace/lowbit_classification/workdirs/DSQ/res50/config2_res50_dsq_m1_16_2w2f/latest.pth \
    gpu:0 \
    ./model_analysis.html
```

 For the complete source, please refer [MMClassification_Tutorial.ipynb](MMClassification_Tutorial.ipynb).