# Quick Start on Detection  Quantization Task
 
## Train a quantized detection model
you can quickly start a training with  one GPU by simply run comand below.

```bash
python tools/train.py thirdparty/configs/LSQ/config3_retinanet_LSQ_m2_2_3w3f.py
```



## Test and Inference

you can quickly start a test or inference  by simply run comand below.

```
python tools/test.py /data/code/4krjf_data/4k_json_coco/mmdetection/work_dirs/config5_atss_4krjf_float_mot/config5_atss_4krjf_float_mot_530.py /data/code/4krjf_data/4k_json_coco/mmdetection/work_dirs/config5_atss_4krjf_float_mot/latest.pth --out work_dirs/vis_res/config5_atss_4krjf_float_mot_530.pkl --eval bbox
```


When need to a more detailed usage, you can reference TuXXXXX


# Quick Start on Classification Task
 
## Train a quantized classification model
you can quickly start a training with one GPU by simply run comand below.

```bash
python tools/train.py  thirdparty/configs/DSQ/res50/config2_res50_dsq_m1_16_2w2f.py
```



## Test and Inference

you can quickly start a test or inference  by simply run comand below.

```
xxxxx
```


When need to a more detailed usage, you can reference TuXXXXX
