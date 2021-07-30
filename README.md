# Welcome to QuantQuant

 &nbsp; &nbsp;**Quant_Quant** (QQ for short) is a lightweight but powerful codebase for quantization aware training(QAT). In this codebase, wo have covered both detection and classification tasks. The codebase integrate majority of  QAT methods.
## Key Features & Capabilty
  - Completeness codebase
   Major of QAT algorithm have been covered in QQ, including [Uniform](https://arxiv.org/abs/1909.13144)/[Dorefa](https://arxiv.org/abs/1606.06160)/[LSQ](https://arxiv.org/abs/1902.08153)/[DSQ](https://arxiv.org/pdf/1908.05033v1.pdf)/[APOT](https://arxiv.org/abs/1909.13144)/[LSQ+](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Bhalgat_LSQ_Improving_Low-Bit_Quantization_Through_Learnable_Offsets_and_Better_Initialization_CVPRW_2020_paper.html)
  - Authoritative exprimental results
  Provide realiable code and reproducible results for state-of-the-art QAT methods.
  - Easy to use and extend
  We define an  code orginazation form which is easy to get started and friendly to customize and conduct own experiments.
  - Effcient training
  Considring training efficiency, our code support multi-machine parallel training.
  
   
## Who should consider use Quant_Quant
-  Researchers
   - For beginners， the one who are interested in quantization algorithm and just want to try it easily.
   - For senior researchers, the one who are familar with this science field and want to  conductive some academic expriments.
 

-  Practitioner of Microchip Designer
   - the one who want to follow the research progress in QAT.
   - the one who want to deploy models in different chips easily.


## The capabilities of QQ
QQ is a simple but strong codebase for quantization aware training. Different form nowadays pubic QAT codebase, we define a module named QQTransformer which convert float operations to  int operation according to the quantization method specified in the config file. This feature makes it easy to expriment on self-defined methods or backbone, for interating quantization methods and backebone are relatively splited.

What’s more,  we have support to quantize on both classfication and detection tasks, which meets majority of researchers requriments. Please take a  glance on the network we have completed.


<table>
    <tr>
        <td> <b>Task</b> </td> 
        <td> <b>Dataset</b> </td> 
        <td><b>BackBone</b></td> 
   </tr>
    <tr>
      	 <td colspan="3"> <b>Quant_Method:</b> <br>   &nbsp; &nbsp;  &nbsp;   Uniform/LSQ/DSQ/APOT/LSQ+</td>    
    </tr>
    <tr>
        <td><p align="left"><b>Classification</b></p></td> 
        <td>ImageNet <br>Cifar-10<br>Cifar-100  </td> 
        <td><a href="https://arxiv.org/abs/1801.04381" target="_blank" rel="noopener noreferrer">Mobilenet_V2</a><br> <a href="https://arxiv.org/abs/2003.13678" target="_blank" rel="noopener noreferrer">RegNet</a><br> <a href="https://arxiv.org/abs/1512.03385" target="_blank" rel="noopener noreferrer">ResNet</a> <br>  <a href=" https://arxiv.org/abs/1611.05431" target="_blank" rel="noopener noreferrer">ResNext</a><br><a href="https://arxiv.org/abs/1709.01507" target="_blank" rel="noopener noreferrer">SEResNeXt</a> <br><a href=" https://arxiv.org/abs/1707.01083" target="_blank" rel="noopener noreferrer">ShuffleNet_v1</a> <br><a href=" https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html" target="_blank" rel="noopener noreferrer">ShuffleNet_v2</a> <br><a href="https://arxiv.org/abs/1409.1556" target="_blank" rel="noopener noreferrer">VGG</a><br></td> 
   </tr> 
        <tr>
        <td><b>Detection</b></td> 
        <td>MSCOCO </td> 
        <td><a href="https://arxiv.org/abs/1708.02002" target="_blank" rel="noopener noreferrer">RetinaNet</a> <br><a href="https://arxiv.org/abs/1804.02767" target="_blank" rel="noopener noreferrer">YOLO_V3</a><br> <a href="https://arxiv.org/abs/1912.02424" target="_blank" rel="noopener noreferrer">ATSS</a></td> 
   </tr> 
</table>


 
## The Structure of QQ 
```
├── ModelAnalysisTools
├── QuanTransformer
├── QuantMmdetection
├── lowbit_classification
├── requirements.txt
└── README.md
```

```QuantQuant:``` is origanized as shown above.  As shown, QuanTransformer is the most important part, in which we define the quantization methods and how to convert the  float operators to the quantized one. Classfication and detection are implemented sperataly.
- ```ModelAnalysisTools:``` intergerate a set of model analysis tools to analysis model performance.
- ```QuanTransformer``` complete QAT algorithm cited most, including uniform, lsq, dsq, dorefa, apot. And we will keep abreast of developments in this area and update our code continuously.
- ```lowbit_classification:``` In classification module, wo have completed lots of classification backbone in it, which totally satisty the purpose on both academical research  and industry deploy. 
- ```QuantMmdetection:```  Nowadays,  the quantization in detection ares has gained more and more attention. But, there still has little paper studied in this problem.  Consistering detection is a significant part of computer vision, we want to construct a baseline for this problem. What’s more, we also want to attract more on this problem. 

![image](https://user-images.githubusercontent.com/31733191/127477981-8cec4ef4-fec1-476c-8474-a1f5fe16135f.png)

## Installation  Documentation



###  How can we start a experiment on detection?

see details in Quick Start on  Classification Task.md and Quick Start on  Detection Task.md



## Getting Help

To start, you can check the documents in docs. If you couldn'd find help there, try search Github issues and we are very pleased to fix the bugs and response to your problem. 

## Update

- 8/2021: Release  QuantQuant v0.9 

## License

QuantQuant is released under the [Apache 2.0 license](https://github.com/ProhetTeam/QuantQuant/blob/master/LICENSE).


## Citing QuantQuant

If you use QuantQuant model in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```
@misc{,
  author =       {Tanfeiyang and Xianyuhaishu and Zhangyi and Zhoujianli and Likeyu},
  title =        {QuantQuant},
  howpublished = {\url{https://github.com/ProhetTeam/QuantQuant}},
  year =         {2018}
}
```
