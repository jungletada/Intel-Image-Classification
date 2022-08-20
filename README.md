# Intel Image Classification
--------------------------------
## Introduction
--------------------------------
* We purpose a Multi-scale ConvMixer in order to
alleviate the drawback of ConvMixer: low inference speed, 
no dimension reduction through the network.

* Compare to traditional ResNet, DenseNet, VGG16, Inception V4, and 
new purposed ViT, MLP-mixer, our model performs well in Intel dataset and reaches a comparable 
test accuracy and inference speed.

* In detail, we down-sample the feature map in original ConvMixer by MaxPooling,
and modify the "depth" and kernel size.

## Usage
--------------------------------
Available models: 
* MConvMixer
* resnet50
* densenet121
* vgg16 
* mixer 
* visionformer-small
* vit-base

## Requirements:
    pytorch, torchvision, timm

You can install `pytorch` and `torchvision` follow https://pytorch.org/
and install `timm` by `pip install timm`.

* To train the model, 

   
   `python main_run_model.py` 

* To make prediction on Seg_pred,

    `python main_pred.py`
--- 
You can use args to change default settings.
Dataset is stored in `./data`.

--------------------------------
This work is the contirbutions of our group member at Waseda University.
Code presented by Dingjie Peng
Presentation: Haoyuan Liu, Fei Bao, Yiming Sun, JiaHao Liu.
