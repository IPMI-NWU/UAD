# UDA_pytorch
Code and models for X-ray pulmonary parenchyma segmentation domain adaptation.

# Run:

`UNet_Models.py`: Contains multiple lung parenchyma segmentation models; this method uses the Attention U-Net.

`2_train.py`: Training process for the lung parenchyma segmentation model based on the source domain (MC).

`EdgeAtt_CycleGAN.py`: Implements the edge-guided X-ray lung parenchyma segmentation model, including the edge-guided style transfer process and the construction of generators and discriminators during the prediction style alignment process.

`EdgeAtt_train_ShenZhen.py`: Model training process on the ShenZhen target dataset.

`EdgeAtt_train_IPMI.py`: Model training process on our lab's private dataset.

# Test:

`3_inference.py`: Inference process code.


# Requirements
Some important required packages include:
* torch == 2.3.0
* torchvision == 0.18.0
* Python == 3.10.14
* numpy == 1.26.4