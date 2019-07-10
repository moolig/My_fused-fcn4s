This is our project for Building Footprint Extraction from VHR Remote Sensing Images Combined with Normalized DSMs using Fused Fully Convolutional Networks.

Abstract

Automatic building extraction and delineation from high-resolution satellite imagery is an important but very challenging task, due to the extremely large diversity of building appearances. Nowadays, it is possible to use multiple high-resolution remote sensing data sources, which allow the integration of different information in order to improve the extraction accuracy of building outlines. Many algorithms are built on spectral-based or appearance-based criteria, from single or fused data sources, to perform the building footprint extraction. But the features for these algorithms are usually manually extracted, which limits their accuracy. Recently developed fully convolutional networks (FCNs), which are similar to normal convolutional neural networks (CNN), but the last fully connected layer is replaced by another convolution layer with a large “receptive field,” quickly became the state-of-the-art method for image recognition tasks, as they bring the possibility to perform dense pixelwise classification of input images. Based on these advantages, i.e., the automatic extraction of relevant features, and dense classification of images, we propose an end-to-end FCN, which effectively combines the spectral and height information from different data sources and automatically generates a full resolution binary building mask. Our architecture ( Fused-FCN4s ) consists of three parallel networks merged at a late stage, which helps propagating fine detailed information from earlier layers to higher levels, in order to produce an output with more accurate building outlines. The inputs to the proposed Fused-FCN4s are three-band (RGB) , panchromatic (PAN) , and normalized digital surface model (nDSM) images. Experimental results demonstrate that the fusion of several networks is able to achieve excellent results on complex data. Moreover, the developed model was successfully applied to different cities to show its generalization capacity. 

The project includes:
- network architecture for training, validation and test phases (train.prototxt, val.prototxt, deploy.prototxt)
- configuration file used to tell caffe how you want the network trained (solver.prototxt)
- main script to run the training (solve_fused_fcn4s.py)
- script to calculate the evaluation metrics (score.py)
- program to transfer weights from src_model to dst_model taken from original https://github.com/shelhamer/fcn.berkeleyvision.org
- snapshot folder with model parameters for proposed architecture

To start the training run the follow command: python solve_fused_fcn4s.py 0(GPU ID)