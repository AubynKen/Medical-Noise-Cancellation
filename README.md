# Medical-Noise-Cancellation
The project is a second year (equivalent to senior year of college in the US system / 1st year of Master in the UK 
system) school project of the CentraleSupelec engineering school in France. 
The goal of the project is to use data provided by General Electric Healthcare France (a partner of the Laboratory of 
signals and systems (L2S) of the CentraleSupelec engineering school) to achieve of proof-of-concept of denosing medical 
X-ray video returns of patients during operations with stents, and comparison of the performance obtained with the 
current state of the art of non-neural-network based image denoising techniques such as BM3D.

## Folder / module structures
- `archives`: contains the archived notebooks and models of the project
- `data`: contains the data used for the project
- `model`: contains the u-net model implemented in the project
- `result_images`: contains the results images of the UNet during training
- `evaluation`: contains functions for evaluating denoising performances

## Notebooks of the project root
In the order in which they are created:
- `data-augmentation.ipynb`: data augmentation with test and visualizations
- `unet-training.ipynb`: first steps of training of the UNet model using augmented data
- `unet-training-online.ipynb`: training of the UNet model using augmented data generated during training to avoid 
taking up too much space on the hard-drive.