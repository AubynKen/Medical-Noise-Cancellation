# Medical-Noise-Cancellation
The project is a second year engineering school (equivalent to senior year of college in the US system / 1st year of Master in the UK 
system) project of CentraleSupelec engineering school (Paris-Saclay University) in France. 

The goal of the project is to use data provided by General Electric Healthcare France (a partner of the Laboratory of 
signals and systems (L2S) of the CentraleSupelec engineering school) to achieve of proof-of-concept of denosing medical 
X-ray video returns of patients during operations with stents, and comparison of the performance obtained with the 
current state of the art of non-neural-network based image denoising techniques such as BM3D.

## Folder / module structures
- `archives`: contains the archived notebooks and models of the project
- `data`: contains the input-output image pairs used for training the models
- `unet`: contains our custom UNet-like model
- `runs`: contains evaluation records during training on Google Colab
- `result_images`: contains the results images of the UNet during training that we've saved for illustration purposes
- `evaluation`: contains functions for evaluating denoising performances such as peak to noise ratio

## Notebooks of the project root
In the order in which they are created:
- `data-augmentation.ipynb`: visualization notebook of our data-augmentation pipeline for illustration purposes
- `unet-training.ipynb`: first steps of training of the UNet model using augmented data
- `unet-training-online.ipynb`: second version of the training notebook using augmented images generated on-the-go using our custom Pytorch Dataset object
- `project-advancements-fr.md`: project advancements and meeting reports written in French

*note: We have chosen to keep some of the archived files on the main branch instead of deleting them and accessing 
them through commit history, because this repository is intended to be accessed by students of the following years
working in follow-up or similar projects of the L2S as a reference project repository.*
