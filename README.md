[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7006074&assignment_repo_type=AssignmentRepo)
# CSE 151B PA3: CNN for Food Images Classification

Contributers:
Ang Li, Fenghao Yang, Yikai Mao, Jinghao Miao

## Get Data
First run `get_data.sh` to get Food 101 dataset


## Run

To run the code, run the ```main.py``` file and there are arguments that user can input for hyperparmeters, which includes:
- device_id (The id of the gpu to use; ```type=int```; ```default=0```)
- model (Model being used including baseline, custom, resnet18, and vgg16; ```type=str```; ```default='custom'```)
- pt_ft (Whether model is for partial fine-tune model; ```type=int```; ```default=1```)
- bz (Batch size; ```type=int```; ```default=32```)
- shuffle_data (Whether shuffle the data; ```type=int```; ```default=1```)
- normalization_mean (Mean value of z-scoring normalization for each channel in image; ```type=tuple```; ```default=(0.485, 0.456, 0.406)```)
- normalization_std (Standard deviation of z-scoring normalization for each channel in image; ```type=tuple```; ```default=(0.229, 0.224, 0.225)```)
- epoch (Number of epoch; ```type=int```; ```default=30```)
- criterion (Which loss function to use; ```type=str```; ```default='cross_entropy'```)
- optimizer (Which optimizer to use; ```type=str```; ```default='adam'```)
- lr (Learning rate; ```type=float```; ```default=1e-4```)
- weight_decay (weight decay; ```type=float```; ```default=1e-4```)
- lr_scheduling (Whether enable learning rate scheduling; ```type=int```; ```default=0```)
- lr_scheduler (Learning rate scheduler; ```type=str```; ```default='steplr'```)
- step_size (Period of learning rate decay; ```type=int```; ```default=7```)
- gamma (Multiplicative factor of learning rate decay; ```type=float```; ```default=0.1```)


Directly run the code with
```bash
python3 main.py
```
will train the and test the performance of baseline model. The results will be saved in ```results.pkl``` after training is finished, and can be used by ```visualization.ipynb``` to visualize the loss/accuracy graph.

## Files
- ```main.py```: file for the entire code
- ```prepare_data.py```: file to load the dataset
- ```data.py```: file to pre-processe data, split train, validation and test set, create dataloader
- ```model.py```: file with implementation of baseline, custom, resnet-18, vgg-16 model
- ```engine.py```: file to prepare, train, test model, and save the results to ```results.pkl```
- ```visualization.ipynb```: notebook to plot graphs, visualize weight maps and feature maps
