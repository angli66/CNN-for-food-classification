import data
import model
import numpy as np
import matplotlib.pyplot as plt
import torch

###########################################################################
# Use the name from model.eval() to choose specify the layer to visualize #
###########################################################################
args = {'model': 'custom', 'pt_ft': '0'} # Change model name to visualize other architectures
model = model.get_model(args)
model.load_state_dict(torch.load("custom-checkpoint.pt"))
print(model.eval())

args = {'bz': 1, 'shuffle_data': False, 'normalization_mean': (0.485, 0.456, 0.406), 'normalization_std': (0.229, 0.224, 0.225)}
_, _, test_dataloader = data.get_dataloaders('food-101/train.csv', 'food-101/test.csv', args)
dataiter = iter(test_dataloader)
input, label = dataiter.next()

# Visualize feature maps
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.conv6.register_forward_hook(get_activation('vis'))
output = model(input)

act = activation['vis'].squeeze()
fig = plt.figure(figsize=(5, 5))
for i in range(4):
    ax = fig.add_subplot(2, 2, i+1)
    plt.imshow(act[i])
plt.suptitle("4 feature maps from custom model final cnn layer")
plt.tight_layout()
plt.show()