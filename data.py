import pickle
from numpy import genfromtxt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch

########## DO NOT change this function ##########
# If you change it to achieve better results, we will deduct points. 
def train_val_split(train_dataset):
    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], 
                                            generator=torch.Generator().manual_seed(42))
    return train_subset, val_subset
#################################################

########## DO NOT change this variable ##########
# If you change it to achieve better results, we will deduct points. 
transform_test = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
#################################################

class FoodDataset(Dataset):
    def __init__(self, data_csv, transforms=None):
        self.data = genfromtxt(data_csv, delimiter=',', dtype=str)
        self.transforms = transforms
        
    def __getitem__(self, index):
        fp, _, idx = self.data[index]
        idx = int(idx)
        img = Image.open(fp)
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, idx)

    def __len__(self):
        return len(self.data)

# Pass in csv path and transform, return created dataset object
def get_dataset(csv_path, transform):
    return FoodDataset(csv_path, transform)

# Helper function for get_dataloaders
def create_dataloaders(train_set, val_set, test_set, args=None):
    batch_size = args['bz'];
    shuffle_data = args['shuffle_data'];
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_data)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader

# Take train set csv data and test set csv data and generate dataloaders
def get_dataloaders(train_csv, test_csv, args=None):
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=args['normalization_mean'], std=args['normalization_std'])
    ])

    train_dataset = get_dataset(train_csv, transform)

########## DO NOT change the following two lines ##########
# If you change it to achieve better results, we will deduct points. 
    test_dataset = get_dataset(test_csv, transform_test)
    train_set, val_set = train_val_split(train_dataset)
###########################################################

    dataloaders = create_dataloaders(train_set, val_set, test_dataset, args)
    return dataloaders # In order of train_dataloader, val_dataloader, test_dataloader

def write_to_file(path, data):
    """
    Dumps pickled data into the specified relative path.

    Args:
        path: relative path to store to
        data: data to pickle and store
    """
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)