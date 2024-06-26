# Third-party imports
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np

# Standard library imports
import os


transformation = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


class ChestXrayDataset(Dataset):
    def __init__(self, type=None, root='data', classes=None):
        # Inizializzazione dei path
        path = ''
        if type == 'train':
            path += str(root.data_dir) + "train//"
        elif type == 'val':
            path += str(root.data_dir) + "val//"
        else:
            path += str(root.data_dir) + "test//"
        self.data = ImageFolder(path, transform=transformation)
        self.classes = self.data.classes
        self.targets = self.data.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data:
            image, label = self.data[idx]
            item = {
                'image' : image,
                'label' : label
            }
            return item
        else:
            return None


def class_count(dataset):
    elemets = np.zeros(len(dataset.classes))
    labels = dataset.targets
    for value in labels:
        if value == 0:
            elemets[0] += 1
        else:
            elemets[1] += 1
    return elemets


class LabelPreservingConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.targets = []
        self.classes = []
        for dataset in datasets:
            self.targets.extend(dataset.targets)
            self.classes.extend(dataset.classes)


class LabelPreservingSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.classes = dataset.classes
        self.targets = [dataset.targets[i] for i in indices]


def resize_datasets(train_dataset, val_dataset):
    # Li unisco in un unico dataset
    all_datasets = []
    all_datasets.append(train_dataset)
    all_datasets.append(val_dataset)
    # Li diviso secondo una dimensione del 80%, 10% e 10% ciascuno
    combined_dataset = LabelPreservingConcatDataset(all_datasets)
    #combined_dataset = torch.utils.data.ConcatDataset(all_datasets)
    train_size = int(0.9 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])
    train_dataset = LabelPreservingSubset(combined_dataset, train_dataset.indices)
    val_dataset = LabelPreservingSubset(combined_dataset, val_dataset.indices)
    return train_dataset, val_dataset
    
def print_shapes(type, train_dataset, val_dataset, test_dataset):
    train_class_counts = class_count(train_dataset)
    val_class_counts = class_count(val_dataset)
    test_class_counts = class_count(test_dataset)
    if type == 'before':
        print("Original datasets shapes:\n")
    else:
        print("Datasets shapes after resize:\n")
    print(f"Train size: {len(train_dataset)}")
    print(f"\t- Train normal dataset size: {int(train_class_counts[0])}")
    print(f"\t- Train pneumonia dataset size: {int(train_class_counts[1])}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"\t- Validation normal dataset size: {int(val_class_counts[0])}")
    print(f"\t- Validation pneumonia dataset size: {int(val_class_counts[1])}")
    print(f"Test size: {len(test_dataset)}")
    print(f"\t- Test normal dataset size: {int(test_class_counts[0])}")
    print(f"\t- Test pneumonia dataset size: {int(test_class_counts[1])}")

def load_datasets(config):
    train_dataset = ChestXrayDataset(type='train', root=config.data)
    val_dataset  = ChestXrayDataset(type='val', root=config.data)
    test_dataset  = ChestXrayDataset(type='test', root=config.data)
    # print some statistics before
    print_shapes('before', train_dataset, val_dataset, test_dataset)
    print("---------------------")
    # Ridistribuisci i datasets
    train_dataset, val_dataset = resize_datasets(train_dataset, val_dataset)
    # print some statistics after
    print_shapes('after', train_dataset, val_dataset, test_dataset)
    print("---------------------")
    return train_dataset, val_dataset, test_dataset

def verify_division(percorso):
    return os.path.isdir(percorso)

def binary_load(config):
    pneu_train = verify_division('dataset//chest_xray//train//PNEUMONIA')
    pneu_val = verify_division('dataset//chest_xray//val//PNEUMONIA')
    pneu_test = verify_division('dataset//chest_xray//test//PNEUMONIA')
    train_dataset, val_dataset, test_dataset = load_datasets(config)
    return train_dataset, val_dataset, test_dataset  
