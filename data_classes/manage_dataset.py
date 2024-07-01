# Third-party imports
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt

# Local application/library specific imports
from utils import save_graph

# Standard library imports
import os
import shutil


transformation = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


class ChestXrayDataset(Dataset):
    def __init__(self, type=None, root='data', classes=None):
        path = ""
        path_removed = []
        # Inizializzazione dei path
        if type == 'train':
            path += str(root.data_dir) + "train//"
        elif type == 'val':
            path += str(root.data_dir) + "val//"
        else:
            path += str(root.data_dir) + "test//"

        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        for subfolder in subfolders:
            if not any(os.listdir(subfolder)):
                print(f"Directory path: '{subfolder}' is empty.")
                path_removed.append(subfolder)
                os.rmdir(subfolder)
        self.data = ImageFolder(path, transform=transformation)
        self.classes = self.data.classes
        self.targets = self.data.targets
        for element in path_removed:
            position = element.rfind('/')
            dir_name = element[position + 1:]
            if not os.path.exists(os.path.join(path, dir_name)):
                os.makedirs(str(path) + '//' + str(dir_name))
                self.classes.append(str(dir_name))
                self.targets.append(0)

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
        elif value == 1:
            elemets[1] += 1
        else:
            elemets[2] += 1
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
    
def print_shapes(type, train_dataset, val_dataset, test_dataset, number_class):
    train_class_counts = class_count(train_dataset)
    val_class_counts = class_count(val_dataset)
    test_class_counts = class_count(test_dataset)
    if type == 'before':
        print("Original datasets shapes:\n")
    else:
        print("Datasets shapes after resize:\n")
    if number_class == 'binary':
        print(f"Train size: {len(train_dataset)}")
        print(f"\t- Train normal dataset size: {int(train_class_counts[0])}")
        print(f"\t- Train pneumonia dataset size: {int(train_class_counts[1])}")
        print(f"Validation size: {len(val_dataset)}")
        print(f"\t- Validation normal dataset size: {int(val_class_counts[0])}")
        print(f"\t- Validation pneumonia dataset size: {int(val_class_counts[1])}")
        print(f"Test size: {len(test_dataset)}")
        print(f"\t- Test normal dataset size: {int(test_class_counts[0])}")
        print(f"\t- Test pneumonia dataset size: {int(test_class_counts[1])}")
    else:
        print(f"Train size: {len(train_dataset)}")
        print(f"\t- Train bacteria dataset size: {int(train_class_counts[0])}")
        print(f"\t- Train normal dataset size: {int(train_class_counts[1])}")
        print(f"\t- Train virus dataset size: {int(train_class_counts[2])}")
        print(f"Validation size: {len(val_dataset)}")
        print(f"\t- Validation bacteria dataset size: {int(val_class_counts[0])}")
        print(f"\t- Validation normal dataset size: {int(val_class_counts[1])}")
        print(f"\t- Validation virus dataset size: {int(val_class_counts[2])}")
        print(f"Test size: {len(test_dataset)}")
        print(f"\t- Test bacteria dataset size: {int(test_class_counts[0])}")
        print(f"\t- Test normal dataset size: {int(test_class_counts[1])}")
        print(f"\t- Test virus dataset size: {int(test_class_counts[2])}")
    
def path_file_binary(path):
    file_list = []
    create_directory_ternary(path)
    file_list.append(os.listdir(os.path.join(path, 'BACTERIA')))
    file_list.append(os.listdir(os.path.join(path, 'VIRUS')))
    return file_list

def create_directory_binary(path):
    if not os.path.exists(os.path.join(path, 'PNEUMONIA')):
        os.makedirs(os.path.join(path, 'PNEUMONIA'))
        
def split_binary_directory():
    file_list = []    
    path = ['dataset//chest_xray//train//', 'dataset//chest_xray//val//', 'dataset//chest_xray//test//']
    create_directory_binary(path[0])
    file_list.append(path_file_binary(path[0]))
    for elements in file_list[0]:
        for element in elements:
            if 'bacteria' in element:
                shutil.move(os.path.join(path[0], 'BACTERIA', element), os.path.join(path[0], 'PNEUMONIA', element))
            else:
                shutil.move(os.path.join(path[0], 'VIRUS', element), os.path.join(path[0], 'PNEUMONIA', element))
    os.rmdir(str(path[0]) + 'BACTERIA')
    os.rmdir(str(path[0]) + 'VIRUS')
    
    create_directory_binary(path[1])
    file_list.append(path_file_binary(path[1]))
    for elements in file_list[1]:
        for element in elements:
            if 'bacteria' in element:
                shutil.move(os.path.join(path[1], 'BACTERIA', element), os.path.join(path[1], 'PNEUMONIA', element))
            else:
                shutil.move(os.path.join(path[1], 'VIRUS', element), os.path.join(path[1], 'PNEUMONIA', element))
    os.rmdir(str(path[1]) + 'BACTERIA')
    os.rmdir(str(path[1]) + 'VIRUS')
    
    create_directory_binary(path[2])
    file_list.append(path_file_binary(path[2]))
    for elements in file_list[2]:
        for element in elements:
            if 'bacteria' in element:
                shutil.move(os.path.join(path[2], 'BACTERIA', element), os.path.join(path[2], 'PNEUMONIA', element))
            else:
                shutil.move(os.path.join(path[2], 'VIRUS', element), os.path.join(path[2], 'PNEUMONIA', element))
    os.rmdir(str(path[2]) + 'BACTERIA')
    os.rmdir(str(path[2]) + 'VIRUS')
    
def path_file_ternary(path):
    create_directory_binary(path)
    file_list = os.listdir(os.path.join(path, 'PNEUMONIA'))
    return file_list
    
def create_directory_ternary(path):
    if not os.path.exists(os.path.join(path, 'BACTERIA')):
        os.makedirs(os.path.join(path, 'BACTERIA'))
    if not os.path.exists(os.path.join(path, 'VIRUS')):
        os.makedirs(os.path.join(path, 'VIRUS'))

def split_ternary_directory():
    file_list = []    
    path = ['dataset//chest_xray//train//', 'dataset//chest_xray//val//', 'dataset//chest_xray//test//']
    create_directory_ternary(path[0])
    file_list.append(path_file_ternary(path[0]))
    for element in file_list[0]:
        if 'bacteria' in element:
            shutil.move(os.path.join(path[0], 'PNEUMONIA', element), os.path.join(path[0], 'BACTERIA', element))
        else:
            shutil.move(os.path.join(path[0], 'PNEUMONIA', element), os.path.join(path[0], 'VIRUS', element))
    os.rmdir(str(path[0]) + 'PNEUMONIA')
    
    create_directory_ternary(path[1])
    file_list.append(path_file_ternary(path[1]))
    for element in file_list[1]:
        if 'bacteria' in element:
            shutil.move(os.path.join(path[1], 'PNEUMONIA', element), os.path.join(path[1], 'BACTERIA', element))
        else:
            shutil.move(os.path.join(path[1], 'PNEUMONIA', element), os.path.join(path[1], 'VIRUS', element))
    os.rmdir(str(path[1]) + 'PNEUMONIA')
    
    create_directory_ternary(path[2])
    file_list.append(path_file_ternary(path[2]))
    for element in file_list[2]:
        if 'bacteria' in element:
            shutil.move(os.path.join(path[2], 'PNEUMONIA', element), os.path.join(path[2], 'BACTERIA', element))
        else:
            shutil.move(os.path.join(path[2], 'PNEUMONIA', element), os.path.join(path[2], 'VIRUS', element))
    os.rmdir(str(path[2]) + 'PNEUMONIA')
    
def visualize_class_distribution(dataset, dataset_name, view, resize):
    class_counts = np.zeros(len(dataset.classes))
    class_counts = class_count(dataset)
    plt.bar(dataset.classes, class_counts, color=['blue', 'orange', 'red', 'yellow', 'purple'])
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.title(f"Class distribution in the {dataset_name.lower()} dataset")
    if not resize:
        dataset_name += '_original'
        save_graph(dataset_name, 'Dataset_original')
    else:
        dataset_name += '_resized'
        save_graph(dataset_name, 'Dataset_resized')
    if view:
        plt.show()
    plt.close()
    
def print_dataset_graph(train_dataset, val_dataset, test_dataset, view, resize):
    print("Drawing graph for class distribution in dataset...")
    visualize_class_distribution(train_dataset, "Train", view, resize)
    visualize_class_distribution(val_dataset, "Validation", view, resize)
    visualize_class_distribution(test_dataset, "Test", view, resize)

def load_datasets(config):
    train_dataset = ChestXrayDataset(type='train', root=config.data)
    val_dataset  = ChestXrayDataset(type='val', root=config.data)
    test_dataset  = ChestXrayDataset(type='test', root=config.data)
    # print some statistics before
    print_shapes('before', train_dataset, val_dataset, test_dataset, config.classification.type)
    if config.graph.create_dataset_graph:
        print_dataset_graph(train_dataset, val_dataset, test_dataset, config.graph.view_dataset_graph, resize=False)
    print("---------------------")
    # Ridistribuisci i datasets
    train_dataset, val_dataset = resize_datasets(train_dataset, val_dataset)
    # print some statistics after
    print_shapes('after', train_dataset, val_dataset, test_dataset, config.classification.type)
    if config.graph.create_dataset_graph:
        print_dataset_graph(train_dataset, val_dataset, test_dataset, config.graph.view_dataset_graph, resize=True)
    print("---------------------")
    return train_dataset, val_dataset, test_dataset

def verify_division(percorso):
    return os.path.isdir(percorso)

def binary_load(config):
    pneu_train = verify_division('dataset//chest_xray//train//PNEUMONIA')
    pneu_val = verify_division('dataset//chest_xray//val//PNEUMONIA')
    pneu_test = verify_division('dataset//chest_xray//test//PNEUMONIA')
    if not (pneu_train and pneu_val and pneu_test):
        print("Transform dataset to binary...")
        split_binary_directory()
    train_dataset, val_dataset, test_dataset = load_datasets(config)
    return train_dataset, val_dataset, test_dataset  

def ternary_load(config):
    bact_train = verify_division('dataset//chest_xray//train//BACTERIA')
    vir_train = verify_division('dataset//chest_xray//train//VIRUS')
    bact_val = verify_division('dataset//chest_xray//val//BACTERIA')
    vir_val = verify_division('dataset//chest_xray//val//VIRUS')
    bact_test = verify_division('dataset//chest_xray//test//BACTERIA')
    vir_test = verify_division('dataset//chest_xray//test//VIRUS')
    if not (bact_train and vir_train and bact_val and vir_val and bact_test and vir_test):
        print("Transform dataset to ternary...")
        split_ternary_directory()
    train_dataset, val_dataset, test_dataset = load_datasets(config)
    return train_dataset, val_dataset, test_dataset
