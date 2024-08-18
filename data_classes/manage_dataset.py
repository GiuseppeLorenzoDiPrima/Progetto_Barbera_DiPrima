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


# Transformation to apply to the dataset divided according to training, validation and testing
transformation = {
    'training' : transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation(10),
            transforms.RandomAffine(translate=(0.1, 0.05), degrees=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    'validation' : transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    'testing' : transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
}


# Class that defines the dataset
class ChestXrayDataset(Dataset):
    """
    A Dataset for chest X-ray images.
    """
    def __init__(self, type=None, root='data', classes=None):
        """
        A Dataset for chest X-ray images.

        :param type: The type of the dataset (e.g., 'train', 'val', 'test').
        :type type: str
        :param root: The root directory of the dataset.
        :type root: str
        :param classes: The classes in the dataset.
        :type classes: list
        """
        # Initializing the variable
        path = ""
        path_removed = []
        # Initializing the path
        if type == 'train':
            path += str(root.data_dir) + "train//"
        elif type == 'val':
            path += str(root.data_dir) + "val//"
        else:
            path += str(root.data_dir) + "test//"
        # Checks if there are any empty subfolders, stores their name in path_removed and removes them
        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        for subfolder in subfolders:
            if not any(os.listdir(subfolder)):
                print(f"Directory path: '{subfolder}' is empty.")
                path_removed.append(subfolder)
                os.rmdir(subfolder)
        # Retrieve data through the ImageFolder class
        if type == 'train':
            self.data = ImageFolder(path, transform=transformation['training'])
        elif type == 'val':
            self.data = ImageFolder(path, transform=transformation['validation'])
        else:
            self.data = ImageFolder(path, transform=transformation['testing'])
        # Sets variables
        self.classes = self.data.classes
        self.targets = self.data.targets
        self.path = path
        # Restore previously removed empty subfolders
        for element in path_removed:
            position = element.rfind('/')
            dir_name = element[position + 1:]
            if not os.path.exists(os.path.join(path, dir_name)):
                os.makedirs(str(path) + '//' + str(dir_name))
                self.classes.append(str(dir_name))
                self.targets.append(0)

    # Return the size of the dataset
    def __len__(self):
        return len(self.data)

    # Return image and label elements via the item dictionary 
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


# Count how many items are contained for each class in the dataset
def class_count(dataset):
    """
    Counts the number of instances of each class in a dataset.

    :param dataset: The dataset.
    :type dataset: Dataset
    :return: Returns a numpy array containing the counts of each class.
    :rtype: numpy.ndarray
    """
    # Initialize an array of the same size as the classes to zero
    elemets = np.zeros(len(dataset.classes))
    # Retrieve the labels
    labels = dataset.targets
    # Increase the counter for the corresponding class by one
    for value in labels:
        if value == 0:
            elemets[0] += 1
        elif value == 1:
            elemets[1] += 1
        else:
            elemets[2] += 1
    # Returns a vector containing the number of elements for each class
    return elemets

# A class used to combine a list of datasets into a single dataset mantaining targets, classes and path attributes
class LabelPreservingConcatDataset(torch.utils.data.ConcatDataset):
    """
    A ConcatDataset that preserves the labels of the original datasets.
    """
    def __init__(self, datasets):
        """
        A ConcatDataset that preserves the labels of the original datasets.

        :param datasets: The datasets to concatenate.
        :type datasets: list
        """
        super().__init__(datasets)
        # Initializing arrays and string
        self.targets = []
        self.classes = []
        self.path = ''
        # For each dataset provided, retrieve targets, classes and path
        for dataset in datasets:
            self.targets.extend(dataset.targets)
            self.classes = dataset.classes
            self.path = dataset.path


# A class used to keep classes and targets attributes in individual subsets
class LabelPreservingSubset(torch.utils.data.Subset):
    """
    A Subset that preserves the labels of the original dataset.
    """
    def __init__(self, dataset, indices):
        """
        A Subset that preserves the labels of the original dataset.

        :param dataset: The original dataset.
        :type dataset: Dataset
        :param indices: The indices of the subset.
        :type indices: list
        """
        super().__init__(dataset, indices)
        self.classes = dataset.classes
        self.targets = [dataset.targets[i] for i in indices]
        self.path = dataset.path


# Redistribute train set and validation set data
def resize_datasets(train_dataset, val_dataset, split_percentage):
    """
    Resizes the datasets to have a split_percentage% and (100 - split_percentage)% split for training and validation.

    :param train_dataset: The training dataset.
    :type train_dataset: Dataset
    :param val_dataset: The validation dataset.
    :type val_dataset: Dataset
    :param split_percentage: The percentage of the combined dataset to be used for training.
    :type split_percentage: float
    :return: Returns the resized train and validation datasets.
    :rtype: tuple (Dataset, Dataset)
    """
    # Initializing an Array
    all_datasets = []
    # Insert the train and val datasets into the array
    all_datasets.append(train_dataset)
    all_datasets.append(val_dataset)
    # Combine datasets
    combined_dataset = LabelPreservingConcatDataset(all_datasets)
    # It distributes 90% to the train set and 10% to the validation set
    train_size = int(split_percentage * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])
    train_dataset = LabelPreservingSubset(combined_dataset, train_dataset.indices)
    val_dataset = LabelPreservingSubset(combined_dataset, val_dataset.indices)
    # Returns the new train set and validation set
    return train_dataset, val_dataset
    
# Print the size of the dataset
def print_shapes(type, train_dataset, val_dataset, test_dataset, number_class):
    """
    Prints the shapes of the datasets.

    :param type: The type of the datasets (e.g., 'before', 'after').
    :type type: str
    :param train_dataset: The training dataset.
    :type train_dataset: Dataset
    :param val_dataset: The validation dataset.
    :type val_dataset: Dataset
    :param test_dataset: The test dataset.
    :type test_dataset: Dataset
    :param number_class: The number of classes (e.g., 'binary', 'ternary').
    :type number_class: str
    """
    # Count classes for each dataset
    train_class_counts = class_count(train_dataset)
    val_class_counts = class_count(val_dataset)
    test_class_counts = class_count(test_dataset)
    # Check that you are printing before or after resize to adjust the print
    if type == 'before':
        print("Original datasets shapes:\n")
    else:
        print("Datasets shapes after resize:\n")
    # Print binary dataset
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
    # Print ternary dataset
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
    
# Returns files that are in the BACTERIA and VIRUS folders
def path_file_binary(path):
    """
    Returns a list of files in the 'BACTERIA' and 'VIRUS' directories.

    :param path: The path to the directory.
    :type path: str
    :return: Returns a list of files in the 'BACTERIA' and 'VIRUS' directories.
    :rtype: list
    """
    # Initialize an array
    file_list = []
    # Verify that the BACTERIA and VIRUS folders exist otherwise it creates them
    create_directory_ternary(path)
    # Add files in the BACTERIA and VIRUS folders to the list
    file_list.append(os.listdir(os.path.join(path, 'BACTERIA')))
    file_list.append(os.listdir(os.path.join(path, 'VIRUS')))
    # Return the list
    return file_list

# Verify that the PNEUMONIA folder exists otherwise it creates it
def create_directory_binary(path):
    """
    Creates a 'PNEUMONIA' directory at the given path.

    :param path: The path to create the directory at.
    :type path: str
    """
    if not os.path.exists(os.path.join(path, 'PNEUMONIA')):
        os.makedirs(os.path.join(path, 'PNEUMONIA'))
        
# Splits files for binary classification
def split_binary_directory(data_path):
    """
    Splits the directory into a binary structure by moving 'BACTERIA' and 'VIRUS' files into a 'PNEUMONIA' directory.

    :param data_path: The path to the dataset directory.
    :type data_path: str
    """
    # Initialize file_list and an array with paths
    file_list = []    
    path = [data_path + 'train//', data_path + 'val//', data_path + 'test//']
    # Check if the train PNEUMONIA folder exists, otherwise create it
    create_directory_binary(path[0])
    # Adds files found in the train BACTERIA and VIRUS folders to the list
    file_list.append(path_file_binary(path[0]))
    # Move file_list items from the BACTERIA and VIRUS folders to the PNEUMONIA folder
    for elements in file_list[0]:
        for element in elements:
            if 'bacteria' in element:
                shutil.move(os.path.join(path[0], 'BACTERIA', element), os.path.join(path[0], 'PNEUMONIA', element))
            else:
                shutil.move(os.path.join(path[0], 'VIRUS', element), os.path.join(path[0], 'PNEUMONIA', element))
    # Removes BACTERIA and VIRUS folders
    os.rmdir(str(path[0]) + 'BACTERIA')
    os.rmdir(str(path[0]) + 'VIRUS')
    # Check if the validation PNEUMONIA folder exists, otherwise create it
    create_directory_binary(path[1])
    # Adds files found in the validation BACTERIA and VIRUS folders to the list
    file_list.append(path_file_binary(path[1]))
    # Move file_list items from the BACTERIA and VIRUS folders to the PNEUMONIA folder
    for elements in file_list[1]:
        for element in elements:
            if 'bacteria' in element:
                shutil.move(os.path.join(path[1], 'BACTERIA', element), os.path.join(path[1], 'PNEUMONIA', element))
            else:
                shutil.move(os.path.join(path[1], 'VIRUS', element), os.path.join(path[1], 'PNEUMONIA', element))
    # Removes BACTERIA and VIRUS folders
    os.rmdir(str(path[1]) + 'BACTERIA')
    os.rmdir(str(path[1]) + 'VIRUS')
    # Check if the test PNEUMONIA folder exists, otherwise create it
    create_directory_binary(path[2])
    # Adds files found in the test BACTERIA and VIRUS folders to the list
    file_list.append(path_file_binary(path[2]))
    # Move file_list items from the BACTERIA and VIRUS folders to the PNEUMONIA folder
    for elements in file_list[2]:
        for element in elements:
            if 'bacteria' in element:
                shutil.move(os.path.join(path[2], 'BACTERIA', element), os.path.join(path[2], 'PNEUMONIA', element))
            else:
                shutil.move(os.path.join(path[2], 'VIRUS', element), os.path.join(path[2], 'PNEUMONIA', element))
    # Removes BACTERIA and VIRUS folders
    os.rmdir(str(path[2]) + 'BACTERIA')
    os.rmdir(str(path[2]) + 'VIRUS')
    
# Returns the files contained in the PNEUMONIA folder
def path_file_ternary(path):
    """
    Returns a list of files in the 'PNEUMONIA' directory.

    :param path: The path to the directory.
    :type path: str
    :return: Returns a list of files in the 'PNEUMONIA' directory.
    :rtype: list
    """
    # Check if the PNEUMONIA folder exists, otherwise create it
    create_directory_binary(path)
    # Add files in the PNEUMONIA folder to the list
    file_list = os.listdir(os.path.join(path, 'PNEUMONIA'))
    # Return the list
    return file_list

# Verify that the BACTERIA and VIRUS folders exist otherwise it creates them
def create_directory_ternary(path):
    """
    Creates 'BACTERIA' and 'VIRUS' directories at the given path.

    :param path: The path to create the directories at.
    :type path: str
    """
    if not os.path.exists(os.path.join(path, 'BACTERIA')):
        os.makedirs(os.path.join(path, 'BACTERIA'))
    if not os.path.exists(os.path.join(path, 'VIRUS')):
        os.makedirs(os.path.join(path, 'VIRUS'))

# Splits files for ternary classification
def split_ternary_directory(data_path):
    """
    Splits the directory into a ternary structure by moving files from the 'PNEUMONIA' directory to 'BACTERIA' and 'VIRUS' directories.

    :param data_path: The path to the dataset directory.
    :type data_path: str
    """
    # Initialize file_list and an array with paths
    file_list = []    
    path = [data_path + 'train//', data_path + 'val//', data_path + 'test//']
    # Check if the train BACTERIA and VIRUS folder exist, otherwise create them
    create_directory_ternary(path[0])
    # Adds files found in the train PNEUMONIA folder to the list
    file_list.append(path_file_ternary(path[0]))
    # Move file_list items from the PNEUMONIA folder to the BACTERIA and VIRUS folders based on filenames
    for element in file_list[0]:
        if 'bacteria' in element:
            shutil.move(os.path.join(path[0], 'PNEUMONIA', element), os.path.join(path[0], 'BACTERIA', element))
        else:
            shutil.move(os.path.join(path[0], 'PNEUMONIA', element), os.path.join(path[0], 'VIRUS', element))
    # Removes PNEUMONIA folder
    os.rmdir(str(path[0]) + 'PNEUMONIA')
    # Check if the validation BACTERIA and VIRUS folder exist, otherwise create them
    create_directory_ternary(path[1])
    # Adds files found in the validation PNEUMONIA folder to the list
    file_list.append(path_file_ternary(path[1]))
    # Move file_list items from the PNEUMONIA folder to the BACTERIA and VIRUS folders based on filenames
    for element in file_list[1]:
        if 'bacteria' in element:
            shutil.move(os.path.join(path[1], 'PNEUMONIA', element), os.path.join(path[1], 'BACTERIA', element))
        else:
            shutil.move(os.path.join(path[1], 'PNEUMONIA', element), os.path.join(path[1], 'VIRUS', element))
    # Removes PNEUMONIA folder
    os.rmdir(str(path[1]) + 'PNEUMONIA')
    # Check if the test BACTERIA and VIRUS folder exist, otherwise create them
    create_directory_ternary(path[2])
    # Adds files found in the test PNEUMONIA folder to the list
    file_list.append(path_file_ternary(path[2]))
    # Move file_list items from the PNEUMONIA folder to the BACTERIA and VIRUS folders based on filenames
    for element in file_list[2]:
        if 'bacteria' in element:
            shutil.move(os.path.join(path[2], 'PNEUMONIA', element), os.path.join(path[2], 'BACTERIA', element))
        else:
            shutil.move(os.path.join(path[2], 'PNEUMONIA', element), os.path.join(path[2], 'VIRUS', element))
    # Removes PNEUMONIA folder
    os.rmdir(str(path[2]) + 'PNEUMONIA')
   
# Print a graph to illustrate the distribution of data across train, validation, and test datasets
def visualize_class_distribution(dataset, dataset_name, view, resize):
    """
    Visualizes the class distribution in a dataset.

    :param dataset: The dataset to visualize.
    :type dataset: Dataset
    :param dataset_name: The name of the dataset.
    :type dataset_name: str
    :param view: Whether to display the plot.
    :type view: bool
    :param resize: Whether to resize the plot.
    :type resize: bool
    """
    # Initialize a vector to zero
    class_counts = np.zeros(len(dataset.classes))
    # Fills the vector with the number of elements for each class
    class_counts = class_count(dataset)
    # Print a bar graph according to the colors shown in the order
    plt.bar(dataset.classes, class_counts, color=['blue', 'orange', 'red', 'yellow', 'purple'])
    # Add labels and title
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.title(f"Class distribution in the {dataset_name.lower()} dataset")
    # Save graphs in the graph folder based on whether it's the original or resized dataset
    if not resize:
        dataset_name += '_original'
        save_graph(dataset_name, 'Dataset_original')
    else:
        dataset_name += '_resized'
        save_graph(dataset_name, 'Dataset_resized')
    # If the user expressed the preference in the base_config file, it shows the result
    if view:
        plt.show()
    # Closes the graph to avoid overlap
    plt.close()
    
# Invoke the visualize_class_distribution once for each dataset [Train, Validation, and Test]
def print_dataset_graph(train_dataset, val_dataset, test_dataset, view, resize):
    """
    Prints the class distribution graph for the train, validation, and test datasets.

    :param train_dataset: The training dataset.
    :type train_dataset: Dataset
    :param val_dataset: The validation dataset.
    :type val_dataset: Dataset
    :param test_dataset: The test dataset.
    :type test_dataset: Dataset
    :param view: Whether to display the plots.
    :type view: bool
    :param resize: Whether to resize the plots.
    :type resize: bool
    """
    # Print graphs
    print("Drawing graph for class distribution in dataset...")
    visualize_class_distribution(train_dataset, "Train", view, resize)
    visualize_class_distribution(val_dataset, "Validation", view, resize)
    visualize_class_distribution(test_dataset, "Test", view, resize)

# Creates dataset objects and redistributes data between Train and Validation sets
def load_datasets(config):
    """
    Loads the train, validation, and test datasets.

    :param config: The configuration for loading the datasets.
    :type config: Config
    :return: Returns the train, validation, and test datasets.
    :rtype: tuple (Dataset, Dataset, Dataset)
    """
    # Create objects of the ChestXrayDataset class for each dataset
    train_dataset = ChestXrayDataset(type='train', root=config.data)
    val_dataset  = ChestXrayDataset(type='val', root=config.data)
    test_dataset  = ChestXrayDataset(type='test', root=config.data)
    # Print statistics before
    print_shapes('before', train_dataset, val_dataset, test_dataset, config.classification.type)
    # If the user expressed the preference in the base_config file, it create the graph
    if config.graph.create_dataset_graph:
        print_dataset_graph(train_dataset, val_dataset, test_dataset, config.graph.view_dataset_graph, resize=False)
    print("---------------------")
    # Redistribute datasets
    train_dataset, val_dataset = resize_datasets(train_dataset, val_dataset, config.data.split_percentage)
    # Print statistics after
    print_shapes('after', train_dataset, val_dataset, test_dataset, config.classification.type)
    # If the user expressed the preference in the base_config file, it create the graph
    if config.graph.create_dataset_graph:
        print_dataset_graph(train_dataset, val_dataset, test_dataset, config.graph.view_dataset_graph, resize=True)
    # Return datasets
    return train_dataset, val_dataset, test_dataset

# Verify that the path passed exists
def verify_division(path):
    """
    Verifies if a directory exists.

    :param path: The path to the directory.
    :type path: str
    :return: Returns True if the directory exists, False otherwise.
    :rtype: bool
    """
    return os.path.isdir(path)

# It verifies that the folder division is adequate and proceeds with the data upload
def binary_load(config):
    """
    Loads the binary datasets.

    :param config: The configuration for loading the datasets.
    :type config: Config
    :return: Returns the binary train, validation, and test datasets.
    :rtype: tuple (Dataset, Dataset, Dataset)
    """
    # Verify that the folders of the binary split exist
    pneu_train = verify_division(config.data.data_dir + 'train//PNEUMONIA')
    pneu_val = verify_division(config.data.data_dir + 'val//PNEUMONIA')
    pneu_test = verify_division(config.data.data_dir + 'test//PNEUMONIA')
    # If the division is not correct, it proceeds with a suitable split
    if not (pneu_train and pneu_val and pneu_test):
        print("Transform dataset to binary...")
        split_binary_directory(config.data.data_dir)
    # Determine new datasets
    train_dataset, val_dataset, test_dataset = load_datasets(config)
    # Return datasets
    return train_dataset, val_dataset, test_dataset  

# It verifies that the folder division is adequate and proceeds with the data upload
def ternary_load(config):
    """
    Loads the ternary datasets.

    :param config: The configuration for loading the datasets.
    :type config: Config
    :return: Returns the ternary train, validation, and test datasets.
    :rtype: tuple (Dataset, Dataset, Dataset)
    """
    # Verify that the folders of the ternary split exist
    bact_train = verify_division(config.data.data_dir + 'train//BACTERIA')
    vir_train = verify_division(config.data.data_dir + 'train//VIRUS')
    bact_val = verify_division(config.data.data_dir + 'val//BACTERIA')
    vir_val = verify_division(config.data.data_dir + 'val//VIRUS')
    bact_test = verify_division(config.data.data_dir + 'test//BACTERIA')
    vir_test = verify_division(config.data.data_dir + 'test//VIRUS')
    # If the division is not correct, it proceeds with a suitable split
    if not (bact_train and vir_train and bact_val and vir_val and bact_test and vir_test):
        print("Transform dataset to ternary...")
        split_ternary_directory(config.data.data_dir)
    # Determine new datasets
    train_dataset, val_dataset, test_dataset = load_datasets(config)
    # Return datasets
    return train_dataset, val_dataset, test_dataset
