# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from joblib import load

# Local application/library specific imports
from data_classes.manage_dataset import ChestXrayDataset
from model_classes.resnet_model import ResNet, ResidualBlock
from model_classes.alexnet_model import AlexNet
from model_classes.svm_model import SVM
from utils import *
from extract_representations.vision_embeddings import VisionEmbeddings

# Configuration and utility imports
from yaml_config_override import add_arguments
from addict import Dict

# Print test set performance metrics
def print_metrics(metrics):
    """
    Prints the metrics.

    :param metrics: The metrics to print.
    :type metrics: Dictionary
    """
    # Scrolls through the dictionary and prints performance metrics
    for key, value in metrics.items():
        print(f"Test {key}: {value:.4f}")

# Extracts metric values from the dictionary
def extract_value(metrics):
    """
    Extracts the values from the metrics.

    :param metrics: The metrics to extract values from.
    :type metrics: dict
    :return: Returns a list of the extracted values.
    :rtype: list
    """
    # Initialize an array
    values = []
    # Iterates through dictionary values and adds them to the list
    for key, value in metrics.items():
        values.append(value)
    # Return the list
    return values


if __name__ == '__main__':
    """
    The main script for training and evaluating the models.

    The script performs the following steps:
    1. Load the configuration.
    2. Set the device for training.
    3. Load the data.
    4. Load the models.
    5. Load the model weights.
    6. Set the criterion for training.
    7. Evaluate the models.
    8. Print the performance of the models.
    9. Compare the performance of the models.
    """
    
    # ---------------------
    # 1. Load configuration
    # ---------------------
    
    # Configuration parameters
    config = Dict(add_arguments())
    
    # ---------------------
    # 2. Set device
    # ---------------------
    
    # Selecting the device to run with: CUDA -> GPU; CPU -> CPU
    if config.training.device.lower() == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("\nDevice: " + torch.cuda.get_device_name()) 
    print("---------------------")

    # ---------------------
    # 3. Load data
    # ---------------------
    
    # Create the test_dataset item
    test_dataset  = ChestXrayDataset(type='test', root=config.data)
    # Loading the test_dataset
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False # Without shuffling the data
    )
    print("Vision embeddings for SVM:\n")
    # Load the pca object determined during the training phase
    pca = load(f"{config.training.checkpoint_dir}/pca.joblib")
    # Create vision_embedding object
    vision_embeddings = VisionEmbeddings()
    # Create the dataset containing features for the svm model
    test_dataset_svm = vision_embeddings.extract_single_dataset(test_dataset, pca, 'test', config.graph.create_model_graph, config.graph.view_model_graph)
    print("---------------------")
    
    # ---------------------
    # 4. Load model
    # ---------------------
    
    # Load the templates and specify their configuration through the config variable
    # ResNet Model
    first_model = ResNet(
        ResidualBlock,
        [3,4,6,3],
        config.classification.type,
        config.ResNet_model.stride,
        config.ResNet_model.padding,
        config.ResNet_model.kernel,
        config.ResNet_model.channels_of_color,
        config.ResNet_model.planes,
        config.ResNet_model.in_features
    )
    first_model.to(device)
    # AlexNet Model
    second_model = AlexNet(
        config.classification.type,
        config.AlexNet_model.stride,
        config.AlexNet_model.padding,
        config.AlexNet_model.kernel,
        config.AlexNet_model.channels_of_color,
        config.AlexNet_model.inplace,
    )
    second_model.to(device)
    # SVM model
    svm_model = SVM(config.training.epochs, config.training.learning_rate, test_dataset_svm.num_of_features)

    # ---------------------
    # 5. Load model weights
    # ---------------------
    
    # Loads the saved model weights to the specified folder during training
    print("Loading models...")
    # First model
    first_model.load_state_dict(torch.load(f"{config.training.checkpoint_dir}/ResNet_best_model.pt"))
    print("-> ResNet model loaded.")
    # Second model
    second_model.load_state_dict(torch.load(f"{config.training.checkpoint_dir}/AlexNet_best_model.pt"))
    print("-> AlexNet model loaded.")
    # SVM model
    svm_model = load(f"{config.training.checkpoint_dir}/SVM_best_model.pkl")
    print("-> SVM model loaded.")
    print("---------------------")

    # ---------------------
    # 6. Criterion
    # ---------------------
    
    # Defines the CrossEntropyLoss and hingeLoss as loss functions for deep and machine learning models, respectively
    criterion = nn.CrossEntropyLoss()
    criterion_svm = svm_model.hinge_loss
    
    # ---------------------
    # 7. Evaluate
    # ---------------------
    
    print("Evaluating models...\n")
    # Evaluate ResNet model performance
    first_metrics, first_conf_matrix = evaluate(first_model, test_dl, criterion, device)
    # Prints the confusion matrix of the ResNet model
    print_confusion_matrix(first_conf_matrix, type_model='ResNet')
    print()
    # Evaluate AlexNet model performance
    second_metrics, second_conf_matrix = evaluate(second_model, test_dl, criterion, device)
    # Prints the confusion matrix of the AlexNet model
    print_confusion_matrix(second_conf_matrix, type_model='AlexNet')
    print()
    # Evaluate SVM model performance
    svm_metrics, svm_conf_matrix = evaluate_svm(svm_model, test_dataset_svm, criterion_svm)
    # Prints the confusion matrix of SVM model
    print_confusion_matrix(svm_conf_matrix, type_model='SVM')
    print("---------------------")
    # Print confusion matrices graphs
    if config.graph.create_model_graph:
        print_confusion_matrix_graph(first_conf_matrix, config.graph.view_model_graph, type_model='ResNet', test=True)
        print_confusion_matrix_graph(second_conf_matrix, config.graph.view_model_graph, type_model='AlexNet', test=True)
        print_confusion_matrix_graph(svm_conf_matrix, config.graph.view_model_graph, type_model='SVM', test=True)

    # ---------------------
    # 8. Print performance
    # ---------------------
    
    print("Performance:\n")
    print("ResNet model performance:")
    # Print the performance of ResNet model
    print_metrics(first_metrics)
    print()
    print("AlexNet model performance:")
    # Print the performance of AlexNet model
    print_metrics(second_metrics)
    print()
    print("SVM model performance:")
    # Print the performance of SVM model
    print_metrics(svm_metrics)
    print("---------------------")
         
    # ---------------------
    # 9. Compare performance
    # ---------------------
        
    # Initialize an array
    values = []
    # Inserts dictionaries containing performance into the array
    values.append(extract_value(first_metrics))
    values.append(extract_value(second_metrics))
    values.append(extract_value(svm_metrics))
    # Compare performance
    compare_performance(values[0], values[1], values[2])
    print("---------------------")
    # Print performance comparison results
    if config.graph.create_compare_graph:
        print_compare_graph(values[0], values[1], values[2], config.graph.view_compare_graph, test=True)

    print("\nTest finish correctly.\n")
