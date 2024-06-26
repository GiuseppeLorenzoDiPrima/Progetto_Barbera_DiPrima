# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local application/library specific imports
from data_classes.manage_dataset import ChestXrayDataset
from model_classes.resnet_model import ResNet, ResidualBlock
from model_classes.alexnet_model import AlexNet
from utils import *

# Configuration and utility imports
from yaml_config_override import add_arguments
from addict import Dict


def print_metrics(metrics):
    for key, value in metrics.items():
        print(f"Test {key}: {value:.4f}")

def extract_value(metrics):
    values = []
    for key, value in metrics.items():
        values.append(value)
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

    config = Dict(add_arguments())
    
    # ---------------------
    # 2. Set device
    # ---------------------
    
    if config.training.device.lower() == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("\nDevice: " + torch.cuda.get_device_name()) 
    print("---------------------")

    # ---------------------
    # 3. Load data
    # ---------------------
    
    test_dataset  = ChestXrayDataset(type='test', root=config.data)
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )
    
    # ---------------------
    # 4. Load model
    # ---------------------
    
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
    
    second_model = AlexNet(
        config.classification.type,
        config.AlexNet_model.stride,
        config.AlexNet_model.padding,
        config.AlexNet_model.kernel,
        config.AlexNet_model.channels_of_color,
        config.AlexNet_model.inplace,
    )
    second_model.to(device)

    # ---------------------
    # 5. Load model weights
    # ---------------------
    
    print("Loading models...")
    
    first_model.load_state_dict(torch.load(f"{config.training.checkpoint_dir}/ResNet_best_model.pt"))
    print("-> ResNet model loaded.")
    
    second_model.load_state_dict(torch.load(f"{config.training.checkpoint_dir}/AlexNet_best_model.pt"))
    print("-> AlexNet model loaded.")
    print("---------------------")

    # ---------------------
    # 6. Criterion
    # ---------------------
    
    criterion = nn.CrossEntropyLoss()
    
    # ---------------------
    # 7. Evaluate
    # ---------------------
    
    first_metrics, first_conf_matrix = evaluate(first_model, test_dl, criterion, device)
    second_metrics, second_conf_matrix = evaluate(second_model, test_dl, criterion, device)
    
    # ---------------------
    # 8. Print performance
    # ---------------------
    
    print("Performance:\n")
    print("ResNet model performance:")
    print_metrics(first_metrics)
    print()
    print("AlexNet model performance:")
    print_metrics(second_metrics)
    print("---------------------")
         
    # ---------------------
    # 9. Compare performance
    # ---------------------
        
    values = [] 
    values.append(extract_value(first_metrics))
    values.append(extract_value(second_metrics))
    compare_performance(values[0], values[1])
    print("---------------------")
    
    print("\nTest finish correctly.\n")
