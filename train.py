#-----  Command to run from terminal  -----#
# python -u train.py -c config/base_config.yaml

# Standard library imports
import os

# Third-party imports
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

# Configuration and utility imports
from yaml_config_override import add_arguments
from addict import Dict

# Local application/library specific imports
from data_classes.manage_dataset import *
from model_classes.resnet_model import ResNet, ResidualBlock
from model_classes.alexnet_model import AlexNet
from utils import *


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    # Inizialize variables
    model.train()
    running_loss = 0.0
    predictions = []
    references = []
    # Batch execution
    for i, batch in enumerate(tqdm(dataloader, desc='Training')):
        # Upload images and labels
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        # Configure optimizer and calculate loss
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        # Compute predictions
        pred = torch.argmax(outputs, dim=1)
        predictions.extend(pred.cpu().numpy())
        # Compute refereces
        references.extend(labels.cpu().numpy())
    # Compute performance metrics based on differences between predictiones and references 
    train_metrics = compute_metrics(predictions, references)
    # Add loss to performance metrics
    train_metrics['loss'] = running_loss / len(dataloader)
    # Return performance metrics
    return train_metrics

def manage_best_model_and_metrics(model, evaluation_metric, val_metrics, best_val_metric, best_model, lower_is_better):
    if lower_is_better:
        is_best = val_metrics[evaluation_metric] < best_val_metric[evaluation_metric]
    else:
        is_best = val_metrics[evaluation_metric] > best_val_metric[evaluation_metric]
    if is_best:
        if str(type(model)) == '<class \'model_classes.resnet_model.ResNet\'>':
             print(f"New best ResNet model found with val {evaluation_metric}: {val_metrics[evaluation_metric]:.4f}")
        else:
             print(f"New best AlexNet model found with val {evaluation_metric}: {val_metrics[evaluation_metric]:.4f}")
        best_val_metric = val_metrics
        best_model = model
    return best_val_metric, best_model

def train_model(model, config, train_dl, device, criterion):
    training_metrics_list = []
    validation_metrics_list = []
    if str(type(model)) == '<class \'model_classes.resnet_model.ResNet\'>':
        string_model = "ResNet model -"
    else:
        string_model = "AlexNet model -"
    if config.training.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    elif config.training.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.training.learning_rate)
    # learning rate scheduler
    total_steps = len(train_dl) * config.training.epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    # warmup + linear decay
    scheduler_lambda = lambda step: (step / warmup_steps) if step < warmup_steps else max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda)
    best_val_metric = {'accuracy': float, 'precision': float, 'recall': float, 'f1': float}
    if config.training.best_metric_lower_is_better:
        best_val_metric[config.training.evaluation_metric.lower()] = float('inf')
    else:
        best_val_metric[config.training.evaluation_metric.lower()] = float('-inf')
    best_model = None
    no_valid_epochs = 0
    for epoch in range(config.training.epochs):
        print("%s Epoch %d/%d" % (string_model, (epoch + 1), config.training.epochs))
        train_metrics = train_one_epoch(model, train_dl, criterion, optimizer, scheduler, device)
        training_metrics_list.append(train_metrics)
        val_metrics, conf_matrix = evaluate(model, val_dl, criterion, device)
        validation_metrics_list.append(val_metrics)
        print(f"Train loss: {train_metrics['loss']:.4f} - Train accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Val loss: {val_metrics['loss']:.4f} - Val accuracy: {val_metrics['accuracy']:.4f}")
        best_val_metric, best_model = manage_best_model_and_metrics(
            model, 
            config.training.evaluation_metric, 
            val_metrics, 
            best_val_metric, 
            best_model, 
            config.training.best_metric_lower_is_better
        )
        # Earling stopping
        if config.training.early_stopping_metric.lower() == 'loss':
            if val_metrics[config.training.early_stopping_metric.lower()] > best_val_metric[config.training.early_stopping_metric.lower()]:
                no_valid_epochs += 1
        else:
            if val_metrics[config.training.early_stopping_metric.lower()] < best_val_metric[config.training.early_stopping_metric.lower()]:
                no_valid_epochs += 1
        if no_valid_epochs == config.training.earling_stopping_max_no_valid_epochs:
            print(f"The training process has ended because the maximum value of early stopping, which is {config.training.earling_stopping_max_no_valid_epochs:}, has been reached.")
            break
    return best_val_metric, best_model, training_metrics_list, validation_metrics_list

def evaluate_model(best_val_metric, best_model, test_dl, criterion, device, type_model):
    print("---------------------")
    print("Best " + type_model + " model performance on validation dataset:")
    for key, value in best_val_metric.items():
        print("\t- Best " + type_model + f" model {key}: {value:.4f}")
    print("\nTesting " + type_model + " model on test dataset...")
    test_metrics, conf_matrix = evaluate(best_model, test_dl, criterion, device)
    metrics = [test_metrics['accuracy'], test_metrics['precision'], test_metrics['recall'], test_metrics['f1'], test_metrics['loss']]
    labels = ['Test accuracy', 'Test precision', 'Test recall', 'Test f1 score', 'Test loss']
    test_result = pd.DataFrame(metrics, columns=['Value'], index=labels).round(4)
    print(test_result)
    return metrics, conf_matrix


if __name__ == '__main__':
    """
    The main script for training and evaluating the models.

    The script performs the following steps:
    1. Parse the configuration.
    2. Set the device for training.
    3. Load the data.
    4. Load the models.
    5. Train the models.
    6. Evaluate the models on the test set.
    7. Compare the performance of the models.
    8. Save the models.
    """
    
    # ---------------------
    # 1. Parse configuration
    # ---------------------
    
    config = Dict(add_arguments())
    
    # ---------------------
    # 2. Device configuration
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
    
    # Calcolo i dataset
    train_dataset, val_dataset, test_dataset = binary_load(config)
        
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False
    )
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
    # 5. Train model
    # ---------------------
    
    criterion = nn.CrossEntropyLoss()
    first_best_val_metric, first_best_model, first_training_metrics, first_validation_metrics = train_model(first_model, config, train_dl, device, criterion)
    print("---------------------")
    second_best_val_metric, second_best_model, second_training_metrics, second_validation_metrics = train_model(second_model, config, train_dl, device, criterion)
    
    # --------------------------------
    # 6. Evaluate model on test set
    # --------------------------------
    
    first_metrics, first_conf_matrix = evaluate_model(first_best_val_metric, first_best_model, test_dl, criterion, device, type_model='ResNet')
    print()
    print_confusion_matrix(first_conf_matrix, type_model='ResNet')
    print()
    second_metrics, second_conf_matrix = evaluate_model(second_best_val_metric, second_best_model, test_dl, criterion, device, type_model='AlexNet')
    print()
    print_confusion_matrix(second_conf_matrix, type_model='AlexNet')
    print("---------------------")
    
    # ---------------------
    # 7. Compare performance
    # ---------------------
    
    compare_performance(first_metrics, second_metrics)
    
    # ---------------------
    # 8. Save model
    # ---------------------
    
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    torch.save(first_best_model.state_dict(), f"{config.training.checkpoint_dir}/ResNet_best_model.pt")
    torch.save(second_best_model.state_dict(), f"{config.training.checkpoint_dir}/AlexNet_best_model.pt")
    print("---------------------")
    print("\nModels saved.")
    
    print("\nTrain finish correctly.\n")
    