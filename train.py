#-----  Command to run from terminal  -----#
# python -u train.py -c config/base_config.yaml

# Documentation command [execute in path "..\docs"]: .\make.bat html

# Standard library imports
import os
import warnings

# Third-party imports
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
from joblib import dump

# Configuration and utility imports
from yaml_config_override import add_arguments
from addict import Dict

# Local application/library specific imports
from data_classes.manage_dataset import *
from model_classes.resnet_model import ResNet, ResidualBlock
from model_classes.alexnet_model import AlexNet
from model_classes.svm_model import SVM
from utils import *
from extract_representations.vision_embeddings import VisionEmbeddings


# Train a training epoch
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Trains a model for one epoch.

    :param model: The model to be trained.
    :type model: torch.nn.Module
    :param dataloader: The DataLoader for the training dataset.
    :type dataloader: torch.utils.data.DataLoader
    :param criterion: The criterion to use for calculating loss during training.
    :type criterion: torch.nn.modules.loss._Loss
    :param optimizer: The optimizer to use for updating the model parameters.
    :type optimizer: torch.optim.Optimizer
    :param scheduler: The learning rate scheduler.
    :type scheduler: torch.optim.lr_scheduler._LRScheduler
    :param device: The device on which to train the model (e.g., 'cpu', 'cuda').
    :type device: str
    :return: Returns a dictionary containing the performance metrics of the training dataset.
    :rtype: dict
    """
    # Set the model to training mode
    model.train() 
    # Inizialize variables
    running_loss = 0.0
    predictions = []
    references = []
    # Batch execution
    for i, batch in enumerate(tqdm(dataloader, desc='Training')):
        # Upload images and labels
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        # Erasing old gradient values
        optimizer.zero_grad()
        # Calculate output
        outputs = model(images)
        # Calculate the loss through the previously chosen loss function
        loss = criterion(outputs, labels)
        # Calculate the gradient
        loss.backward()
        # Optimization
        optimizer.step()
        # Scheduler learning rate
        scheduler.step()
        # Add the current loss to the total
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

# Manage best model and best validation metrics
def manage_best_model_and_metrics(model, evaluation_metric, val_metrics, best_val_metric, best_model, lower_is_better):
    """
    Manages the best model and its metrics.

    :param model: The current model.
    :type model: torch.nn.Module
    :param evaluation_metric: The metric against which to determine the best model.
    :type evaluation_metric: str
    :param val_metrics: The validation metrics of the current model.
    :type val_metrics: dict
    :param best_val_metric: The best validation metrics so far.
    :type best_val_metric: dict
    :param best_model: The best model so far.
    :type best_model: torch.nn.Module
    :param lower_is_better: Whether a lower metric value is better.
    :type lower_is_better: bool
    :return: Returns the best validation metrics and the best model.
    :rtype: tuple (dict, torch.nn.Module)
    """
    # Based on the evaluation metric you choose, evaluate whether the current one is higher or not
    if lower_is_better:
        is_best = val_metrics[evaluation_metric] < best_val_metric[evaluation_metric]
    else:
        is_best = val_metrics[evaluation_metric] > best_val_metric[evaluation_metric]
    if is_best:
        # Verify the model for which a new best model has been found
        if str(type(model)) == '<class \'model_classes.resnet_model.ResNet\'>':
             print(f"New best ResNet model found with val {evaluation_metric}: {val_metrics[evaluation_metric]:.4f}")
        elif str(type(model)) == '<class \'model_classes.alexnet_model.AlexNet\'>':
             print(f"New best AlexNet model found with val {evaluation_metric}: {val_metrics[evaluation_metric]:.4f}")
        else:
             print(f"New best SVM model found with val {evaluation_metric}: {val_metrics[evaluation_metric]:.4f}")
        # Store the new best validation metrics and the new best model
        best_val_metric = val_metrics
        best_model = model
    return best_val_metric, best_model

# Train deep learning models
def train_model(model, config, train_dl, device, criterion):
    """
    This function trains a model given a training dataset, a device, and a criterion.

    :param model: The model to be trained.
    :type model: torch.nn.Module
    :param config: The configuration for training the model.
    :type config: Config
    :param train_dl: The DataLoader for the training dataset.
    :type train_dl: torch.utils.data.DataLoader
    :param device: The device on which to train the model (e.g., 'cpu', 'cuda').
    :type device: str
    :param criterion: The criterion to use for calculating loss during training.
    :type criterion: torch.nn.modules.loss._Loss
    :return: Returns the best validation metric, the best model, the list of training metrics, and the list of validation metrics.
    :rtype: tuple (dict, torch.nn.Module, list, list)
    """
    # Initializing variables
    training_metrics_list = []
    validation_metrics_list = []
    # Identify the model
    if str(type(model)) == '<class \'model_classes.resnet_model.ResNet\'>':
        string_model = "ResNet model -"
    else:
        string_model = "AlexNet model -"
    # Select the optimization algorithm based on the configuration chosen in the config
    if config.training.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    elif config.training.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.training.learning_rate)
    # learning rate scheduler
    total_steps = len(train_dl) * config.training.epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    # Warmup for [warmup_ratio]% and linear decay
    scheduler_lambda = lambda step: (step / warmup_steps) if step < warmup_steps else max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda)
    # Initialization of the dictionary that contains performance metrics
    best_val_metric = {'accuracy': float, 'precision': float, 'recall': float, 'f1': float}
    # Initializing the best performance metrics and the other variables
    if config.training.best_metric_lower_is_better:
        best_val_metric[config.training.evaluation_metric.lower()] = float('inf')
    else:
        best_val_metric[config.training.evaluation_metric.lower()] = float('-inf')
    best_model = None
    no_valid_epochs = 0
    # Training cycle that iterates through all epochs
    for epoch in range(config.training.epochs):
        print("%s Epoch %d/%d" % (string_model, (epoch + 1), config.training.epochs))
        # Train an epoch and calculate performance on training
        train_metrics = train_one_epoch(model, train_dl, criterion, optimizer, scheduler, device)
        # Add the performance to the list that will contain the history of all the training performance
        training_metrics_list.append(train_metrics)
        # Calculate performance on validation
        val_metrics, conf_matrix = evaluate(model, val_dl, criterion, device)
        # Add the performance to the list that will contain the history of all the validation performance
        validation_metrics_list.append(val_metrics)
        # Print results
        print(f"Train loss: {train_metrics['loss']:.4f} - Train accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Val loss: {val_metrics['loss']:.4f} - Val accuracy: {val_metrics['accuracy']:.4f}")
        # Calculates the new best model and new best performance as a result of the last training
        best_val_metric, best_model = manage_best_model_and_metrics(
            model, 
            config.training.evaluation_metric, 
            val_metrics, 
            best_val_metric, 
            best_model, 
            config.training.best_metric_lower_is_better
        )
        # Earling stopping
        # Check which metric was chosen for early stopping
        if config.training.early_stopping_metric.lower() == 'loss':
            # If performance is not improved, the early stopping counter increases by one
            if val_metrics[config.training.early_stopping_metric.lower()] > best_val_metric[config.training.early_stopping_metric.lower()]:
                no_valid_epochs += 1
        else:
            # If performance is not improved, the early stopping counter increases by one
            if val_metrics[config.training.early_stopping_metric.lower()] < best_val_metric[config.training.early_stopping_metric.lower()]:
                no_valid_epochs += 1
        # If you have reached the maximum value chosen through the config file, end the training
        if no_valid_epochs == config.training.earling_stopping_max_no_valid_epochs:
            print(f"The training process has ended because the maximum value of early stopping, which is {config.training.earling_stopping_max_no_valid_epochs:}, has been reached.")
            break
    # Returns the best metrics, best model, and performance history for training and validation
    return best_val_metric, best_model, training_metrics_list, validation_metrics_list

# Evaluate the performance of the model on the incoming passed dataset
def evaluate_model(best_val_metric, best_model, test_set, criterion, device, type_model):
    """
    This function evaluates the best model on a test dataset.

    :param best_val_metric: The best validation metric obtained during training.
    :type best_val_metric: dict
    :param best_model: The best model obtained during training.
    :type best_model: torch.nn.Module
    :param test_set: Dataloader or SVM dataset based on the model
    :type test_set: torch.utils.data.DataLoader or test_svm
    :param criterion: The criterion to use for calculating loss during evaluation.
    :type criterion: torch.nn.modules.loss._Loss or Funcion
    :param device: The device on which to evaluate the model (e.g., 'cpu', 'cuda').
    :type device: str
    :param type_model: The type of the model (e.g., 'ResNet', 'AlexNet').
    :type type_model: str
    :return: Returns the metrics and confusion matrix for the test dataset.
    :rtype: tuple (list, numpy.ndarray)
    """
    print("---------------------")
    print_best_val_metrics(type_model, best_val_metric)
    # Evaluate the performance of the test_dataset
    # SVM model
    if type_model == 'SVM':
        test_metrics, conf_matrix = evaluate_svm(best_model, test_set, criterion)
    # Deep learning models
    else:
        test_metrics, conf_matrix = evaluate(best_model, test_set, criterion, device)
    # Compute performance metrics
    metrics = print_evaluation(test_metrics)
    # Return performance metrics and confusion matrix
    return metrics, conf_matrix

# Train svm model
def train_smv_model(model, train, validation, config, criterion):
    """
    Trains the SVM model, evaluates it applying early stopping and scheduler learning rate.

    :param model: The SVM model to be trained.
    :type model: SVM class
    :param train: The training dataset.
    :type train: SVM_dataset
    :param validation: The validation dataset.
    :type validation: SVM_dataset
    :param config: The configuration settings.
    :type config: Dict(add_arguments())
    :param criterion: The loss function to use.
    :type criterion: Function
    :return: The best validation metrics, the best model, and the lists of training and validation metrics.
    :rtype: Tuple (best_val_metric, best_model, training_metrics_list, validation_metrics_list)
    """
    # Initializing the best performance metrics and the other variables
    training_metrics_list = []
    validation_metrics_list = []
    best_val_metric = {'accuracy': float, 'precision': float, 'recall': float, 'f1': float, 'loss': float}
    best_model = None
    no_valid_epochs = 0
    if config.training.best_metric_lower_is_better:
        best_val_metric[config.training.evaluation_metric.lower()] = float('inf')
    else:
        best_val_metric[config.training.evaluation_metric.lower()] = float('-inf')
    # Print some statistics
    print("Samples: " + str(train.num_of_samples) + " - Features: " + str(train.num_of_features) + "\n")
    # For epochs in range (0, number of total epochs * 2)
    for epoch in range(config.training.epochs * 2):
        print("SVM model - Epoch %d/%d" % ((epoch + 1), (config.training.epochs * 2)))
        # Scheduler learning rate
        learning_rate = 1 / (500 + ((epoch ** 2) + (epoch * 5)))
        # Train an epoch and calculate performance on training
        train_metrics = model.fit(train, criterion, learning_rate)
        # Add the performance to the list that will contain the history of all the training performance
        training_metrics_list.append(train_metrics)
        # Calculate performance on validation
        val_metrics, conf_matrix = model.evaluate_svm(validation, criterion)
        # Add the performance to the list that will contain the history of all the validation performance
        validation_metrics_list.append(val_metrics)
        # Print results
        print(f"Train loss: {train_metrics['loss']:.4f} - Train accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Val loss: {val_metrics['loss']:.4f} - Val accuracy: {val_metrics['accuracy']:.4f}")
        # Calculates the new best model and new best performance as a result of the last training
        best_val_metric, best_model = manage_best_model_and_metrics(
            model, 
            config.training.evaluation_metric, 
            val_metrics, 
            best_val_metric, 
            best_model, 
            config.training.best_metric_lower_is_better
        )
        # Early stopping
        if config.training.early_stopping_metric.lower() == 'loss':
            # If performance is not improved, the early stopping counter increases by one
            if val_metrics[config.training.early_stopping_metric.lower()] > best_val_metric[config.training.early_stopping_metric.lower()]:
                no_valid_epochs += 1
        else:
            # If performance is not improved, the early stopping counter increases by one
            if val_metrics[config.training.early_stopping_metric.lower()] < best_val_metric[config.training.early_stopping_metric.lower()]:
                no_valid_epochs += 1
        # If you have reached the maximum value chosen through the config file, end the training
        if no_valid_epochs == config.training.earling_stopping_max_no_valid_epochs:
            print(f"The training process has ended because the maximum value of early stopping, which is {config.training.earling_stopping_max_no_valid_epochs:}, has been reached.")
            break
    return best_val_metric, best_model, training_metrics_list, validation_metrics_list

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
    
    # Configuration parameters
    config = Dict(add_arguments())
    
    # ---------------------
    # 2. Device configuration
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
    
    # Create the dataset items based on the type of classification choosen
    if config.classification.type.lower() == 'binary':
        train_dataset, val_dataset, test_dataset = binary_load(config)
    else:    
        train_dataset, val_dataset, test_dataset = ternary_load(config)
        
    # Use vision embedding to create features from tensor for SVM model
    vision_embeddings = VisionEmbeddings(model_name='google/vit-base-patch16-224', device=device)
    print("Vision embeddings for SVM:\n")
    # Extract new dataset using embeddings
    train_svm, validation_svm, test_svm = vision_embeddings.extract_all_datasets(
        train_dataset,
        val_dataset,
        test_dataset,
        config.data.principal_component,
        config.training.checkpoint_dir,
        config.graph.create_model_graph,
        config.graph.view_model_graph
    )
    
    # --- Oversampling for deep learning models ---
    # Compute class count for each class
    targets = torch.tensor(train_dataset.targets)
    class_counts = torch.bincount(targets)
    # Compute weights for each class [weight is inversely proportional to class_count]
    class_weights = 1. / class_counts.float()
    # Give a sample the weight of its class
    sample_weights = class_weights[train_dataset.targets] * config.data.strength_of_oversampling
    # Create a bilanced sampler
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Loading the train_dataset
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=False, # Shuffling is mutually exclusive with sampler
        sampler=sampler
    )
    
    # Loading the train_dataset
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False # Without shuffling the data
    )
    
    # Loading the train_dataset
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False # Without shuffling the data
    )
    print("---------------------")
    
    # ---------------------
    # 4. Load model
    # ---------------------
    
    # Load the templates and specify their configuration through the config variable
    
    # ResNet model
    first_model = ResNet(
        ResidualBlock,
        config.ResNet_model.layers,
        config.classification.type,
        config.ResNet_model.stride,
        config.ResNet_model.padding,
        config.ResNet_model.kernel,
        config.ResNet_model.channels_of_color,
        config.ResNet_model.planes,
        config.ResNet_model.in_features,
        config.ResNet_model.inplanes
    )
    first_model.to(device)
    
    # AlexNet model
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
    svm_model = SVM(config.training.epochs, config.training.learning_rate, train_svm.num_of_features)
    
    # ---------------------
    # 5. Train model
    # ---------------------
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    criterion_svm = svm_model.hinge_loss
    
    print("Training deep learning models:\n")
    # Train the ResNet model
    first_best_val_metric, first_best_model, first_training_metrics, first_validation_metrics = train_model(first_model, config, train_dl, device, criterion)
    print("---------------------")
    # Train the AlexNet model
    second_best_val_metric, second_best_model, second_training_metrics, second_validation_metrics = train_model(second_model, config, train_dl, device, criterion)
    print("---------------------")
    print("Training SVM model:")
    # Train SVM model
    best_val_metric_svm, best_model_svm, train_metrics_svm, validation_metrics_svm = train_smv_model(svm_model, train_svm, validation_svm, config, criterion_svm)
    # Depending on the configuration you choose, create graphs for performance
    if config.graph.create_model_graph:
        print_metrics_graph(first_training_metrics, first_validation_metrics, config.graph.metric_plotted_during_traininig, config.graph.view_model_graph, type_model='ResNet')
        print_metrics_graph(second_training_metrics, second_validation_metrics, config.graph.metric_plotted_during_traininig, config.graph.view_model_graph, type_model='AlexNet')
        print_metrics_graph(train_metrics_svm, validation_metrics_svm, config.graph.metric_plotted_during_traininig, config.graph.view_model_graph, type_model='SVM')
        
    # --------------------------------
    # 6. Evaluate model on test set
    # --------------------------------
    
    # Evaluate the performance of the ResNet model on the test_dataset
    first_metrics, first_conf_matrix = evaluate_model(first_best_val_metric, first_best_model, test_dl, criterion, device, type_model='ResNet')
    print()
    # Prints the confusion matrix of the ResNet model
    print_confusion_matrix(first_conf_matrix, type_model='ResNet')
    print()
    
    # Evaluate the performance of the AlexNet model on the test_dataset
    second_metrics, second_conf_matrix = evaluate_model(second_best_val_metric, second_best_model, test_dl, criterion, device, type_model='AlexNet')
    print()
    # Prints the confusion matrix of the AlexNet model
    print_confusion_matrix(second_conf_matrix, type_model='AlexNet')
    print("---------------------")
    
    # Evaluate the performance of the SVM model on the test_dataset
    svm_metrics, svm_conf_matrix = evaluate_model(best_val_metric_svm, best_model_svm, test_svm, criterion_svm, device, type_model='SVM')
    print()
    # Prints the confusion matrix of SVM model
    print_confusion_matrix(svm_conf_matrix, type_model='SVM')
    print("---------------------")
    # Depending on the configuration you choose, create graphs for confusion matrix
    if config.graph.create_model_graph:
        print_confusion_matrix_graph(first_conf_matrix, config.graph.view_model_graph, type_model='ResNet', test=False)
        print_confusion_matrix_graph(second_conf_matrix, config.graph.view_model_graph, type_model='AlexNet', test=False)
        print_confusion_matrix_graph(svm_conf_matrix, config.graph.view_model_graph, type_model='SVM', test=False)

    # ---------------------
    # 7. Compare performance
    # ---------------------
    
    # Compare the performance of the two deep learning models and SVM model
    compare_performance(first_metrics, second_metrics, svm_metrics)
    # Depending on the configuration you choose, create graphs for compare graph
    if config.graph.create_compare_graph:
        print_compare_graph(first_metrics, second_metrics, svm_metrics, config.graph.view_compare_graph, test=False)

    # ---------------------
    # 8. Save model
    # ---------------------
    
    # Verify that the checkpoint folder passed by the configuration parameter exists
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    # Store model weights in the checkpoin folder
    torch.save(first_best_model.state_dict(), f"{config.training.checkpoint_dir}/ResNet_best_model.pt")
    torch.save(second_best_model.state_dict(), f"{config.training.checkpoint_dir}/AlexNet_best_model.pt")
    dump(svm_model, f"{config.training.checkpoint_dir}/SVM_best_model.pkl")

    print("---------------------")
    print("\nModels saved.")
    
    print("\nTrain finish correctly.\n")
    