# Third-party imports
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Standard library imports
import os
import tkinter

# Local application/library specific imports
from data_classes.manage_dataset import *


# Calculate performance metrics
def compute_metrics(predictions, references):
    """
    Computes accuracy, precision, recall, and F1 score.

    :param predictions: The predicted labels.
    :type predictions: List
    :param references: The true labels.
    :type references: List
    :return: A dictionary containing the accuracy, precision, recall, and F1 score.
    :rtype: Dictionary
    """
    # Compute performance metrics: accuracy, precision, recall and f1
    acc = accuracy_score(references, predictions)
    precision = precision_score(references, predictions, average='macro', zero_division=0.0)
    recall = recall_score(references, predictions, average='macro', zero_division=0.0)
    f1 = f1_score(references, predictions, average='macro', zero_division=0.0)
    # Return metrics to a dictionary
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Evaluate performance metrics and confusion matrix
def evaluate(model, dataloader, criterion, device):
    """
    Evaluates a model on a given dataset.

    :param model: The model to be evaluated.
    :type model: torch.nn.Module
    :param dataloader: The DataLoader for the dataset.
    :type dataloader: torch.utils.data.DataLoader
    :param criterion: The criterion to use for calculating loss during evaluation.
    :type criterion: torch.nn.modules.loss._Loss
    :param device: The device on which to evaluate the model (e.g., 'cpu', 'cuda').
    :type device: String
    :return: Returns the evaluation metrics and confusion matrix.
    :rtype: tuple (dict, numpy.ndarray)
    """
    # Set the model to evaluation mode
    model.eval()
    # Initialize variables
    running_loss = 0.0
    predictions = []
    references = []
    # Specify that you don't want to calculate the gradient to save computational power
    with torch.no_grad():
        # Iterates through all batches in the dataloader
        for batch in dataloader:
            # Get images and targets
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            # Calculate output
            outputs = model(images)
            # Calculate the loss through the previously chosen loss function
            loss = criterion(outputs, labels)
            # Add the current loss to the total
            running_loss += loss.item()
            # Compute predictions
            pred = torch.argmax(outputs, dim=1)
            predictions.extend(pred.cpu().numpy())
            # Compute refereces
            references.extend(labels.cpu().numpy())
    # Compute performance metrics based on differences between predictiones and references
    val_metrics = compute_metrics(predictions, references)
    # Add loss to performance metrics
    val_metrics['loss'] = running_loss / len(dataloader)
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(predictions, references)
    # Return metrics and confusion matrix
    return val_metrics, conf_matrix

# Compare performance between the models
def compare_performance(metrics_list, model_to_train):
    """
    This function compares the performance metrics between multiple models.

    :param metrics_list: A list of performance metrics for each model.
    :type metrics_list: list
    :param model_to_train: A list of model names that were trained.
    :type model_to_train: list
    :return: None
    """
    
    print("Comparing performance:\n")
    # Initialize variables
    labels = ['Test accuracy', 'Test precision', 'Test recall', 'Test f1 score', 'Test loss']
    data = []
    # For each performance metric and model, add them to an array
    for i in range(len(metrics_list)):
        data.append(metrics_list[i])
    # Print metrics
    test_result = pd.DataFrame(data, columns=labels, index=model_to_train).round(4)
    print(test_result.transpose())

# Create the graph for performance metrics
def print_metrics_graph(training_metrics, validation_metrics, metric_plotted, view, type_model):
    """
    Prints the graph of a given metric for the training and validation datasets.

    :param training_metrics: The training metrics.
    :type training_metrics: List
    :param validation_metrics: The validation metrics.
    :type validation_metrics: List
    :param metric_plotted: The metric to be plotted.
    :type metric_plotted: String
    :param view: Whether to display the plot.
    :type view: bool
    :param type_model: The type of the model (e.g., 'ResNet', 'AlexNet', 'SVM').
    :type type_model: String
    """
    # Print the graph with for all epochs for training and validation for each performance metric
    # For deep learning models
    if type(training_metrics) == list and type(validation_metrics) == list:
        for element in metric_plotted:
            plt.plot([metrics[element] for metrics in training_metrics], label = 'Training')
            plt.plot([metrics[element] for metrics in validation_metrics], label = 'Validation')
            plt.legend()
            plt.title("Graph of " + str(element) + " per epoch for " + str(type_model) + " model:")
            # Improves graph visibility
            plt.tight_layout()
            save_graph(str('Graph of ' + str(element)), (str(type_model).capitalize() + ' model'))
            # Check if your configuration likes a print or not
            if view:
                plt.show()
            # Close the graph to avoid overlap
            plt.close()
    # For machine learning models
    else: 
        for element in metric_plotted:
            current_ticks = plt.gca().get_yticks()
            # Enter the values of the two bars in the vertical scale
            new_ticks = np.unique(np.concatenate((current_ticks, [training_metrics[element], validation_metrics[element]])))
            # Upgrade the vertical scale
            plt.gca().set_yticks(new_ticks)
            plt.bar([0], training_metrics[element], label = 'Training')
            plt.bar([1], validation_metrics[element], label = 'Validation')
            plt.legend()
            plt.title("Graph of " + str(element) + " for "+ str(type_model) + " model:")
            # Improves graph visibility
            plt.tight_layout()
            save_graph(str('Graph of ' + str(element)), (str(type_model).capitalize() + ' model'))
            # Check if your configuration likes a print or not
            if view:
                plt.show()
            # Close the graph to avoid overlap
            plt.close()

# Create the graph to compare performance of the models
def print_compare_graph(metrics_list, model_to_train, view, test):
    """
    This function creates graphs to compare the performance of multiple models.

    :param metrics_list: A list of performance metrics for each model.
    :type metrics_list: list
    :param model_to_train: A list of model names that were trained.
    :type model_to_train: list
    :param view: A boolean indicating whether to display the graphs.
    :type view: bool
    :param test: A boolean indicating whether the graphs are for testing results.
    :type test: bool
    :return: None
    """
    
    # Inizialize metris
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    # For each metric, print a graph
    for i in range(len(metrics)):
        # Inizialize an array
        counts = np.zeros(len(model_to_train))
        for j in range(len(model_to_train)):
            # Fill array with i-th performance metrics for each model
            counts[j] = metrics_list[j][i]
        # Create a bar graph
        plt.bar(model_to_train, counts, color=['blue', 'orange', 'red', 'yellow', 'purple'])
        # Set labels and title
        plt.xlabel("Models")
        plt.ylabel("Values")
        plt.title("Comparing " + metrics[i] + " per epoch for models:")
        # Retrieve the vertical scale of the graph
        current_ticks = plt.gca().get_yticks()
        # Enter the values of the three bars in the vertical scale
        new_ticks = np.unique(np.concatenate((current_ticks, counts)))
        # Upgrade the vertical scale
        plt.gca().set_yticks(new_ticks)
        # Horizontal dashed lines at bar levels, compute gap and set y position to locate the texts "Difference"
        first_gap = abs(counts[0] - counts[1])
        first_y_pos = max(counts) + 0.02 * max(counts)
        plt.hlines(counts[0], 0.4, 0.6, colors='green', linestyles='dashed')
        plt.hlines(counts[1], 0.4, 0.6, colors='green', linestyles='dashed')   
        # Create symbol <-> from minimum to maximum value of the graph (between first and second models)
        plt.annotate('',
                    xy=(0.4, counts[0]), xycoords='data',
                    xytext=(0.6, counts[1]), textcoords='data',
                    arrowprops=dict(arrowstyle='<->', color='red'))
        # Texts "Difference"
        plt.text(0.5, first_y_pos, f'Difference: {first_gap:.4f}', ha='center', va='center')
        # Only in the case where you are interested in comparing the performance of three models
        if len(model_to_train) == 3:
            # Horizontal dashed lines at bar levels, compute gap and set y position to locate the texts "Difference"
            second_gap = abs(counts[1] - counts[2])
            second_y_pos = max(counts) + 0.02 * max(counts)
            plt.hlines(counts[1], 1.4, 1.6, colors='green', linestyles='dashed')
            plt.hlines(counts[2], 1.4, 1.6, colors='green', linestyles='dashed')
            # Create symbol <-> from minimum to maximum value of the graph (between second and third models)
            plt.annotate('',
                        xy=(1.4, counts[1]), xycoords='data',
                        xytext=(1.6, counts[2]), textcoords='data',
                        arrowprops=dict(arrowstyle='<->', color='red'))
            # Texts "Difference"
            plt.text(1.5, second_y_pos, f'Difference: {second_gap:.4f}', ha='center', va='center')
        # Improves graph visibility
        plt.tight_layout()
        # Save the graph to a specific path
        if test:
            save_graph(str('Compare of ' + metrics[i]), 'Testing result')
        else:
            save_graph(str('Compare of ' + metrics[i]), 'Compare performance')
        # Check if your configuration likes a print or not
        if view:
            plt.show()
        # Close the graph to avoid overlap
        plt.close()

# Save the created graph
def save_graph(filename, type_of_graph):
    """
    Saves a graph to a file.

    :param filename: The name of the file to save the graph to.
    :type filename: String
    :param type_of_graph: The type of the graph (e.g., 'Testing result', 'Compare performance').
    :type type_of_graph: String
    """
    # Get the current path
    path = os.getcwd()
    # Check if the graph folder exists, if not, create it
    if not os.path.exists(os.path.join(path, 'graph')):
        os.makedirs(os.path.join(path, 'graph'))
    # Check if the type_of_graph subfolder exists, if not, create it
    if not os.path.exists(os.path.join((str(path) + '//graph'), type_of_graph)):
        os.makedirs(os.path.join((str(path) + '//graph'), type_of_graph))
    # Save the graph
    plt.savefig(str(str(path) + '//graph//' + str(type_of_graph) + '//' + str(filename) + '.png'))

# Create a graph for the confusion matrix
def print_confusion_matrix_graph(conf_matrix, view, type_model, test):
    """
    Prints a confusion matrix graph for a model.

    :param conf_matrix: The confusion matrix to plot.
    :type conf_matrix: numpy.ndarray
    :param view: Whether to display the plot.
    :type view: bool
    :param type_model: The type of the model (e.g., 'ResNet', 'AlexNet', 'SVM').
    :type type_model: String
    :param test: Whether the model is in the testing phase.
    :type test: Bool
    """
    # Select color
    sns.color_palette("YlOrBr", as_cmap=True)
    # Create a heatmap with confusion matrix
    sns.heatmap(conf_matrix, annot=True)
    # Set labels
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    # Improves graph visibility
    plt.tight_layout()
    # Save the graph to a specific path
    if test:
        save_graph(str(str(type_model).capitalize() + ' model\'s heatmap'), 'Testing result')
    else:
        save_graph('Heatmap of confusion matrix', str(str(type_model).capitalize() + ' model'))
    # Check if your configuration likes a print or not
    if view:
        plt.show()
    # Close the graph to avoid overlap
    plt.close()

# Print confusion matrix on the screen
def print_confusion_matrix(conf_matrix, type_model):
    """
    Prints a confusion matrix for a model.

    :param conf_matrix: The confusion matrix to print.
    :type conf_matrix: Numpy.ndarray
    :param type_model: The type of the model (e.g., 'ResNet', 'AlexNet', 'SVM').
    :type type_model: String
    """
    print("Confusion matrix for " + str(type_model) + " model:")
    # Based on the type of classification [binary or ternary], set the labels
    if len(conf_matrix[0]) == 2:
        labels = ['NORMAL', 'PNEUMONIA']
    else:
        labels = ['BACTERIA', 'NORMAL', 'VIRUS']
    # Print the confusion matrix with DataFrame
    df_confusion_matrix = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    print(df_confusion_matrix)

# Prints the best evaluation metrics found during the evaluation phase
def print_best_val_metrics(type_model, best_val_metric):
    """
    Prints the best model performance on the validation dataset.

    :param type_model: The type of the model.
    :type type_model: String
    :param best_val_metric: The best validation metrics.
    :type best_val_metric: Dictionary
    """
    print("Best " + type_model + " model performance on validation dataset:")
    # Print the performance of the best model based on the validation_dataset on which the test_dataset will be tested
    for key, value in best_val_metric.items():
        print("\t- Best " + type_model + f" model {key}: {value:.4f}")
    print("\nTesting " + type_model + " model on test dataset...")

# Print the result of the evaluation
def print_evaluation(test_metrics):
    """
    Prints the evaluation metrics and returns them as a list.

    :param test_metrics: The test metrics.
    :type test_metrics: Dictionary
    :return: The test metrics as a list.
    :rtype: List
    """
    # Store performance in lists so you can pass it to the DataFrame
    metrics = [test_metrics['accuracy'], test_metrics['precision'], test_metrics['recall'], test_metrics['f1'], test_metrics['loss']]
    labels = ['Test accuracy', 'Test precision', 'Test recall', 'Test f1 score', 'Test loss']
    # Print performance on the test_dataset
    test_result = pd.DataFrame(metrics, columns=['Value'], index=labels).round(4)
    print(test_result)
    return metrics

# Evaluate SVM model
def evaluate_svm(model, test_set, criterion):
    """
    Evaluates the SVM model.

    :param model: The SVM model.
    :type model: SVM class
    :param test_set: The test dataset.
    :type test_set: SVM_dataset
    :param criterion: The loss function to use for evaluation.
    :type criterion: Function
    :return: Test metrics and confusion matrix.
    :rtype: Tuple (test_metrics, conf_matrix)
    """
    # Compute metrics and confusion matrix for SVM model
    test_metrics, conf_matrix = model.evaluate_svm(test_set, criterion)
    # Return metrics and confusion matrix
    return test_metrics, conf_matrix

# Print scree graph after PCA on dataset features
def print_scree_graph(pca, test, view):
    """
    Prints the scree graph of the PCA.

    :param pca: The PCA object.
    :type pca: PCA
    :param test: Whether it's a test or not.
    :type test: Bool
    :param view: Whether to view the graph or not.
    :type view: Bool
    """
    # Get the percentage value of the variance
    explained_variance_ratio = pca.explained_variance_ratio_
    # Plot values on bars
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5,
            align='center', label='Eigenvalues of the principal components', color='g')
    # Set labels and legend
    plt.ylabel('Variance explained (%)')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    # Improves graph visibility
    plt.tight_layout()
    # Check the path where to save the graph
    if test:
        save_graph('PCA scree graph', 'Testing result')
    else:
        save_graph('PCA scree graph', 'SVM model')
    # Check if your configuration likes a print or not
    if view:
        plt.show()
    # Close the graph to avoid overlap
    plt.close()

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

# Extract the names of the models from their paths
def get_name(saved_models_path):
    """
    This function extracts the names of the models from their saved file paths.

    :param saved_models_path: A list of file paths where the models are saved.
    :type saved_models_path: list
    :return: Returns a list of model names based on the file paths.
    :rtype: list
    """
    
    name = []
    for path in saved_models_path:
        if "svm" in path.lower():
            name.append("SVM")
        elif "resnet" in path.lower():
            name.append("ResNet")
        elif "alexnet" in path.lower():
            name.append("AlexNet")
    return name

# Extract a list of metrics from a list of dictionaries
def extract_list_of_metrics(list_of_dictionary):
    """
    This function extracts a list of metrics from a list of dictionaries.

    :param list_of_dictionary: A list of dictionaries containing metrics.
    :type list_of_dictionary: list
    :return: Returns a list of metrics extracted from the dictionaries.
    :rtype: list
    """
    
    list_of_metrics = []
    for dictionary in list_of_dictionary:
        list_of_metrics.append(extract_value(dictionary))
    return list_of_metrics
