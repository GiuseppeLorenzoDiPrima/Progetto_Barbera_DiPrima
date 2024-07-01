# Third-party imports
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tkinter

# Standard library imports
import os

# Local application/library specific imports
from data_classes.manage_dataset import *

def compute_metrics(predictions, references):
    acc = accuracy_score(references, predictions)
    precision = precision_score(references, predictions, average='macro', zero_division=0.0)
    recall = recall_score(references, predictions, average='macro', zero_division=0.0)
    f1 = f1_score(references, predictions, average='macro', zero_division=0.0)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    references = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            predictions.extend(pred.cpu().numpy())
            references.extend(labels.cpu().numpy())
    val_metrics = compute_metrics(predictions, references)
    val_metrics['loss'] = running_loss / len(dataloader)
    conf_matrix = confusion_matrix(predictions, references)
    return val_metrics, conf_matrix

def compare_performance(first_metrics, second_metrics):
    print("Comparing performance:\n(Difference = First_model - Second_model)\n")
    labels = ['Test accuracy', 'Test precision', 'Test recall', 'Test f1 score', 'Test loss']
    difference = []
    data = []
    for i in range(len(labels)):
        difference.append((first_metrics[i] - second_metrics[i]))
        data.append([first_metrics[i], second_metrics[i], difference[i]])
    test_result = pd.DataFrame(data, columns=['First_model', 'Second_model', 'Difference'], index=labels).round(4)
    print(test_result)

def print_metrics_graph(training_metrics, validation_metrics, metric_plotted, view, type_model):
    for element in metric_plotted:
        plt.plot([metrics[element] for metrics in training_metrics], label = 'Training')
        plt.plot([metrics[element] for metrics in validation_metrics], label = 'Validation')
        plt.legend()
        plt.title("Graph of " + str(element) + " per epoch for " + str(type_model) + " model:")
        save_graph(str('Graph of ' + str(element)), (str(type_model).capitalize() + ' model'))
        if view:
            plt.show()
        plt.close()
            
def print_compare_graph(first_metrics, second_metrics, view, test):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    for i in range(len(metrics)):
        counts = np.zeros(2)
        counts[0] = first_metrics[i]
        counts[1] = second_metrics[i]
        plt.bar(['First model', 'Second model'], counts, color=['blue', 'orange', 'red', 'yellow', 'purple'])
        plt.xlabel("Models")
        plt.ylabel("Values")
        plt.title("Comparing " + metrics[i] + " per epoch for the two model:")
        current_ticks = plt.gca().get_yticks()
        new_ticks = np.unique(np.concatenate((current_ticks, counts)))
        plt.gca().set_yticks(new_ticks)
        # Linee tratteggiate orizzontali ai livelli delle barre
        plt.hlines(counts[0], 0.4, 0.6, colors='green', linestyles='dashed')
        plt.hlines(counts[1], 0.4, 0.6, colors='green', linestyles='dashed')
        # Annotazione con freccia verticale
        gap = abs(counts[0] - counts[1])
        # Calcolo della posizione y per l'annotazione
        y_pos = max(counts) + 0.02 * max(counts)
        plt.annotate('',
                    xy=(0.4, counts[0]), xycoords='data',
                    xytext=(0.6, counts[1]), textcoords='data',
                    arrowprops=dict(arrowstyle='<->', color='red'))
        # Testo che indica la differenza
        plt.text(0.5, y_pos, f'Difference: {gap:.4f}', ha='center', va='center')
        if test:
            save_graph(str('Compare of ' + metrics[i]), 'Testing result')
        else:
            save_graph(str('Compare of ' + metrics[i]), 'Compare performance')
        if view:
            plt.show()
        plt.close()

def save_graph(filename, type_of_graph):
    path = os.getcwd()
    if not os.path.exists(os.path.join(path, 'graph')):
        os.makedirs(os.path.join(path, 'graph'))
    if not os.path.exists(os.path.join((str(path) + '//graph'), type_of_graph)):
        os.makedirs(os.path.join((str(path) + '//graph'), type_of_graph))
    plt.savefig(str(str(path) + '//graph//' + str(type_of_graph) + '//' + str(filename) + '.png'))

def print_confusion_matrix_graph(conf_matrix, view, type_model, test):
    sns.color_palette("YlOrBr", as_cmap=True)
    sns.heatmap(conf_matrix, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    if test:
        save_graph(str(str(type_model).capitalize() + ' model\'s heatmap'), 'Testing result')
    else:
        save_graph('Heatmap of confusion matrix', str(str(type_model).capitalize() + ' model'))
    if view:
        plt.show()
    plt.close()

def print_confusion_matrix(conf_matrix, type_model):
    print("Confusion matrix for " + str(type_model) + " model:")
    if len(conf_matrix[0]) == 2:
        labels = ['NORMAL', 'PNEUMONIA']
    else:
        labels = ['BACTERIA', 'NORMAL', 'VIRUS']
    df_confusion_matrix = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    print(df_confusion_matrix)
