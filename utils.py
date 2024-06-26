# Third-party imports
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

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
