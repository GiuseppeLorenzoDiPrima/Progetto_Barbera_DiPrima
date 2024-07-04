# Third-party imports
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Local application/library specific imports
from utils import compute_metrics

# Class to define the SVM model
class SVM:
    """
    Initializes the SVM model.

    :param epochs: Number of training epochs.
    :type epochs: Int
    :param learning_rate: Learning rate for gradient descent.
    :type learning_rate: Float
    :param num_features: Number of features in the dataset.
    :type num_features: Int
    """
    def __init__(self, epochs, learning_rate, num_features):
        # Inizialize and set all variables
        self.learning_rate = learning_rate
        self.num_epochs = epochs
        self.weights = np.zeros(num_features)
        self.bias = 0
    
    # Define loss function
    def hinge_loss(self, dataset):
        """
        Calculates the Hinge Loss for an SVM model.

        :param dataset: The dataset with features and labels.
        :type dataset: SVM_dataset
        :return: The calculated Hinge Loss.
        :rtype: float
        """
        # Compute the predicted outputs
        scores = np.dot(dataset.features, self.weights) - self.bias
        # Compute the loss 
        loss = np.maximum(0, 1 - (scores * dataset.labels))
        # Return the loss
        return np.mean(loss)

    # Evaluate model performance
    def evaluate_svm(self, dataset, criterion):
        """
        Evaluates the SVM model.

        :param dataset: The dataset to evaluate.
        :type dataset: SVM_dataset
        :param criterion: The loss function to use for evaluation.
        :type criterion: Function
        :return: Evaluation metrics and confusion matrix.
        :rtype: tuple (metrics, conf_matrix)
        """
        # Evaluate whether the prediction is binary or ternary and compute predicted values
        if self.num_classes == 2:
            prediction = self.predict(dataset.features, 'binary')
        else:
            prediction = self.predict(dataset.features, 'ternary')
        # Compute the loss
        loss = criterion(dataset)
        # Compute metrics
        metrics = compute_metrics(prediction, dataset.labels)
        # Add loss to metrics
        metrics['loss'] = loss
        # Compute confusion matrix
        conf_matrix = confusion_matrix(prediction, dataset.labels)
        # Return metrix and confusion matrix
        return metrics, conf_matrix
    
    # Fit model to the data
    def fit(self, train, criterion, learning_rate):
        """
        Trains the SVM using gradient descent.

        :param train: The training dataset.
        :type train: SVM_dataset
        :param criterion: The loss function to use for training.
        :type criterion: Function
        :param learning_rate: The learning rate for gradient descent.
        :type learning_rate: Float
        :return: Training metrics.
        :rtype: Dictionary
        """
        # Set variables
        self.learning_rate = learning_rate
        self.num_classes = len(train.classes)
        # For each class
        for class_idx, _ in enumerate(tqdm(range(self.num_classes))):
            # Create labels for the current class (1 current and -1 others)
            labels = np.where(np.array(train.labels) == class_idx, 1, -1)
            # For each feature update weights and bias
            for idx, features in enumerate((train.features)):
                condition = labels[idx] * (np.dot(features, self.weights) - self.bias) >= 1
                # Correct prediction
                if condition:
                    self.weights -= self.learning_rate * (2 * 1/self.num_epochs * self.weights)
                # Wrong prediction
                else:
                    self.weights -= self.learning_rate * (2 * 1/self.num_epochs * self.weights - np.dot(features, labels[idx]))
                    self.bias -= self.learning_rate * labels[idx]
        # Compute metrics and confusion matrix
        train_metrics, train_conf_matrix = self.evaluate_svm(train, criterion)
        # Return train metrics
        return train_metrics

    # Compute predicted values
    def predict(self, features, classification):
        """
        Predicts labels for input data.

        :param features: Input features.
        :type features: np.ndarray
        :param classification: Type of classification ('binary' or 'ternary').
        :type classification: String
        :return: Predicted labels.
        :rtype: np.ndarray
        """
        # Compute predicted values
        scores = np.dot(features, self.weights) - self.bias
        # Binary case
        if classification == 'binary':
            # Approximates integer values ​​0 or 1
            value = np.sign(scores)
            value = np.where(value <= 0.5, 0, 1)
        # Ternary case
        else: 
            # Approximates integer values ​​0, 1 or 2
            value = np.where(scores < 0.5, 0, np.where(scores < 1.125, 1, 2))
        return value
