# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from joblib import load

# Local application/library specific imports
from data_classes.manage_dataset import ChestXrayDataset
from model_classes.resnet_model import ResNet, ResidualBlock
from model_classes.alexnet_model import AlexNet
from utils import *
from extract_representations.vision_embeddings import VisionEmbeddings
from sklearn import svm
from sklearn.metrics import hinge_loss, log_loss

# Configuration and utility imports
from yaml_config_override import add_arguments
from addict import Dict

# Print test set performance metrics
def print_metrics(list_of_metrics, saved_models):
    """
    Prints the metrics.

    :param list_of_metrics: The list of metrics to print.
    :type list_of_metrics: list of dict
    :param saved_models: The list of saved model names.
    :type saved_models: list of str
    """
    for idx, metrics in enumerate(list_of_metrics):
        print("\n" + saved_models[idx] + " model performance:\n")
        # Scrolls through the dictionary and prints performance metrics
        for key, value in metrics.items():
            print(f"Test {key}: {value:.4f}")

# Test the machine learning model
def test_ml_model(model_name, config, device, test_dataset):
    """
    This function tests a machine learning model on a test dataset.

    :param model_name: The name of the model to be tested (e.g., 'SVM').
    :type model_name: str
    :param config: The configuration settings for testing the model.
    :type config: object
    :param device: The device on which to test the model (e.g., 'cpu', 'cuda').
    :type device: str
    :param test_dataset: The dataset used for testing the model.
    :type test_dataset: torch.utils.data.Dataset
    :return: Returns the metrics for the test dataset.
    :rtype: dict
    """
    
    # ---------------------
    # 1. Compute dataset for svm
    # ---------------------
    
    print("Vision embeddings for SVM:\n")
    # Load the pca object determined during the training phase
    pca = load(f"{config.training.checkpoint_dir}/pca.joblib")
    # Create vision_embedding object
    vision_embeddings = VisionEmbeddings()
    # Create the dataset containing features for the svm model
    test_dataset_svm = vision_embeddings.extract_single_dataset(test_dataset, pca, 'test', config.graph.create_model_graph, config.graph.view_model_graph)
    print("---------------------")
    
    # ---------------------
    # 2. Load model
    # ---------------------
    
    # Load the templates and specify their configuration through the config variable
    # SVM model
    svm_model = svm.SVC(
        gamma=config.svm_training.gamma,
        kernel=config.svm_training.kernel,
        C=config.svm_training.C,
        probability=config.svm_training.probability
    )
    
    # ---------------------
    # 3. Load model weights
    # ---------------------
    
    # Loads the saved model weights to the specified folder during training
    print("Loading " + model_name + " model...")
    # SVM model
    svm_model = load(f"{config.training.checkpoint_dir}/{model_name}_best_model.pkl")
    print("-> " + model_name + " model loaded.")
    print("---------------------")
    
    # ---------------------
    # 4. Evaluate
    # ---------------------
    
    print("Evaluating model..\n")
    # Evaluate the performance of the SVM model on the test_dataset
    svm_metrics = compute_metrics(test_dataset_svm.labels, svm_model.predict(test_dataset_svm.features))
    # If it is a binary classification, use hinge_loss as the loss function; otherwise, use log_loss
    if config.classification.type.lower() == 'binary':
        # Compute the loss using hinge_loss
        svm_metrics['loss'] = hinge_loss(test_dataset_svm.labels, svm_model.predict(test_dataset_svm.features))
    else:
        # Compute the loss using log_loss
        svm_metrics['loss'] = log_loss(test_dataset_svm.labels, svm_model.predict_proba(test_dataset_svm.features))
    # Compute the confusion matrix for test
    svm_conf_matrix = confusion_matrix(test_dataset_svm.labels, svm_model.predict(test_dataset_svm.features))
    # Prints the confusion matrix of SVM model
    print_confusion_matrix(svm_conf_matrix, type_model=model_name)
    print("---------------------")
    # Depending on the configuration you choose, create graphs for confusion matrix
    if config.graph.create_model_graph:
        print_confusion_matrix_graph(svm_conf_matrix, config.graph.view_model_graph, type_model=model_name, test=True)
        
    return svm_metrics

# Test the deep learning model
def test_dl_model(model_name, config, device, test_dataset):
    """
    This function tests a deep learning model on a test dataset.

    :param model_name: The name of the model to be tested (e.g., 'ResNet', 'AlexNet').
    :type model_name: str
    :param config: The configuration settings for testing the model.
    :type config: object
    :param device: The device on which to test the model (e.g., 'cpu', 'cuda').
    :type device: str
    :param test_dataset: The dataset used for testing the model.
    :type test_dataset: torch.utils.data.Dataset
    :return: Returns the metrics for the test dataset.
    :rtype: dict
    """
    
    # ---------------------
    # 1. Load data
    # ---------------------
    
    # Loading the test_dataset
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.deep_learning_training.batch_size,
        shuffle=False # Without shuffling the data
    )
    
    # ---------------------
    # 2. Load model
    # ---------------------
    
    # Load the templates and specify their configuration through the config variable
    if 'resnet' in model_name.lower():
        # ResNet Model
        model = ResNet(
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
        model.to(device)
    # AlexNet Model
    else:
        model = AlexNet(
            config.classification.type,
            config.AlexNet_model.stride,
            config.AlexNet_model.padding,
            config.AlexNet_model.kernel,
            config.AlexNet_model.channels_of_color,
            config.AlexNet_model.inplace,
        )
        model.to(device)

    # ---------------------
    # 3. Load model weights
    # ---------------------
    
    # Loads the saved model weights to the specified folder during training
    print("Loading " + model_name + " model...")
    model.load_state_dict(torch.load(f"{config.training.checkpoint_dir}/{model_name}_best_model.pt"))
    print("-> " + model_name + " model loaded.")
    print("---------------------")
    
    # ---------------------
    # 4. Criterion
    # ---------------------
    
    # Defines the CrossEntropyLoss as loss functions for deep learning model
    criterion = nn.CrossEntropyLoss()
    
    # ---------------------
    # 5. Evaluate
    # ---------------------
    
    print("Evaluating model...\n")
    # Evaluate model performance
    metrics, conf_matrix = evaluate(model, test_dl, criterion, device)
    # Prints the confusion matrix of the model
    print_confusion_matrix(conf_matrix, type_model=model_name)
    print("---------------------")
    # Print confusion matrices graphs
    if config.graph.create_model_graph:
        print_confusion_matrix_graph(conf_matrix, config.graph.view_model_graph, type_model=model_name, test=True)
    
    return metrics


# Main
if __name__ == '__main__':
    """
    The main script for testing the models.

    The script performs the following steps:
    
    1. Load configuration
    2. Set device
    3. Load data
    4. Get saved model
    5. Test on saved models
    6. Print performance    
    7. Compare performance
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
    # Print selected device
    print("\nDevice: " + torch.cuda.get_device_name()) 
    print("---------------------")

    # ---------------------
    # 3. Load data
    # ---------------------
    
    # Create the test_dataset item
    test_dataset  = ChestXrayDataset(type='test', root=config.data)
    
    # ---------------------
    # 4. Get saved model
    # ---------------------
    
    # Get the current path
    path = os.getcwd()
    if not os.path.exists(os.path.join(path, config.training.checkpoint_dir)):
        os.makedirs(os.path.join(path, config.training.checkpoint_dir))
    # Get path of saved models
    saved_models_path = os.listdir(os.path.join(path, config.training.checkpoint_dir))
    # Get name of saved models
    saved_models = get_name(saved_models_path)
    
    # ---------------------
    # 5. Test on saved models
    # ---------------------
    
    metrics_list = []
    # Perform the intersection between the list config.model.model_to_test and saved_models to store only the models that have already been trained and that you want to test
    model_to_test = [model for model in saved_models if model in config.model.model_to_test]
    for model in model_to_test:
        # Test SVM model
        if 'svm' in model.lower():
            metrics = test_ml_model(model, config, device, test_dataset)
        # Test deep learning models
        else:
            metrics = test_dl_model(model, config, device, test_dataset)
            
        # Store the performances in a list
        metrics_list.append(metrics)

    # ---------------------
    # 6. Print performance
    # ---------------------
    
    print("Performance:")
    print_metrics(metrics_list, model_to_test)
    print("---------------------")
         
    # ---------------------
    # 7. Compare performance
    # ---------------------
    
    # Extract metrics
    values = extract_list_of_metrics(metrics_list)
    
    # The comparison makes sense if at least two models are tested
    if len(model_to_test) > 1:
        # Compare performance
        compare_performance(values, model_to_test)
        print("---------------------")
        
        # Print performance comparison results
        if config.graph.create_compare_graph:
            print_compare_graph(values, model_to_test, config.graph.view_compare_graph, test=True)

    print("\nTest finish correctly.\n")
