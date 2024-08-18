# Third-party imports
from transformers import ViTImageProcessor, ViTModel
import torch
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from joblib import dump

# Standard library imports
from PIL import Image
import os
import logging
import warnings

# Local application/library specific imports
from utils import print_scree_graph


#Prevents warnings from being printed during execution
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


# Class that defines the SVM dataset
class SVM_dataset:
    """
    Dataset used by the svm model.
    """
    def __init__(self, classes, targets, path):
        """
        Dataset used by the svm model.

        :param classes: List of class labels.
        :type classes: List
        :param targets: List of target values.
        :type targets: List
        :param path: Path to the dataset.
        :type path: String
        """
        # Set variables
        self.classes = classes
        self.targets = targets
        self.path = path
        # Inizialize variables
        self.data = []
        self.features = None
        self.labels = None
        self.num_of_features = 0
        self.num_of_samples = 0


# Class used to define embidding methods
class VisionEmbeddings:
    """
    This class is intended to extract embeddings from vision models.
    """
    
    def __init__(self, model_name='google/vit-base-patch16-224', device='cuda'):
        """
        This class is intended to extract embeddings from vision models.
        It uses ViT (Vision Transformer) as a default model.

        :param model_name: Name of the pre-trained ViT model.
        :type : String
        :param device: Device (e.g., 'cuda' or 'cpu') for inference.
        :type : String
        """
        # Set model to extract embeddings
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        # Set device and model name
        self.device = device
        self.model.to(self.device)
        self.model_name = model_name
        # Set evaluation mode
        self.model.eval()
        
    # Denormalize the tensors normalized by the transoform function when loading the ChestXray dataset
    def denormalize(self, tensor, mean, std):
        """
        Denormalizes a tensor using mean and standard deviation.

        :param tensor: Input tensor.
        :type tensor: torch.Tensor
        :param mean: List of mean values.
        :type mean: List
        :param std: List of standard deviation values.
        :type std: List
        :return: Denormalized tensor.
        :rtype: torch.Tensor
        """
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    
    # Starting from a dataset of tensors, obtain the same dataset with features thanks to the embeddings method
    def extract(self, dataset, dataset_type):
        """
        Extracts embeddings from the dataset.

        :param dataset: Input dataset.
        :type dataset: SVM_dataset
        :param dataset_type: Type of the dataset (e.g., 'train', 'test').
        :type dataset_type: String
        :return: New dataset with extracted features.
        :rtype: SVM_dataset
        """
        # Create a new SVM_dataset object
        new_dataset = SVM_dataset(dataset.classes, dataset.targets, dataset.path)
        # For each element of the dataset
        for idx, sample in enumerate(tqdm(dataset, desc='Vision embeddings for ' + dataset_type)):
            # Denormalize the image tensor
            image = sample['image']
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image = self.denormalize(image, mean, std)
            # Extract features from the denormalized tensor
            inputs = self.feature_extractor(images=image, return_tensors="pt", do_rescale=False)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
            # Convert the result to a list
            features = features.flatten().tolist()
            new_item = {
                'feature': features,
                'label': sample['label']
            }
            # Adds the list containing the features
            new_dataset.data.append(new_item)
        # Return dataset
        return new_dataset
    
    # Gets features from the dataset
    def get_features(self, dataset):
        """
        Gets features from the dataset.

        :param dataset: Input dataset.
        :type dataset: SVM_dataset
        :return: Features as a numpy array.
        :rtype: np.ndarray
        """
        # Inizialize a list
        items = []
        # Add features to a list (list of list)
        for item in dataset.data:
            items.append(item['feature'])
        features = np.array(items)
        # Return features of the dataset
        return features
        
    # Gets labels from the dataset
    def get_labels(self, dataset):
        """
        Gets labels from the dataset.

        :param dataset: Input dataset.
        :type dataset: SVM_dataset
        :return: List of labels.
        :rtype: List
        """
        # Inizialize a list
        labels = []
        # Add labels to a list
        for item in dataset.data:
            labels.append(item['label'])
        # Return labels of the dataset
        return labels
    
    # Extracts the attributes of a dataset
    def extract_from_dataset(self, dataset):
        """
        Extracts features, labels, number of samples and number of features from the dataset.

        :param dataset: Input dataset.
        :type dataset: SVM_dataset
        :return: Features, labels, number of features, and number of samples.
        :rtype: tuple
        """
        # Compute attributes
        features = self.get_features(dataset)
        labels = self.get_labels(dataset)
        num_of_samples, num_of_features = features.shape
        # Return attributes
        return features, labels, num_of_features, num_of_samples
    
    # Calculate the new dataset obtained by starting from a dataset of tensors
    def extract_single_dataset(self, dataset, pca, type_of_dataset, create, view):
        """
        Extracts features from a single dataset.

        :param dataset: Input dataset.
        :type dataset: SVM_dataset
        :param pca: Principal Component Analysis object.
        :type pca: PCA
        :param type_of_dataset: Type of the dataset (e.g., 'train', 'test').
        :type type_of_dataset: String
        :param create: Create for scree graph.
        :type create: Bool
        :param view: View for scree graph.
        :type view: Bool
        :return: New dataset with transformed features.
        :rtype: SVM_dataset
        """
        # Compute attributes
        new_dataset = self.extract(dataset, dataset_type=type_of_dataset)
        new_dataset.features, new_dataset.labels, new_dataset.num_of_features, new_dataset.num_of_samples = self.extract_from_dataset(new_dataset)
        # Dimensionality reduction
        new_dataset.features = pca.transform(new_dataset.features)
        # Re-compute attributes
        new_dataset.num_of_samples, new_dataset.num_of_features = new_dataset.features.shape
        if create:
            # Create scree graph
            print_scree_graph(pca, True, view)
        # Return dataset
        return new_dataset

    # Calculate all new datasets obtained by starting from datasets of tensors
    def extract_all_datasets(self, train, validation, test, num_features, path, create, view):
        """
        Extracts the training, validation, and test datasets, applies PCA, and performs oversampling.

        :param train: Training dataset.
        :type train: SVM_dataset
        :param validation: Validation dataset.
        :type validation: SVM_dataset
        :param test: Test dataset.
        :type test: SVM_dataset
        :param num_features: Number of components to retain after PCA.
        :type num_features: Int
        :param path: Path to save the PCA object.
        :type path: String
        :param create: Create for the scree graph.
        :type create: Bool
        :param view: View for the scree graph.
        :type view: Bool
        :return: New training, validation, and test datasets with transformed features.
        :rtype: tuple (new_train, new_validation, new_test)
        """
        # Extract the three sets
        new_train = self.extract(train, dataset_type='train')
        new_validation = self.extract(validation, dataset_type='validation')
        new_test = self.extract(test, dataset_type='test')
        # Compute attributes for the three datasets
        new_train.features, new_train.labels, new_train.num_of_features, new_train.num_of_samples = self.extract_from_dataset(new_train)
        new_validation.features, new_validation.labels, new_train.num_of_features, new_validation.num_of_samples = self.extract_from_dataset(new_validation)
        new_test.features, new_test.labels, new_train.num_of_features, new_test.num_of_samples = self.extract_from_dataset(new_test)
        # --- PCA ---
        # Set the number of components to keep after PCA
        n_components = num_features
        # Create a PCA object
        pca = PCA(n_components=n_components)
        # Fit PCA to training data
        pca.fit(new_train.features)
        # Transform all three datasets
        new_train.features = pca.transform(new_train.features)
        new_validation.features = pca.transform(new_validation.features)
        new_test.features = pca.transform(new_test.features)
        if create:
            # Create scree graph
            print_scree_graph(pca, False, view)
        
        # --- Oversampling ---
        # Create the smote object
        smote = SMOTE()
        # Oversample the training dataset
        new_train.features, new_train.labels = smote.fit_resample(new_train.features, new_train.labels)
        
        # Re-compute attributes for the three datasets
        new_train.num_of_samples, new_train.num_of_features = new_train.features.shape
        new_validation.num_of_samples, new_validation.num_of_features = new_validation.features.shape
        new_test.num_of_samples, new_test.num_of_features = new_test.features.shape
        # Store the pca object in the "../pca.joblib" file
        dump(pca, f"{path}/pca.joblib")
        # Return new datasets
        return new_train, new_validation, new_test
