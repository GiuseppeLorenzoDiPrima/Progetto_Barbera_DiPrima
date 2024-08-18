## Machine Learning Project Chest X-Ray

This is a guide to assist in reading the “Chest X_Ray” project related to the machine learning exam for the Kore University of Enna.

| | |
| --- | --- |
| **Description** | Machine learning project Chest X-Ray |
| **Authors** | Barbera Antonino e Di Prima Giuseppe Lorenzo |
| **Course** | [Machine Learning @ UniKore](https://unikore.it) |
| **License** | [MIT](https://opensource.org/licenses/MIT) |

---

### Table of Contents

- [Machine Learning Project Chest X-Ray](#machine-learning-project-template)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Code structure](#code-structure)
  - [Use of base_config.yaml file](#Use-of-base_config.yaml-file)
  - [Documentation](#Documentation)
  - [License](#license)

---

### Introduction

The goal of the project is to correctly classify, through the use of a convolutional neural network and artificial neural network, images related to X-ray radiographs of the chests of children.
The project presents two convolutional neural networks, respectively ResNet and AlexNet, and a Support Vector Machine model, whose performance will be subsequently compared.

[Chest X-ray images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou.

The project is divided into two main scripts:
- `train.py` for training the model.
- `test.py` for testing the model.

The dataset is managed by the `manage_dataset.py` class, while the three models are defined in the `resnet_model.py`, `alexnet_model.py` and `svm_model.py` class.

> [!IMPORTANT]  
> To reproduce the project, you need to run the following commands to include the configuration file:
>
>\>>> python -u train.py -c config/base_config.yaml
>
> Replace "train.py" with "test.py" to evaluate performance on the test dataset after training the model

The main idea is that, the project can be reproduced by running the following commands:

```bash
git clone https://github.com/GiuseppeLorenzoDiPrima/Esame_Barbera_Di_Prima.git
cd Esame_Barbera_Di_Prima
bash prepare.sh
python train.py
python test.py
```

> [!CAUTION]
>  You must have a version of git installed that is compatible with your operating system to perform the git clone operation.

---

The `prepare.sh` script is used to install the requirements for the project and to set up the environment (e.g. download the dataset)

### Requirements

The project is based on **Python 3.12.3** - one of the latest versions of Python at the time of writing.

- The requirements are listed in the `requirements.txt` file and can be installed using `pip install -r requirements.txt`.

This project is based on the following libraries:
- `torch` for PyTorch.
- `torchvision` for PyTorch vision.
- `yaml_config_override` for configuration management.
- `addict` for configuration management.
- `tqdm` for progress bars.
- `scikit-learn` as a newer version of sklearn
- `sklearn` for machine learning algorithms
- `numpy` for numerical computing
- `matplotlib` for data visualization
- `pandas` for data manipulation
- `shutil` for file and directory manipulation
- `seaborn` for confusion matrix
- `transformers` for embeddings
- `logging` to hid warnings
- `joblib` to save and load models
---

### Code structure

The code is structured as follows:

```
main_repository/
|
├── config/
|   ├── base_config.yaml
|
|
├── data_classes/
|   ├── manage_dataset.py
|
|
├── docs
|   ├── _modules/..
|   ├── _sources/..
|   ├── _static/..
|   ├── .buildinfo
|   ├── data_classes.html
|   ├── extract_representations.html
|   ├── genindex.html
|   ├── index.html
|   ├── model_classes.html
|   ├── modules.html
|   ├── objects.inv
|   ├── py-modindex.html
|   ├── search.html
|   ├── searchindex.js
|   ├── test.html
|   ├── train.html
|   ├── utils.html
|
|
├── extract_representations/
|   ├── vision_embeddings.py
|
|
├── model_classes/
|   ├── resnet_model.py
|   ├── alexnet_model.py
|   ├── svm_model.py
|
|
├── LICENCE
├── prepare.sh
├── README.md
├── requirememts.txt
├── utils.py
|
├── train.py
├── test.py
```

- `config/` contains the configuration parameters.
- `data_classes/` contains the classe for managing the dataset.
- `docs/` contains project documentation.
- `extract_representations/` contains classes to manage embeddings for the SVM model.
- `model_classes/` contains the classes for the models design.
- `LICENCE/` contains the license to use the project.
- `prepare.sh` is a script for setting up the environment installing the requirements.
- `README.md` is the file you are currently reading.
- `requirements.txt` contains the list of dependencies for the project.
- `utils.py` is the script that evaluates the performance metrics, print them and contain other useful functions.
- `train.py` is the script for training the models.
- `test.py` is the script for testing the models.


> [!IMPORTANT]  
> After executing the script train.py to the entire local running directory, additional folders will be generated: checkpoints, dataset and graph

- `checkpoints/` this folder contains two files in .pt format that represent the saving of the weights of the best models found during training.
- `dataset/` this folder contains the dataset already divided into train, val, and test. Based on the choice of classification (binary or ternary) each folder will be divided into the appropriate classes.
- `graph/` this folder contains the graphs that originated from the manipulation of the dataset, the training, comparison, and testing phases .

> [!CAUTION]
>  In reference to what has been said regarding the possibility of being able to proceed with both binary and ternary classifications, it is advisable to pay close attention to the testing phase. It is possible to test the obtained model only on a dataset of the same type. A trained binary model can therefore be tested on the binary dataset while the same cannot be said on a possible ternary test dataset and vice versa.
---

### Use of base_config.yaml file
Through the use of the base_config.yaml file, it is possible to modify the configuration parameters related to the training of the model. Here are just a few of the most common examples:

- `classification.type` you can choose between a binary ([NORMAL] [PNEUMONIA]) or ternary ([NORMAL] [BACTERIA] [VIRUS]) classification.
- `create_dataset_graph` you can choose to create [TRUE] or not [FALSE] dataset graph.
- `view_dataset_graph` you can choose to view [TRUE] or not [FALSE] dataset graph during execution.
- `model_to_train` you can choose the model you want to train.
- `model_to_test` you can choose the model you want to test if they have been correct trained yet.
- `create_model_graph` you can choose to create [TRUE] or not [FALSE] models graph.
- `view_model_graph` you can choose to view [TRUE] or not [FALSE] models graph.
- `create_compare_graph` you can choose to create [TRUE] or not [FALSE] compare graph.
- `view_compare_graph` you can choose to view [TRUE] or not [FALSE] compare graph.
- `metric_plotted_during_traininig` you can select the only performance metrics you prefer to view.
- `epochs` you can choose number of epochs based on your device computing capacity.
- `early_stopping_metric` you can choose the metric against which you want to check for performance improvement according to early stopping.
- `earling_stopping_max_no_valid_epochs` you can choose the max value of epochs that do not produce performance improvement.
- `evaluation_metric` you can choose the performace metric by which you want to evaluate your performance.
- `many other...`
---

### Documentation
Inside the docs folder you can open the index.html file to access the documentation of this project. The file will be opened through the default browser set by the user.

---

### License
This project is licensed under the terms of the MIT license. You can find the full license in the `LICENSE` file.
