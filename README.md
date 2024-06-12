## Machine Learning Project Chest X-Ray

This is a guide to assist in reading the “Chest X_Ray” project related to the machine learning exam for the Kore University of Enna.

| | |
| --- | --- |
| **Description** | Template for a machine learning project |
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
  - [License](#license)

---

### Introduction

The goal of the project is to correctly classify, through the use of a convolutional neural network, images related to X-ray radiographs of the chests of children.

[Chest X-ray images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou.

The project is divided into two main scripts:
- `train.py` for training the model.
- `test.py` for testing the model.

The dataset is managed by the `manage_dataset.py` class, while the model is defined in the `ff_model.py` class.

> [!IMPORTANT]  
> To reproduce the project, you need to run the following commands:


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

---

### Code structure

The code is structured as follows:

```
main_repository/
│
├── data_classes/
│   ├── manage_dataset.py
|
├── model_classes/
│   ├── manage_dataset.py
|
├── prepare.sh
├── requirememts.txt
├── README.md
├── utils.py
│
├── train.py
├── test.py
```

- `data_classes/` contains the classes for managing the dataset.
- `model_classes/` contains the classes for the model design.
- `train.py` is the script for training the model.
- `test.py` is the script for testing the model.
- `utils.py` is the script that evaluates the performance metrics.
- `prepare.sh` is a script for setting up the environment - at the moment it only installs the requirements.
- `requirements.txt` contains the list of dependencies for the project.
- `README.md` is the file you are currently reading.

---

### License

This project is licensed under the terms of the MIT license. You can find the full license in the `LICENSE` file.
