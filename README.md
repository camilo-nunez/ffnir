# Feature-Fusion Neck Model for Content-Based Histopathological Image Retrieval

## Abstract
Feature descriptors in histopathological images pose a significant challenge for the implementation of Content-Based Image Retrieval (CBIR) systems, which are essential tools for assisting pathologists. The complexity arises from the diverse types of tissues and the high dimensionality of Whole Slide Images. Deep learning models like Convolutional Neural Networks and Vision Transformers improve the extraction of these feature descriptors. These models typically generate embeddings by leveraging deeper single-scale linear layers or advanced pooling layers. However, these embeddings, by focusing on local spatial details at a single scale, miss out on the richer spatial context from earlier layers. This gap, pointing towards the development of methods that incorporate multi-scale information to enhance the depth and utility of feature descriptors in histopathological image analysis. In this work, we propose the Local-Global Feature Fusion Embedding Model, an approach composed of a pre-trained backbone for feature extraction from multi-scales, a neck branch for local-global feature fusion, and a Generalized Mean (GeM)-based pooling head for feature descriptors. Based on our experiments, the model's neck and head were trained on ImageNet-1k and PanNuke datasets employing the Sub-center ArcFace loss and compared with the state-of-the-art Kimia Path24C dataset for histopathological image retrieval, achieving a Recall@1 of 99.40% for test patches.

## Usage

### Installation

#### Docker Environment
We highly recommend using the official [Nvidia Docker PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). We used version `24.01-py3`.

You can use this example code to execute the container and start a Jupyter Lab session:

```bash
docker run --name ffnir-pytorch-01 --gpus all --ipc=host -d -p 8888:8888 -p 6006:6006 -v /PATH-TO-YOUR-DATASETS:/datasets nvcr.io/nvidia/pytorch:24.01-py3 jupyter lab --allow-root --ip='*' --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.allow_origin='*' --notebook-dir=/
```

In this case, we have two ports: (I) port 8888 for the Jupyter Lab service and (II) port 6006 for the Tensorboard service. Additionally, we share the folder with the dataset used in this work.

#### Requirements
Before installing the required Python packages, it's necessary to install the external dependencies for the [PathML](https://pathml.org) package.

As the root user in the container, install these dependencies:

```bash
apt-get install openslide-tools g++ gcc libblas-dev liblapack-dev openjdk-8-jre openjdk-8-jdk
```

Then, install the Python requirements using:

```bash
pip install -r requirements.txt
```

### Download the Datasets
Our neck model and GeM head are trained using three different strategies, utilizing the public training dataset ImageNet-1k, which consists of 1,281,167 training images and 1,000 object classes. Additionally, we employed the PanNuke dataset with the PathML toolbox, containing 189,744 segmented nuclei and encompassing 19 different types of tissues. Additionally, Kimia Patch24C, a training dataset consisting of 22,591 training patches from 24 WSIs representing various tissues, was employed.

#### Download ImageNet-1k
We recommend downloading the train set `ILSVRC2012_img_train.tar` using [academic torrent](https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2). Also, you must download the `ILSVRC2012_devkit_t12.tar.gz` file using the command:

```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
```

These two files must be in the same folder. Do not decompress the files because the dataset class [`ImageNet`](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html) from PyTorch will do it for you.

#### Download PanNuke
Since we use the [`PanNukeDataModule`](https://pathml.readthedocs.io/en/latest/api_datasets_reference.html#pannuke) from PathML, you can download the complete dataset by setting the parameters `download=True` and `data_dir=/PATH-TO-YOUR-DATASETS`.

#### Download Kimia Patch24C
To download this dataset, use the official website [https://tizhoosh.com/download-corner/kimia-path24c-dataset/](https://tizhoosh.com/download-corner/kimia-path24c-dataset/).

### Train files
Actually, we use three different training files for the three datasets used: (I) `train_in1k.py`, `train_pannuke.py`, and `train_kimia.py`. The obligatory arguments available in the training files are:

- `--cfg_model_backbone`: Path to the BACKBONE config file. Must be a YAML file. The available files are in the folder `config/files/model/backbone`. We used the config file `11_convnextv2.yaml` in our experiment.
- `--cfg_model_neck`: Path to the NECK config file. Must be a YAML file. The available files are in the folder `config/files/model/neck`. We used the config file `02_neck_512_3.yaml` in our experiment.
- `--cfg_model_head`: Path to the HEAD config file. Must be a YAML file. The available files are in the folder `config/files/model/head`. We used the config file `00_head_A.yaml` in our experiment.