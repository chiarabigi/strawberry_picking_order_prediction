# Strawberry Picking Scheduling

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [References](#references)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

## About The Project

This repo contains the code to predict scheduling of strawberry picking starting from images of clusters of strawberries with bounding box and occlusion annotations.

There are some *side projects*:

1) Success of picking
2) DETR for occlusion properties detection
3) Detectron2 for ripeness detection

More informations are in the READMEs of the respective folders or in the Usage section of this README.
## Built With

- **Operating system Ubuntu 10.04**
- **Language Python 3.9**
- **Main library Torch 1.12** 

## Getting Started

To predict a scheduling order, each strawberry learns its representation in the contest of the cluster. To do so, we use a Graph Neural Neural Network (GNN). We represent the annotation of the images as graphs, where the nodes are the strawberries, represented by the bounding box coordinates and an occlusion and ripeness score.

Three main trials were followed:

1) classify each strawberry as "first to be picked" or not. The output of the model is a probability for each strawberry. The last layer of the model is a Sigmoid activation function and the loss is Binary Cross Entropy loss. Since the dataset is imbalanced (there are more strawberries that are not first to be picked), the loss can be weighted

2) predict an "easiness of picking" score, for each strawberry. The last layer of the model is a custom LeakyReLU, with variable negative and positive slopes. The loss function is Mean Squared Error.

4) Predict a probability distribution for each cluster (vector of probabilities of being picked first for each strawberry, that sums to one). The last to one layer is a Log-Softmax. The loss is KL Divergence.

For all the approaches, we can retrive a scheduling order by sorting the outputs.


### Prerequisites

All the Pytorch libraries needed are in the 'requirements.txt' file

### Installation

Just clone this repo, and download the dataset if you want to train or test with our data. The dataset can be retrieved from the repo: https://github.com/imanlab/pickingScheduling_successPredict

## Usage

### Get bbox and occlusion properties

Download DETR checkpoint from: https://drive.google.com/file/d/1fmmrY3Z-DwKdr1_M8XX5tUic_yaqKK5E/view?usp=sharing

Then follow the instructions of the README inside detr folder

### Get bbox and ripeness property

Download Detectron2 checkpoint from: https://drive.google.com/file/d/1j62QIjH1Uq3YPmM59WrT9VnbOzYVb-EN/view?usp=sharing

Go to detectron2/inference_dyson_keypoints.py and add the path to the images you want to process in the variable 'img_dir'.

Then cd to the detectron2 folder and run:

python inference_dyson_keypoints.py

If you have troubles installing detectron2 package, check: https://detectron2.readthedocs.io/en/latest/tutorials/install.html

### EXPERIMENT: from raw image to image with white patch on target

Go to experiments.py and put the path to your image in the variable 'image_path'. You also need to have the detr model checkpoint downloaded. Then:

python experiments_pickall.py

if you want to use the model that outputs all the sequence of picking scheduling, or:

python experiments_pick1.py

if you want to use the model that works better at predicting only the first strawberry to be picked. You will have all the scheduling order, since every strawberry picked then is masked so the new image can be processed by the model.

### Test our models

Choose a model from the 'best_models' folder, load it in the 'test.py' script. More instructions are in the comments to the code.

To run from terminal, cd to the workspace folder and type: 

python test.py

### Train a model

To obtain annotations in COCO format from your own annotations, use data_scripts/ann_to_coco.py

To obtain graph representation of images, use data_scripts/coco_to_gnnann.py

Choose an approach, from the 'config.py' script. Remeber to delete the 'process' folders in the data_train/test/val folders when you are switching approach.

To run from terminal, cd to the workspace folder and type: 

python train.py

## Roadmap

We have a good prediction of the first strawberry to be picked, but not of the whole scheduling.

Future steps could be:

- trying to customize a loss function for the "scores" approach: in fact, the model tends just to output small numbers so that the loss can be small
- use the map attention output of the DETR as node feature

## References

Usefull papers I came across while studying GNNs:

[1] Y. Huang, A. Conkey, and T. Hermans, “Planning for multi-object manipulation with graph neural network relational classifiers,” arXiv preprint arXiv:2209.11943, 2022.

[2] P. Veliˇckovi ́c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, “Graph attention networks,” arXiv preprint arXiv:1710.10903, 2017.

[3] K. Han, Y. Wang, J. Guo, Y. Tang, and E. Wu, “Vision gnn: An image is worth graph of nodes,” arXiv preprint arXiv:2206.00272, 2022.

[4] D. Xu, Y. Zhu, C. B. Choy, and L. Fei-Fei, “Scene graph generation by iterative message passing,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5410–5419.

[5] H. Xu, C. Jiang, X. Liang, and Z. Li, “Spatial-aware graph relation network for large-scale object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 92989307.

[6] S. Thakur, B. Pandey, J. Peethambaran, and D. Chen, “A graph attention network for object detection from raw lidar data,” in IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2022, pp. 3091–3094.

[7] F. Ma, F. Gao, J. Sun, H. Zhou, and A. Hussain, “Attention graph convolution network for image segmentation in big sar imagery data,”Remote Sensing, vol. 11, no. 21, p. 2586, 2019.

[8] A. S. Nassar, S. D’aronco, S. Lef`evre, and J. D. Wegner, “Geograph: Graph-based multi-view object detection with geometric cues end-to-end,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VII 16. Springer, 2020, pp. 488–504.

[9] Z. Zhao, G. Verma, C. Rao, A. Swami, and S. Segarra, “Distributed scheduling using graph neural networks,” in ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021, pp. 4720–4724.

[10] R. Sambamoorthy, J. Mandalapu, S. S. Peruru, B. Jain, and E. Altman, “Graph neural network based scheduling: Improved throughput under a generalized interference model,” in Performance Evaluation Methodologies and Tools: 14th EAI International Conference, VALUETOOLS 2021, Virtual Event, October 30–31, 2021, Proceedings. Springer,2021, pp. 144–153.

## Contact 

For any issue please contact me at bigichiara1@gmail.com :)


## Acknowledgements

A. Tafuro, A. Adewumi, S. Parsa, G. E. Amir, and B. Debnath, “Strawberry picking point localization ripeness and weight estimation,” in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 2295–2302.
