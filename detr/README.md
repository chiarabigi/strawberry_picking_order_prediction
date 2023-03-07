# DETR fine tuning

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

This repo contains the code that we use to train a model that outputs bounding boxes and occlusion properties of berries for images of clusters of strawberries.

We fine-tuned a DETR-DC5 benchmark model End-to-end Object Detection with Transformer was shown to significantly outperform competitive baselines, viewing object detection as a direct set prediction problem.


### Built With

- **Operating system Ubuntu 10.04**
- **Language Pytorch 3.9**
- **Main library Torch 1.12** 

## Getting Started

The dataset can be retrieved from the repo: https://github.com/imanlab/pickingScheduling_successPredict

### Prerequisites

All the requirements are in the 'requirements.txt' file

### Installation

You can use this folder.

Alternatively, clone DETR official repo: https://github.com/facebookresearch/detr.

The only files you need to change are test.py and main.py, that you can find in this repo. Also the number of classes should be modified to 5.

## Usage

### Train

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py

### Test

python test.py

For more information check DETR official README from the link above.

## Roadmap

As it can be seen in the log.txt file, we didn't obtain good results.

## References 

DE⫶TR: End-to-End Object Detection with Transformers https://github.com/facebookresearch/detr

## Contact 

For any issue please contact me at bigichiara1@gmail.com :)

## Acknowledgements

DE⫶TR: End-to-End Object Detection with Transformers https://github.com/facebookresearch/detr
