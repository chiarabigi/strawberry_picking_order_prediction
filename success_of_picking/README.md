# Success of pickig


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

This repo contains the code to predict if a target strawberry in a cluster will be picked by our robot. We start from a dataset of images in which just one strawberry is annotated, with binary information of whether it was successfully picked or not.


We add information of bounding boxes and occlusion properties to the other strawberries in the cluster, too (with our fine tuned DETR model), and convert this information in a form of a graph.


The nodes of the graph are the strawberries, represented by bounding box coordinates, occlusion property, and a binary information of wether the strawberry is the target or not.


Then we use a GNN model.



### Built With



- **Operating system Ubuntu 10.04**

- **Language Python 3.9**

- **Main library Torch 1.12** 




## Getting Started



The dataset can be retrieved from the repo: https://github.com/imanlab/pickingScheduling_successPredict



### Prerequisites



See the main README of this repo


### Installation



Just clone the main repo, and download the dataset in another folder.



## Usage



### Train

Run the train_val.py script

### Test

Run the test.py script

### Compare different schedulings

Instruction are in the comments to the code of the script different_scheduling_choices.py

### From raw image to target strawberry

Instruction are in the comments to the code of the script experiments.py


## Roadmap



I have a milion doubts about the consistency of this datastet. But the model did learn something.



We imporved the graph representation for the scheduling model, we could import the same changes here.



## References 



See the references of the main branch.



## Contact 


For any issue please contact me at bigichiara1@gmail.com :)





## Acknowledgements



Soran's Robofruit





















