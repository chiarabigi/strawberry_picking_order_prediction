import torch
import json
from PIL import Image

data_test = '/home/chiara/SEGMENTATION/DATASETS/DATASET_ASSIGNMENT1/coco/annotations/instances_test.json'
images_path = '/home/chiara/SEGMENTATION/DATASETS/DATASET_ASSIGNMENT1/coco/test/'
model = torch.load('/home/chiara/SCHEDULING/GNN/detr/checkpoints/checkpoint.pth', map_location=torch.device('cpu') )

with open(data_test) as f:
    anns = json.load(f)


for a in range(len(anns)):
    filename = anns['images'][a]['file_name']
    d = Image.open(images_path + filename)
    model.eval()
    output = model(d)
