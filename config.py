import torch
from model import GAT_classes, GAT_scores, GAT_prob
from data_scripts.dataset01class import Scheduling01ClassDataset
from data_scripts.dataset01scores import Scheduling01ScoreDataset
from data_scripts.dataset01probabilities import Scheduling01ProbDataset
from utils.custom import CustomMSE

LR = 1e-3
HL = 8
BATCHSIZE = 32
NL = 0
WEIGHTDECAY = 0.01
SEEDNUM = 0
TORCHSEED = 0

approach = 'class'

if approach == 'class':
    MODEL = GAT_classes
    DATASET = Scheduling01ClassDataset
    LOSS = torch.nn.BCELoss()
elif approach == 'score':
    MODEL = GAT_scores
    DATASET = Scheduling01ScoreDataset
    LOSS = torch.nn.MSELoss()
elif approach == 'probability':
    MODEL = GAT_prob
    DATASET = Scheduling01ProbDataset
    LOSS = torch.nn.KLDivLoss()