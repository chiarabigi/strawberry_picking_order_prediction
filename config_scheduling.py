from dataset import SchedulingDataset
from model import GCN_scheduling
from utils import BatchAccuracy_scheduling

LR = 1e-3
HL = 8
BATCHSIZE = 32
WEIGHTDECAY = 0.01
SEEDNUM = 24
TORCHSEED = 24

ACCURACY = BatchAccuracy_scheduling

DATASET = SchedulingDataset

MODEL = GCN_scheduling(HL)
