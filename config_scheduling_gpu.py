from dataset import SchedulingDataset
from model_gpu import GCN_scheduling
from utils import BatchAccuracy_scheduling

LR = 1e-3
HL = 128
BATCHSIZE = 32
NL = 5
WEIGHTDECAY = 0.01
SEEDNUM = 0
TORCHSEED = 0

ACCURACY = BatchAccuracy_scheduling

DATASET = SchedulingDataset

MODEL = GCN_scheduling(HL, NL)


