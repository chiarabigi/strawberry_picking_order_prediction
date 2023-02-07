from dataset import PickingSuccessDataset
from model import GCN_success
from utils import BatchAccuracy_success

LR = 1e-3
HL = 64
BATCHSIZE = 32
WEIGHTDECAY = 0.01
SEEDNUM = 0
TORCHSEED = 0

ACCURACY = BatchAccuracy_success

DATASET = PickingSuccessDataset

MODEL = GCN_success(HL)

