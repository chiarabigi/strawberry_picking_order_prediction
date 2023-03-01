from dataset import PickingSuccessDataset
from model import GCN_success
from utils.metrics import BatchAccuracy_success

LR = 1e-3
HL = 64
NL = 0
BATCHSIZE = 32
WEIGHTDECAY = 0.01
SEEDNUM = 0
TORCHSEED = 0

ACCURACY = BatchAccuracy_success

DATASET = PickingSuccessDataset

MODEL = GCN_success(HL)

