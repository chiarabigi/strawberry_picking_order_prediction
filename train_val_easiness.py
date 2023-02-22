import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import config_scheduling
import config_scheduling_gpu
from utils import get_occlusion1, get_realscheduling, get_whole_scheduling, get_label_scheduling
import multiprocessing
from collections import Counter

with torch.no_grad():
   torch.cuda.empty_cache()


def train_one_epoch():
    model.train()
    running_loss = 0.0
    real_scheduling = np.zeros(18)
    tot_nodes = 0.0
    step = 0
    occ_1 = np.zeros(5)
    matches = []
    storeP = []
    storeT = []
    for i, batch in enumerate(train_loader, 0):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # batch.to(device)
        outputs = model(batch)
        weights = torch.ones_like(batch.y) * 84.6 + (1.0 - 1.0 * 84.6) * batch.y
        # loss = torch.nn.functional.binary_cross_entropy(outputs, batch.y, weights)
        loss = criterion(outputs, batch.y)
        # backward
        loss.backward()

        # optimize
        optimizer.step()

        # statistics
        # print(f'\nTrain Loss: {loss.item():.4f}, \t at iteration: {int(i):.4f}')
        running_loss += loss.item()

        if loss.item() > 1:
            print(0)

        # real_scheduling += get_realscheduling(outputs, batch.label, batch.batch)
        # occ_1 += get_occlusion1(outputs, batch.info, batch.batch)
        tot_nodes += len(batch.batch)
        step += 1

        storeT.extend(batch.y.t().tolist()[0])
        storeP.extend(outputs.t().tolist()[0])
        outputsL = [round(x, 2) for x in outputs.t().tolist()[0]]
        batchyL = [round(x, 2) for x in batch.y.t().tolist()[0]]
        matches += [outputsL[x] for x in range(len(outputsL)) if outputsL[x] == batchyL[x]]

    print('\n Matches TRAIN', sorted(Counter(matches).most_common()))
    print(f'\n% guessed TRAIN: {100 * sum(Counter(matches).values()) / tot_nodes:.4f}, \twith: {len(Counter(matches).most_common()):.4f} different scores')
    print(f'\ntot pred < 0 TRAIN: {len([x for x in Counter(matches).keys() if x < 0]):.4f}, \t == 0: {len([x for x in Counter(matches).keys() if x == 0]):.4f}')
    # for loss plot
    y_loss['train'].append(running_loss / step)
    wP = Counter(storeP)
    wT = Counter(storeT)
    plt.bar(wT.keys(), wT.values(), width=0.001)
    plt.bar(wP.keys(), wP.values(), width=0.001)

    plt.title('Strawberry train easiness score. Blue: gt. Orange: predicted')
    plt.savefig('imgs/truetrainEasiness.png')
    #print('True scheduling of predicted as first (TRAIN): ', real_scheduling)
    #print('Occlusion property for node with higher probability (TRAIN): ', occ_1)
    return running_loss / step


def validation():
    model.eval()
    running_vloss = 0.0
    real_vscheduling = np.zeros(18)
    tot_vnodes = 0.0
    step = 0
    occ_1 = np.zeros(5)
    matches = []
    storeT = []
    storeP = []

    for i, vbatch in enumerate(val_loader):
        # vbatch.to(device)
        voutputs = model(vbatch)
        weights = torch.ones_like(vbatch.y) / 0.3 + (1.0 - 1.0 / 0.3) * vbatch.y
        # vloss = torch.nn.functional.binary_cross_entropy(voutputs, vbatch.y, weights)
        vloss = criterion(voutputs, vbatch.y)
        running_vloss += vloss.item()
        # real_vscheduling += get_realscheduling(voutputs, vbatch.label, vbatch.batch)
        # occ_1 += get_occlusion1(voutputs, vbatch.info, vbatch.batch)
        tot_vnodes += len(vbatch.batch)
        step += 1
        storeT.extend(vbatch.y.t().tolist()[0])
        storeP.extend(voutputs.t().tolist()[0])
        outputsL = [round(x, 2) for x in voutputs.t().tolist()[0]]
        batchyL = [round(x, 2) for x in vbatch.y.t().tolist()[0]]
        matches += [outputsL[x] for x in range(len(outputsL)) if outputsL[x] == batchyL[x]]

    print('Matches VAL', sorted(Counter(matches).most_common()))
    print(f'\n% guessed VAL: {100 * sum(Counter(matches).values()) / tot_vnodes:.4f}, \twith: {len(Counter(matches).most_common()):.4f} different scores')
    print(f'\ntot pred < 0 VAL: {len([x for x in Counter(matches).keys() if x < 0]):.4f}, \t == 0: {len([x for x in Counter(matches).keys() if x == 0]):.4f}')
    avg_vloss = running_vloss / step

    y_loss['val'].append(avg_vloss)
    wP = Counter(storeP)
    wT = Counter(storeT)
    plt.bar(wT.keys(), wT.values(), width=0.001)
    plt.bar(wP.keys(), wP.values(), width=0.001)

    plt.title('Strawberry val easiness score. Blue: gt. Orange: predicted')
    plt.savefig('imgs/truevalEasiness.png')
    #print('True scheduling of predicted as first (VAL): ', real_vscheduling)
    #print('Occlusion property for node with higher probability (VAL): ', occ_1)
    return avg_vloss


def test():
    model.eval()
    real_tscheduling = np.zeros(18)
    occ_1 = np.zeros(5)
    sched_pred = np.zeros(18)
    sched_true = np.zeros(18)
    sched_students = np.zeros(18)
    sched_heuristic = np.zeros(18)
    storeT = []
    storeP = []
    matches = []
    tot_tnodes = 0.0

    for i, tbatch in enumerate(test_loader):
        # tbatch.to(device)
        pred = model(tbatch)
        #real_tscheduling += get_realscheduling(pred, tbatch.label, tbatch.batch)
        #occ_1 += get_occlusion1(pred, tbatch.info, tbatch.batch)
        #s_pred, s_true = get_whole_scheduling(pred, tbatch.label, tbatch.batch)
        #sched_students += get_label_scheduling(tbatch.students_ann, tbatch.batch)
        #sched_heuristic += get_label_scheduling(tbatch.heuristic_ann, tbatch.batch)
        #sched_pred += s_pred
        #sched_true += s_true
        outputsL = [round(x, 2) for x in pred.t().tolist()[0]]
        batchyL = [round(x, 2) for x in tbatch.y.t().tolist()[0]]
        storeT.extend(tbatch.y.t().tolist()[0])
        storeP.extend(pred.t().tolist()[0])
        matches += [outputsL[x] for x in range(len(outputsL)) if outputsL[x] == batchyL[x]]
        tot_tnodes += len(tbatch.batch)

    print('Matches TEST', sorted(Counter(matches).most_common()))
    print(f'\n% guessed TEST: {100 * sum(Counter(matches).values()) / tot_tnodes:.4f}, \twith: {len(Counter(matches).most_common()):.4f} different scores')
    print(f'\ntot pred < 0 TEST: {len([x for x in Counter(matches).keys() if x < 0]):.4f}, \t == 0: {len([x for x in Counter(matches).keys() if x == 0]):.4f}')
    wP = Counter(storeP)
    wT = Counter(storeT)

    plt.figure(2)
    plt.bar(wT.keys(), wT.values(), width=0.001)
    plt.bar(wP.keys(), wP.values(), width=0.001)

    plt.title('Strawberry test easiness score. Blue: gt. Orange: predicted')
    plt.savefig('imgs/truetestEasiness.png')

    '''
    for i in range(len(real_tscheduling) - 1):
        print(
            f'\n {100 * real_tscheduling[i] / real_tscheduling.sum():.4f}% of strawberries predicted as EASIEST TO PICK in the cluster where scheduled as {i + 1:.4f}')
    print(
        f'\n {100 * real_tscheduling[-1] / real_tscheduling.sum():.4f}% of strawberries predicted as EASIEST TO PICK in the cluster where UNRIPE')
    occlusion_properties = ['NON OCCLUDED', 'OCCLUDED BY LEAF', 'OCCLUDED BY A BERRY']
    occ = np.zeros(3)
    occ[0] += occ_1[1] + occ_1[3]
    occ[1] += occ_1[0] + occ_1[2]
    occ[2] += occ_1[4]
    for i in range(len(occ)):
        print(
            f'\n {100 * occ[i] / occ.sum():.4f}% of strawberries predicted as EASIEST TO PICK in the cluster where labeled as {occlusion_properties[i]}')

    sched_pred += 2
    sched_students += 2
    sched_true += 2
    sched_heuristic += 2
    print('\nPREDICTION VS STUDENTS')
    for i in range(len(sched_pred) - 1):
        print(
            f'\n {100 * (abs(sched_pred[i] - abs(sched_pred[i] - sched_students[i]))) / sched_pred[i]:.4f}% of strawberries predicted as {int(i + 1):.4f} to be picked is the same as students annotation')
    print(
        f'\n {100 * (abs(sched_pred[-1] - abs(sched_pred[-1] - sched_students[-1]))) / sched_pred[-1]:.4f}% of strawberries predicted as last to be picked (because unripe) is the same as students annotation')
    print('\nHEURISTIC')
    for i in range(len(sched_pred) - 1):
        print(
            f'\n {100 * (abs(sched_pred[i] - abs(sched_pred[i] - sched_heuristic[i]))) / sched_pred[i]:.4f}% of strawberries predicted as {int(i + 1):.4f} to be picked is the same as heuristic')
    print(
        f'\n {100 * (abs(sched_pred[-1] - abs(sched_pred[-1] - sched_heuristic[-1]))) / sched_pred[-1]:.4f}% of strawberries predicted as last to be picked (because unripe) is the same as heuristic')
    print('\nPREDICTION VS GROUND TRUTH  (=scheduling from easiness: first is easiest, last is least easy...)')
    for i in range(len(sched_pred) - 1):
        print(
            f'\n {100 * (abs(sched_pred[i] - abs(sched_pred[i] - sched_true[i]))) / sched_pred[i]:.4f}% of strawberries predicted as {int(i + 1):.4f} to be picked is the same as ground truth')
    print(
        f'\n {100 * (abs(sched_pred[-1] - abs(sched_pred[-1] - sched_true[-1]))) / sched_pred[-1]:.4f}% of strawberries predicted as last to be picked (because unripe) is the same as ground truth')'''

def draw_curve(current_epoch, cfg, lastEpoch, best_loss):
    if current_epoch != lastEpoch:
        x_epoch.append(current_epoch)
        ax.plot(x_epoch, y_loss['train'], 'bo-', label='train', linewidth=1)
        ax.plot(x_epoch, y_loss['val'], 'ro-', label='val', linewidth=1)
    if current_epoch == 0:
        ax.legend()
    elif current_epoch == lastEpoch:
        ax.text(0.5, 0.5, 'T' + str(best_loss),
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    fig.savefig(os.path.join('./plots/'.format(goal), 'train_{}_{}_{}_L2{}_{}.jpg'.format(cfg.HL, cfg.NL, cfg.BATCHSIZE, cfg.WEIGHTDECAY, cfg.SEEDNUM)))


# Main
def train():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    best_vloss = 1_000_000.
    best_loss = 1_000_000.
    early_stopping_counter = 0
    NumEpochs = 300
    lastEpoch = NumEpochs + 1
    lastEpochlist = []
    for epoch in trange(1, NumEpochs + 1):
        if early_stopping_counter <= 10:
            # Training
            plt.figure(3)
            train_loss = train_one_epoch()
            plt.close(3)
            # Validation
            plt.figure(4)
            val_loss = validation()
            plt.close(4)

            print(f'\nTrain Loss: {train_loss:.4f}, \tValidation Loss: {val_loss:.4f}')  # , \tTest Loss: {test_loss:.4f}

            scheduler.step(val_loss)

            # draw curve
            draw_curve(epoch, cfg, lastEpoch, best_loss)

            # Log the running loss averaged per batch for training, and validation
            writer.add_scalars('Training vs. Validation Loss', {'Training': train_loss, 'Validation': val_loss}, epoch + 1)
            writer.flush()

            # Track the best performance, and save the model's state
            if val_loss < best_vloss:
                best_vloss = val_loss
                model_path = '/home/chiara/strawberry_picking_order_prediction/best_models/model_{}.pth'.format(timestamp)
                torch.save(model.state_dict(), model_path)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if train_loss < best_loss:
                best_loss = train_loss

        else:
            lastEpochlist.append(epoch)
            continue

    if early_stopping_counter >= 10:
        print("Early stopping due to no improvement.")

    # Test
    model.load_state_dict(torch.load(model_path))
    test()

    # to print loss and accuracy best values
    if len(lastEpochlist) > 0:
        lastEpoch = int(lastEpochlist[0])
    draw_curve(lastEpoch, cfg, lastEpoch, best_loss)


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('main', device)

    if device.type == 'cpu':
        cfg = config_scheduling
    else:
        cfg = config_scheduling  # _gpu

    goal = 'easiness'
    #multiprocessing.set_start_method('spawn')

    # Tuned Parameters

    learningRate = cfg.LR
    hiddenLayers = cfg.HL
    numlayers = cfg.NL
    batchSize = cfg.BATCHSIZE
    weightDecay = cfg.WEIGHTDECAY

    # Set Seed

    SeedNum = cfg.SEEDNUM
    TorchSeed = cfg.TORCHSEED

    np.random.seed(SeedNum)
    torch.manual_seed(TorchSeed)
    torch.cuda.manual_seed(TorchSeed)

    # Load Dataset

    print("Loading data_scripts...")
    train_path = 'dataset/data_train/'.format(goal)
    train_dataset = cfg.DATASET(train_path)
    val_path = 'dataset/data_val/'.format(goal)
    val_dataset = cfg.DATASET(val_path)
    test_path = 'dataset/data_test/'.format(goal)
    test_dataset = cfg.DATASET(test_path)

    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)  #, num_workers=2, pin_memory_device='cuda:1', pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)  #, num_workers=2, pin_memory_device='cuda:1', pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)  #, num_workers=2, pin_memory_device='cuda:1', pin_memory=True)
    print("Done!")

    # Initialize model, optimizer and loss

    model = cfg.MODEL.to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
    scheduler = ReduceLROnPlateau(optimizer)
    criterion = torch.nn.MSELoss()  # torch.nn.BCELoss()  # weight it!! classes are imbalanced
    batchAccuracy = cfg.ACCURACY()

    # Parameters for plots

    y_loss = {}
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []
    x_epoch = []
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)

    train()

# Clear cuda cache
with torch.no_grad():
    torch.cuda.empty_cache()
