import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import trange
from datetime import datetime
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import config as cfg
from utils.metrics import get_comparison, plot_heatmap
from collections import Counter


with torch.no_grad():
    torch.cuda.empty_cache()


def train_one_epoch():
    model.train()
    running_loss = 0.0
    tot_nodes = 0.0
    step = 0

    if criterion._get_name() == 'BCELoss':
        sched_easiness = np.zeros((32, 32))
        sched_students = np.zeros((32, 32))
        sched_heuristic = np.zeros((32, 32))
        occ_1 = np.zeros((3, 32))
    else:
        matches = []
        storeP = []
        storeT = []

    for i, batch in enumerate(train_loader, 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(batch)
        loss = criterion(outputs, batch.y)
        # backward
        loss.backward()

        # optimize
        optimizer.step()

        # statistics
        # print(f'\nTrain Loss: {loss.item():.4f}, \t at iteration: {int(i):.4f}')
        running_loss += loss.item()
        tot_nodes += len(batch.batch)
        step += 1
        ''''''
        if criterion._get_name() == 'BCELoss':
            sched_easiness, sched_students, sched_heuristic, occ_1 = get_comparison(outputs, batch, sched_easiness,
                                                                            sched_students, sched_heuristic, occ_1)
        else:
            storeT += [x for x in batch.y.t().tolist()[0] if x != 0]
            storeP += [x for x in torch.exp(outputs).t().tolist()[0] if x != 0]
            outputsL = [round(x, 2) for x in outputs.t().tolist()[0]]
            batchyL = [round(x, 2) for x in batch.y.t().tolist()[0]]
            matches += [outputsL[x] for x in range(len(outputsL)) if outputsL[x] == batchyL[x]]

    # for loss plot
    y_loss['train'].append(running_loss / step)

    if criterion._get_name() == 'BCELoss':
        plot_heatmap(sched_easiness, list(range(0, 31)), 'TRAIN heuristic easiness score scheduling')
        plot_heatmap(sched_heuristic, list(range(0, 31)), 'TRAIN heuristic min max scheduling')
        plot_heatmap(sched_students, list(range(0, 31)), 'TRAIN students scheduling')
        plot_heatmap(occ_1, ['NON', ' BY LEAF', 'BY BERRY'], 'TRAIN occlusion property')
    else:
        print('\n Matches TRAIN', sorted(Counter(matches).most_common()))
        print(f'\n% guessed TRAIN: {100 * sum(Counter(matches).values()) / tot_nodes:.4f}, \twith: {len(Counter(matches).most_common()):.4f} different scores')
        print(f'\ntot pred < 0 TRAIN: {len([x for x in Counter(matches).keys() if x < 0]):.4f}, \t == 0: {len([x for x in Counter(matches).keys() if x == 0]):.4f}')
        wP = Counter(storeP)
        wT = Counter(storeT)
        plt.figure(3)
        plt.bar(wP.keys(), wP.values(), width=0.01)
        plt.bar(wT.keys(), wT.values(), width=0.001)

        plt.title('Strawberry train easiness score. Orange: gt. Blue: predicted')
        plt.savefig('imgs/truetrainEasiness.png')
        plt.close(3)

    return running_loss / step


def validation():
    model.eval()
    running_vloss = 0.0
    tot_vnodes = 0.0
    step = 0

    if criterion._get_name() == 'BCELoss':
        sched_easiness = np.zeros((32, 32))
        sched_students = np.zeros((32, 32))
        sched_heuristic = np.zeros((32, 32))
        occ_1 = np.zeros((3, 32))
    else:
        matches = []
        storeT = []
        storeP = []

    for i, vbatch in enumerate(val_loader):

        voutputs = model(vbatch)
        vloss = criterion(voutputs, vbatch.y)
        running_vloss += vloss.item()
        tot_vnodes += len(vbatch.batch)
        step += 1

        if criterion._get_name() == 'BCELoss':
            sched_easiness, sched_students, sched_heuristic, occ_1 = get_comparison(voutputs, vbatch, sched_easiness,
                                                                                    sched_students, sched_heuristic,
                                                                                    occ_1)
        else:
            storeT += [x for x in vbatch.y.t().tolist()[0] if x != 0]
            storeP += [x for x in torch.exp(voutputs).t().tolist()[0] if x != 0]
            outputsL = [round(x, 2) for x in voutputs.t().tolist()[0]]
            batchyL = [round(x, 2) for x in vbatch.y.t().tolist()[0]]
            matches += [outputsL[x] for x in range(len(outputsL)) if outputsL[x] == batchyL[x]]

    avg_vloss = running_vloss / step
    y_loss['val'].append(avg_vloss)

    if criterion._get_name() == 'BCELoss':
        plot_heatmap(sched_easiness, list(range(0, 31)), 'VAL heuristic easiness score scheduling')
        plot_heatmap(sched_heuristic, list(range(0, 31)), 'VAL heuristic min max scheduling')
        plot_heatmap(sched_students, list(range(0, 31)), 'VAL students scheduling')
        plot_heatmap(occ_1, ['NON', ' BY LEAF', 'BY BERRY'], 'VAL occlusion property')
    else:
        print('Matches VAL', sorted(Counter(matches).most_common()))
        print(
            f'\n% guessed VAL: {100 * sum(Counter(matches).values()) / tot_vnodes:.4f}, \twith: {len(Counter(matches).most_common()):.4f} different scores')
        print(
            f'\ntot pred < 0 VAL: {len([x for x in Counter(matches).keys() if x < 0]):.4f}, \t == 0: {len([x for x in Counter(matches).keys() if x == 0]):.4f}')

        wP = Counter(storeP)
        wT = Counter(storeT)
        plt.figure(4)
        plt.bar(wP.keys(), wP.values(), width=0.01)
        plt.bar(wT.keys(), wT.values(), width=0.001)
        plt.title('Strawberry val easiness score. Orange: gt. Blue: predicted')
        plt.savefig('imgs/truevalEasiness.png')
        plt.close(4)

    return avg_vloss


def test():
    model.eval()
    tot_tnodes = 0.0

    if criterion._get_name() == 'BCELoss':
        sched_easiness = np.zeros((32, 32))
        sched_students = np.zeros((32, 32))
        sched_heuristic = np.zeros((32, 32))
        occ_1 = np.zeros((3, 32))
    else:
        storeT = []
        storeP = []
        matches = []

    for i, tbatch in enumerate(test_loader):
        pred = model(tbatch)
        tot_tnodes += len(tbatch.batch)

        if criterion._get_name() == 'BCELoss':
            sched_easiness, sched_students, sched_heuristic, occ_1 = get_comparison(pred, tbatch, sched_easiness,
                                                                                    sched_students, sched_heuristic,
                                                                                    occ_1)
        else:
            outputsL = [round(x, 2) for x in pred.t().tolist()[0]]
            batchyL = [round(x, 2) for x in tbatch.y.t().tolist()[0]]
            storeT += [x for x in tbatch.y.t().tolist()[0] if x != 0]
            storeP += [x for x in torch.exp(pred).t().tolist()[0] if x != 0]
            matches += [outputsL[x] for x in range(len(outputsL)) if outputsL[x] == batchyL[x]]

    if criterion._get_name() == 'BCELoss':
        plot_heatmap(sched_easiness, list(range(0, 31)), 'TEST heuristic easiness score scheduling')
        plot_heatmap(sched_heuristic, list(range(0, 31)), 'TEST heuristic min max scheduling')
        plot_heatmap(sched_students, list(range(0, 31)), 'TEST students scheduling')
        plot_heatmap(occ_1, ['NON', ' BY LEAF', 'BY BERRY'], 'TEST occlusion property')
    else:
        print('Matches TEST', sorted(Counter(matches).most_common()))
        print(
            f'\n% guessed TEST: {100 * sum(Counter(matches).values()) / tot_tnodes:.4f}, \twith: {len(Counter(matches).most_common()):.4f} different scores')
        print(
            f'\ntot pred < 0 TEST: {len([x for x in Counter(matches).keys() if x < 0]):.4f}, \t == 0: {len([x for x in Counter(matches).keys() if x == 0]):.4f}')
        wP = Counter(storeP)
        wT = Counter(storeT)

        plt.figure(2)
        plt.bar(wP.keys(), wP.values(), width=0.01)
        plt.bar(wT.keys(), wT.values(), width=0.001)

        plt.title('Strawberry test easiness score. Orange: gt. Blue: predicted')
        plt.savefig('imgs/truetestEasiness.png')
        plt.close(2)

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
    fig.savefig(os.path.join('./plots/', 'train_{}_{}_{}_L2{}_{}.jpg'.format(cfg.HL, cfg.NL, cfg.BATCHSIZE, cfg.WEIGHTDECAY, cfg.SEEDNUM)))


# Main
def train():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = 1_000_000.
    best_loss = 1_000_000.
    early_stopping_counter = 0
    NumEpochs = 300
    lastEpoch = NumEpochs + 1
    lastEpochlist = []
    for epoch in trange(1, NumEpochs + 1):
        if early_stopping_counter <= 10:
            # Training
            train_loss = train_one_epoch()
            # Validation
            val_loss = validation()

            print(f'\nTrain Loss: {train_loss:.4f}, \tValidation Loss: {val_loss:.4f}')

            scheduler.step(val_loss)

            # draw curve
            draw_curve(epoch, cfg, lastEpoch, best_loss)

            # Track the best performance, and save the model's state
            if val_loss < best_vloss:
                best_vloss = val_loss
                model_path = base_path + '/best_models/model_{}.pth'.format(timestamp)
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

    # to print loss and accuracy best values on the plots
    if len(lastEpochlist) > 0:
        lastEpoch = int(lastEpochlist[0])
    draw_curve(lastEpoch, cfg, lastEpoch, best_loss)


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('main', device)
    base_path = os.path.dirname(os.path.abspath(__file__))

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
    train_path = 'dataset/data_train/'
    train_dataset = cfg.DATASET(train_path)
    val_path = 'dataset/data_val/'
    val_dataset = cfg.DATASET(val_path)
    test_path = 'dataset/data_test/'
    test_dataset = cfg.DATASET(test_path)

    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)  #, num_workers=2, pin_memory_device='cuda:1', pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)  #, num_workers=2, pin_memory_device='cuda:1', pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)  #, num_workers=2, pin_memory_device='cuda:1', pin_memory=True)
    print("Done!")

    # Initialize model, optimizer and loss
    model = cfg.MODEL(hiddenLayers, numlayers).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
    scheduler = ReduceLROnPlateau(optimizer)
    criterion = cfg.LOSS

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

    # call main function
    train()

# Clear cuda cache
with torch.no_grad():
    torch.cuda.empty_cache()
