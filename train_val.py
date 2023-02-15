import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import trange
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import config_scheduling
import config_scheduling_gpu
from utils import get_occlusion1
import multiprocessing

with torch.no_grad():
   torch.cuda.empty_cache()


def train_one_epoch(BestModel, Phase):
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    real_scheduling = np.zeros(18)
    prob1 = 0
    prob2 = 0
    tot_nodes = 0.0
    step = 0

    all_y_pred = []
    all_y_true = []
    occ_1 = np.zeros(4)
    occ_class = np.zeros(4)
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
        print(f'\nTrain Loss: {loss.item():.4f}, \t at iteration: {int(i):.4f}')
        running_loss += loss.item()
        running_corrects += batchAccuracy(outputs, batch.y, batch.batch)
        for j in range(len(batch.y)):
            if outputs[j] > 0.5:
                real_scheduling[batch.label[j] - 1] += 1
                if batch.label[j] == 1:
                    prob1 += outputs[j]
                elif batch.label[j] == 2:
                    prob2 += outputs[j]
                occ_class[batch.info[j]] += 1
        occ_1 += get_occlusion1(outputs, batch.info, batch.batch)
        tot_nodes += len(batch.batch)
        step += 1

        # for additional metrics
        for j in range(len(outputs)):
            if outputs[j] > 0.5:
                outputs[j] = 1
            else:
                outputs[j] = 0
        y = torch.zeros_like(batch.y)
        for j in range(len(batch.y)):
            if batch.y[j] > 0.5:
                y[j] = 1
        all_y_pred.append(outputs.cpu().detach().numpy())
        all_y_true.append(y.cpu().detach().numpy())

    all_preds = np.concatenate(all_y_pred)
    all_labels = np.concatenate(all_y_true)
    calculate_metrics(all_preds, all_labels, BestModel, Phase)

    running_loss = running_loss / step
    running_corrects = running_corrects / tot_nodes
    # for loss and accuracy plot
    y_loss['train'].append(running_loss)
    y_err['train'].append(running_corrects)
    print('True scheduling of predicted as first (TRAIN): ', real_scheduling)
    print(f'\nAverage probability (TRAIN) of first being first: {float(prob1 / real_scheduling[0]):.4f},'
          f'\t or second: {float(prob2 / real_scheduling[1]):.4f}')
    print('Occlusion property for classified as first (TRAIN): ', occ_class)
    print('Occlusion property for node with higher probability (TRAIN): ', occ_1)
    return running_loss, running_corrects


def validation():
    model.eval()
    running_vloss = 0.0
    running_vcorrects = 0.0
    real_vscheduling = np.zeros(18)
    vprob1 = 0
    vprob2 = 0
    tot_vnodes = 0.0
    step = 0
    occ_1 = np.zeros(4)
    occ_class = np.zeros(4)

    for i, vbatch in enumerate(val_loader):
        # vbatch.to(device)
        voutputs = model(vbatch)
        weights = torch.ones_like(vbatch.y) / 0.3 + (1.0 - 1.0 / 0.3) * vbatch.y
        # vloss = torch.nn.functional.binary_cross_entropy(voutputs, vbatch.y, weights)
        vloss = criterion(voutputs, vbatch.y)
        running_vloss += vloss.item()
        running_vcorrects += batchAccuracy(voutputs, vbatch.y, vbatch.batch)
        for j in range(len(vbatch.y)):
            if voutputs[j] > 0.5:
                occ_class[vbatch.info[j]] += 1
                real_vscheduling[vbatch.label[j] - 1] += 1
                if vbatch.label[j] == 1:
                    vprob1 += voutputs[j]
                elif vbatch.label[j] == 2:
                    vprob2 += voutputs[j]

            occ_1 += get_occlusion1(voutputs, vbatch.info, vbatch.batch)
        tot_vnodes += len(vbatch.batch)
        step += 1

    avg_vloss = running_vloss / step
    avg_vcorrects = running_vcorrects / tot_vnodes

    y_loss['val'].append(avg_vloss)
    y_err['val'].append(avg_vcorrects)
    print('True scheduling of predicted as first (VAL): ', real_vscheduling)
    print(f'\nAverage probability (VAL) of first being first: {float(vprob1 / real_vscheduling[0]):.4f},'
          f'\t or second: {float( vprob2 / real_vscheduling[1]):.4f}')
    print('Occlusion property for classified as first (VAL): ', occ_class)
    print('Occlusion property for node with higher probability (VAL): ', occ_1)
    return avg_vloss, avg_vcorrects


def test(BestModel, Phase):
    model.eval()
    all_y_pred = []
    all_y_true = []
    real_tscheduling = np.zeros(18)
    tprob1 = 0
    tprob2 = 0
    occ_1 = np.zeros(4)
    occ_class = np.zeros(4)

    for i, tbatch in enumerate(test_loader):
        # tbatch.to(device)
        pred = model(tbatch)
        for j in range(len(tbatch.y)):
            if pred[j] > 0.5:
                occ_class[tbatch.info[j]] += 1
                real_tscheduling[tbatch.label[j] - 1] += 1
                if tbatch.label[j] == 1:
                    tprob1 += pred[j]
                elif tbatch.label[j] == 2:
                    tprob2 += pred[j]

            occ_1 += get_occlusion1(pred, tbatch.info, tbatch.batch)

        for j in range(len(pred)):
            if pred[j] > 0.5:
                pred[j] = 1
            else:
                pred[j] = 0
        y = torch.zeros_like(tbatch.y)
        for j in range(len(tbatch.y)):
            if tbatch.y[j] > 0.5:
                y[j] = 1
        all_y_pred.append(pred.cpu().detach().numpy())
        all_y_true.append(y.cpu().detach().numpy())

    all_preds = np.concatenate(all_y_pred)
    all_labels = np.concatenate(all_y_true)
    calculate_metrics(all_preds, all_labels, BestModel, Phase)
    print('True scheduling of predicted as first (TEST): ', real_tscheduling)
    print(f'\nAverage probability (TEST) of first being first: {float(tprob1 / real_tscheduling[0]):.4f},'
          f'\t or second: {float(tprob2 / real_tscheduling[1]):.4f}')
    print('Occlusion property for classified as first (TEST): ', occ_class)
    print('Occlusion property for node with higher probability (TEST): ', occ_1)


def calculate_metrics(y_pred, y_true, BestModel, Phase):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_true, y_pred)}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average=None)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Precision: {precision_score(y_true, y_pred, average=None)}")
    print(f"Recall: {recall_score(y_true, y_pred, average=None)}")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred))
    disp.plot()
    if BestModel:
        plt.savefig(os.path.join('./plots/{}', 'CM{}_{}_{}_{}_L2{}_{}.jpg'.format(goal, Phase, hiddenLayers, cfg.NL, batchSize, weightDecay, SeedNum)))


def draw_curve(current_epoch, cfg, lastEpoch, best_loss, best_vloss, best_accuracy, best_vaccuracy):
    if current_epoch != lastEpoch:
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train', linewidth=1)
        ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val', linewidth=1)
        ax1.plot(x_epoch, y_err['train'], 'bo-', label='train', linewidth=1)
        ax1.plot(x_epoch, y_err['val'], 'ro-', label='val', linewidth=1)
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    elif current_epoch == lastEpoch:
        ax0.text(0.5, 0.5, 'T' + str(best_loss),
                 horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes)
        ax0.text(0.5, 0.2, 'V' + str(best_vloss),
                 horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes)
        ax1.text(0.5, 0.5, 'T' + str(best_accuracy),
                 horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.text(0.5, 0.2, 'V' + str(best_vaccuracy),
                 horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    fig.savefig(os.path.join('./plots/{}', 'train_{}_{}_{}_L2{}_{}.jpg'.format(goal, cfg.HL, cfg.NL, cfg.BATCHSIZE, cfg.WEIGHTDECAY, cfg.SEEDNUM)))


# Main
def train():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    best_vloss = 1_000_000.
    best_loss = 1_000_000.
    best_vaccuracy = 0
    best_accuracy = 0
    early_stopping_counter = 0
    NumEpochs = 300
    lastEpoch = NumEpochs + 1
    lastEpochlist = []
    Phase = 'train'
    BestModel = False
    for epoch in trange(1, NumEpochs + 1):
        if early_stopping_counter <= 10:
            # Training
            train_loss, train_accuracy = train_one_epoch(BestModel, Phase)
            # Validation
            val_loss, val_accuracy = validation()

            print(f'\nTrain Loss: {train_loss:.4f}, \tValidation Loss: {val_loss:.4f}')  # , \tTest Loss: {test_loss:.4f}
            print(f'\nTrain Accuracy: {train_accuracy:.4f}, \tValidation Accuracy: {val_accuracy:.4f}')

            scheduler.step(val_loss)

            # draw curve
            draw_curve(epoch, cfg, lastEpoch, best_loss, best_vloss, best_accuracy, best_vaccuracy)

            # Log the running loss averaged per batch for training, and validation
            writer.add_scalars('Training vs. Validation Loss', {'Training': train_loss, 'Validation': val_loss}, epoch + 1)
            writer.flush()

            # Track the best performance, and save the model's state
            if val_loss < best_vloss:
                best_vloss = val_loss
                model_path = 'best_models/{}/model_best'.format(goal)
                torch.save(model.state_dict(), model_path)
                early_stopping_counter = 0
                BestModel = True
            else:
                early_stopping_counter += 1
                BestModel = False

            if train_loss < best_loss:
                best_loss = train_loss

            if train_accuracy > best_accuracy:
                best_accuracy = train_accuracy

            if val_accuracy > best_vaccuracy:
                best_vaccuracy = val_accuracy
        else:
            lastEpochlist.append(epoch)
            continue

    if early_stopping_counter >= 10:
        print("Early stopping due to no improvement.")

    # Test
    Phase = 'test'
    BestModel = True  # to plot the confusion matrix of the loss
    model.load_state_dict(torch.load(model_path))
    test(BestModel, Phase)

    # to print loss and accuracy best values
    if len(lastEpochlist) > 0:
        lastEpoch = int(lastEpochlist[0])
    draw_curve(lastEpoch, cfg, lastEpoch, best_loss, best_vloss, best_accuracy, best_vaccuracy)


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('main', device)

    if device.type == 'cpu':
        cfg = config_scheduling
    else:
        cfg = config_scheduling_gpu

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
    train_path = 'dataset/{}/data_train/'.format(goal)
    train_dataset = cfg.DATASET(train_path)
    val_path = 'dataset/{}/data_val/'.format(goal)
    val_dataset = cfg.DATASET(val_path)
    test_path = 'dataset/{}/data_test/'.format(goal)
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
    criterion = torch.nn.BCELoss()  # weight it!! classes are imbalanced
    batchAccuracy = cfg.ACCURACY()

    # Parameters for plots

    y_loss = {}
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")

    train()

# Clear cuda cache
with torch.no_grad():
    torch.cuda.empty_cache()
