import torch
import torch.nn as nn
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import time
import copy
from torch.utils.tensorboard import SummaryWriter


def train(model, datasets, criterion, optimizer, scheduler, epochs):
    """
    function: train neural network model normally
    parameters input:
        -model: model structure  --torch.nn.Module
        -datasets: data for training  --dictionary
            +structure: {'train': torch.utils.data.Dataset, 'val': torch.utils.data.Dataset}
        -criterion: type of loss (CrossEntropyLoss(output, target), MSELoss(output, target), vv)
        -optimizer: optimize loss (SGD, Adam, vv) --torch.optim
        -scheduler: adjust parameter(learning_rate) of optimizer --torch.optim
        -epochs: number of epochs to train
    """

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=32,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    since = time.time()

    for epoch in range(epochs):
        print('Epoch {}/{}:'.format(epoch+1, epochs))

        epoch_loss = {}
        epoch_acc = {}
        start = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history when training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics( one-hot-coding)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss[phase] = running_loss / dataset_sizes[phase]
            epoch_acc[phase] = running_corrects.double() / dataset_sizes[phase]

            if phase == 'val' and epoch_acc[phase] > best_acc:
                best_acc = epoch_acc[phase]
                best_model_wts = copy.deepcopy(model.state_dict())
        time_per_epoch = time.time() - start
        print('time: {:.0f}m{:.0f}s    train_loss: {:.4f}    train_acc: {:.4f}    val_loss: {:.4f}    val_acc: {:.4f}'.format(
            time_per_epoch//60, time_per_epoch % 60, epoch_loss['train'], epoch_acc['train'], epoch_loss['val'], epoch_acc['val']
        ))
        print()

    time_elapsed = time.time() - since
    print('\n'+'_'*20)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    # load or save model here( example load model)
    model.load_state_dict(best_model_wts)
    return model
