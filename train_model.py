import time
import copy
import torch
import math

def multilabelCrossEntropyLoss(output, target):
    loss = -torch.sum(target*torch.log(output) + (1-target)*torch.log(1-output))
    return loss

def collate(batch):
        return {
            "input": torch.stack([item["imageTensor"] for item in batch]),
            "labels": torch.stack([item["labelsTensor"] for item in batch])
        }


def train_model(model, trainDataset, valDataset, device, numberOfEpochs=5, numberOfLabels = 14, criterion = multilabelCrossEntropyLoss, optimizer=None, predictionCutoffs=None, scheduler=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if optimizer == None:
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)


    if predictionCutoffs == None:
        predictionCutoffs = torch.tensor([0.5]*numberOfLabels)
    predictionCutoffs = predictionCutoffs.to(device)

    datasets = {
        "train": trainDataset,
        "val": valDataset,
    }

    batchSize = 4

    dataloaders = {
        "train": torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=4, collate_fn=collate),
        "val": torch.utils.data.DataLoader(valDataset, batch_size=batchSize, shuffle=True, num_workers=4, collate_fn=collate)
    }

    for epoch in range(numberOfEpochs):
        print('Epoch {}/{}'.format(epoch+1, numberOfEpochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            print("Training..." if phase == "train" else "Validating...")
            print()

            lastSeenProgressProcent = -10

            # Iterate over data.
            for index, data in enumerate(dataloaders[phase]):
                inputs = data["input"].to(device)
                labels = data["labels"].to(device)

                progress = ((index+1)*batchSize)/(len(datasets[phase]))
                progressProcent = math.floor(progress * 100)

                if progressProcent >= lastSeenProgressProcent + 10 :
                    print("Progress: {}%".format(progressProcent))
                    lastSeenProgressProcent = progressProcent

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = outputs >= predictionCutoffs
                    preds = preds.float()
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / (len(datasets[phase])*len(predictionCutoffs))
            epoch_acc = running_corrects.double() / (len(datasets[phase])*len(predictionCutoffs))

            print()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model