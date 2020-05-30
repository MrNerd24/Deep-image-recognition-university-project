import time
import copy
import torch
import math

def multilabelCrossEntropyLoss(output, target):
    output = 0.99*output + 0.005
    loss = -torch.sum(target*torch.log(output) + (1-target)*torch.log(1-output))
    return loss

def collate(batch):
        return {
            "input": torch.stack([item["imageTensor"] for item in batch]),
            "labels": torch.stack([item["labelsTensor"] for item in batch])
        }


def train_model(model, trainDataset, valDataset, device, numberOfEpochs=5, numberOfLabels=14, criterion=multilabelCrossEntropyLoss, optimizer=None, decisionThresholds=None, scheduler=None, selectionCriteria="f1", selectionCriteriaBiggerBetter = True, returnLogs=False, optimizerStep=None, batchSize=4):
    # FIXME: implement selection_criteria: {'accuracy', 'loss'} -> select model by largest accuracy or smallest loss
    since = time.time()
    bestModelWeights = copy.deepcopy(model.state_dict())
    bestSelectionCriteriaValue = 0.0
    trainingLogs = None

    if optimizer == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    if not decisionThresholds:
        decisionThresholds = torch.tensor([0.5]*numberOfLabels)
    decisionThresholds = decisionThresholds.to(device)

    dataloaders = {
        "train": torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=min(8, batchSize), collate_fn=collate),
        "val": torch.utils.data.DataLoader(valDataset, batch_size=batchSize, shuffle=True, num_workers=min(8, batchSize), collate_fn=collate)
    }

    epochPhases = [["val"]] + [["train", "val"]]*numberOfEpochs

    for epoch in range(numberOfEpochs+1):
        print('Epoch {}/{}'.format(epoch, numberOfEpochs))
        print('-' * 10)

        for phase in epochPhases[epoch]:
            
            stats = doEpoch(phase, model, dataloaders[phase], device, optimizer, criterion, decisionThresholds, scheduler, optimizerStep)

            trainingLogs = saveStatsToLog(stats, phase, epoch, trainingLogs)
            
            if phase == "val":
                selectionCriteriaValue = stats[selectionCriteria]

                betterMetric = False
                if selectionCriteriaBiggerBetter:
                    betterMetric = selectionCriteriaValue > bestSelectionCriteriaValue
                else:
                    betterMetric = selectionCriteriaValue < bestSelectionCriteriaValue

                if(betterMetric):
                    bestSelectionCriteriaValue = selectionCriteriaValue
                    bestModelWeights = copy.deepcopy(model.state_dict())


    timeElapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        timeElapsed // 60, timeElapsed % 60))
    print('Best {}: {:4f}'.format(selectionCriteria, bestSelectionCriteriaValue))

    # load best model weights
    model.load_state_dict(bestModelWeights)
    if returnLogs:
        return model, trainingLogs
    return model


def doEpoch(phase, model, dataloader, device, optimizer, criterion, decisionThresholds, scheduler, optimizerStep):

    since = time.time()

    if phase == 'train':
        model.train()  # Set model to training mode
    else:
        model.eval()   # Set model to evaluate mode

    runningLoss = 0.0
    runningTruePositives = 0
    runningFalsePositives = 0
    runningTrueNegatives = 0
    runningFalseNegatives = 0

    print("Training..." if phase == "train" else "Validating...")
    print()

    lastSeenProgressProcent = -1

    # Iterate over data.
    for index, data in enumerate(dataloader):
        inputs = data["input"].to(device)
        labels = data["labels"].to(device)

        lastSeenProgressProcent = printProgress(index, dataloader.batch_size, len(dataloader.dataset), lastSeenProgressProcent)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(phase == "train"):
            outputs = model(inputs)
            preds = outputs >= decisionThresholds
            preds = preds.float()
            loss = criterion(outputs, labels)

            if(phase == "train"):
                loss.backward()
                if optimizerStep is None:
                    optimizer.step()
                else:
                    optimizerStep(optimizer)

        # statistics
        runningLoss += loss.item()
        runningTruePositives += torch.sum((preds == 1) & (labels.data == 1)).item()
        runningTrueNegatives += torch.sum((preds == 0) & (labels.data == 0)).item()
        runningFalsePositives += torch.sum((preds == 1) & (labels.data == 0)).item()
        runningFalseNegatives += torch.sum((preds == 0) & (labels.data == 1)).item()

    if scheduler:
        scheduler.step()

    stats = calculateEpochStats(runningLoss, runningTruePositives, runningTrueNegatives, runningFalsePositives, runningFalseNegatives, since)

    print()
    print()
    printStats(stats)
    print()

    return stats


def printProgress(index, batchSize, datasetLength, lastSeenProgressProcent):
    progress = ((index+1)*batchSize)/(datasetLength)
    progressProcent = math.floor(progress * 100)

    if progressProcent >= lastSeenProgressProcent + 1 :
        print("\rProgress: {}%".format(progressProcent), end="")
        lastSeenProgressProcent = progressProcent
    return lastSeenProgressProcent

def calculateEpochStats(loss, truePositives, trueNegatives, falsePositives, falseNegatives, since):
    epochTime = time.time() - since
    numOfPredictions = truePositives + trueNegatives + falsePositives + falseNegatives
    averageLoss = loss / numOfPredictions
    accuracy = (truePositives + trueNegatives) / numOfPredictions
    precision = truePositives / (falsePositives + truePositives)
    recall = truePositives / (falseNegatives + truePositives)
    f1 = 2*(precision*recall)/(precision+recall)
    
    return {
        "duration": epochTime,
        "loss": averageLoss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true positives": truePositives,
        "true negatives": trueNegatives,
        "false positives": falsePositives,
        "false negatives": falseNegatives,
    }

def printStats(stats):
    print('Loss: {:.4f} Accuracy: {:.4f}'.format(stats["loss"], stats["accuracy"]))
    print('Precision: {:.4f} Recall: {:.4f}'.format(stats["precision"], stats["recall"]))
    print('F1: {:.4f} Duration: {:.0f}m {:.0f}s'.format(stats["f1"], stats["duration"] // 60, stats["duration"] % 60))

def saveStatsToLog(stats, phase, epoch, trainingLogs):
    if trainingLogs is None:
        trainingLogs = {
            "train": None,
            "val": None,
        }

    if trainingLogs[phase] is None:
        trainingLogs[phase] = {}
        for stat in stats:
            trainingLogs[phase][stat] = []

    for stat in stats:
        trainingLogs[phase][stat].append(stats[stat])

    return trainingLogs
