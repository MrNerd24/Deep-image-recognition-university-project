from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import train_model
import torch


def test_model(model, dataset, numberOfLabels=14, decisionThreshold=0.5):
    model.eval()
    decisionThresholds = torch.tensor([decisionThreshold]*numberOfLabels)
    y_hats = []
    y_trues = []
    for data in dataset:
        inputs = data["imageTensor"].cpu().unsqueeze(0)
        labels = data["labelsTensor"]
        outputs = model(inputs).cpu()
        pred_labels = (outputs >= decisionThresholds).numpy()[0]
        y_hat = [int(e) for e in pred_labels]
        y_true = list(labels.numpy())
        y_hats.append(y_hat)
        y_trues.append(y_true)
    return y_hats, y_trues


def get_metric(y_trues, y_hats, metric, average='micro'):
    """
    Get precision, recall or f1 score for multilabel classification.
    
    Args:
      - y_trues: ground truth 2d array (n_samples, n_classes)
      - y_hats: predicted 2d array (n_samples, n_classes)
      - metric: {'precision', 'recall', 'f1', 'accuracy'}
    Kwargs:
      - average: {None, 'binary', 'micro', 'macro', 'samples', 'weighted'}
    
    Returns score (scalar)
    """
    if metric not in ['precision', 'recall', 'f1', 'accuracy']:
        return None
    
    if metric == 'accuracy':
        all_hats = []
        all_trues = []
        for hat, true in zip(y_hats, y_trues):
            all_hats.extend(hat)
            all_trues.extend(true)
        return accuracy_score(all_trues, all_hats)
    
    return {
        'precision': precision_score(y_trues, y_hats, average=average),
        'recall': recall_score(y_trues, y_hats, average=average),
        'f1': f1_score(y_trues, y_hats, average=average)
    }[metric]
