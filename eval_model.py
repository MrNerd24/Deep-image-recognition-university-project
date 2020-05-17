def test_model(model, data, numberOfLabels=14, decisionThreshold=0.5):
    model.eval()
    decisionThresholds = torch.tensor([decisionThreshold]*numberOfLabels)
    loader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=4, collate_fn=train_model.collate)
    y_hats = []
    y_trues = []
    for i, data in enumerate(loader):
        inputs = data["input"].to(device)
        labels = data["labels"].to(device)
        outputs = model(inputs)
        pred_labels = (outputs >= decisionThresholds).numpy()[0]
        y_hat = [int(e) for e in pred_labels]
        y_true = list(labels.numpy()[0])
        y_hats.append(y_hat)
        y_trues.append(y_true)
    return y_hats, y_trues
