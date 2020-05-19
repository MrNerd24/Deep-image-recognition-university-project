import random
import matplotlib.pyplot as plt


def showExamplePredictions(model, dataset, numberOfExamples):
    self.image_id_and_labels = pd.read_csv("file_to_labels_table.csv", index_col=False)
    model.eval()
    decisionThresholds = torch.tensor([decisionThreshold]*numberOfLabels)
    for i in range(numberOfExamples):
        data = dataset[random.randint(0, len(dataset))]
        inputs = data["imageTensor"].cpu().unsqueeze(0)
        image = data["imagePhil"]
        labels = data["labelsStrings"]
        outputs = model(inputs).cpu()
        predLabels = (outputs >= decisionThresholds).numpy()[0]
        predStringLabels = [self.image_id_and_labels.columns[i+1] for i in range(len(predLabels)) if predLabels[i] == 1]

        plt.imshow(image)
        plt.show()
        print("True labels: ", labels)
        print("Predicted labels:" predStringLabels)
        print()