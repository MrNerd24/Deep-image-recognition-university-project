import pandas as pd
import random
import numpy as np


class DataBalancer():

    def __init__(self):
        self.filesAndLabels = pd.read_csv("file_to_labels_table.csv", index_col=False)


    def balanceData(self, ids, amountOfNewIds, baseIdList = []):
        ids = list(ids).copy()
        baseIdList = list(baseIdList).copy()
        random.shuffle(ids)
        newIdsList = np.concatenate((np.array(baseIdList), np.zeros(amountOfNewIds)))
        ids = np.array(ids)
        idAmounts = pd.DataFrame(np.concatenate((np.matrix(ids).T,np.zeros((len(ids),1))), axis=1), columns=["ids", "amountUsed"])
        classAmounts = np.zeros(len(self.filesAndLabels.columns)-1)
        idsByClass = self.getIdsByClass(ids)

        for baseId in baseIdList:
            self.increaseIdAmountUsedById(idAmounts, baseId)
            classAmounts = self.increaseClassAmountsById(classAmounts, baseId)

        for i in range(len(baseIdList), len(baseIdList) + amountOfNewIds):
            leastRepresentedClass = self.getleastRepresentedClass(classAmounts)
            newId = self.getNextBalancingId(leastRepresentedClass, idsByClass, idAmounts)
            self.increaseIdAmountUsedById(idAmounts, newId)
            classAmounts = self.increaseClassAmountsById(classAmounts, newId)
            newIdsList[i] = newId 
        
        return newIdsList

    def getleastRepresentedClass(self, classAmounts):

        leastRepresentation = float('inf')
        leastRepresentedClass = None
        
        for i, column in enumerate(self.filesAndLabels.columns[1:]):
            representation = classAmounts[i]
            if representation < leastRepresentation:
                leastRepresentation = representation
                leastRepresentedClass = column
        
        return leastRepresentedClass

    def getNextBalancingId(self, leastRepresentedClass, idsByClass, idAmounts):
        imageIds = idsByClass[leastRepresentedClass]
        validAmounts = idAmounts[idAmounts.ids.isin(imageIds)]
        minUsed = validAmounts["amountUsed"].min()
        imageId = int(validAmounts[validAmounts["amountUsed"] == minUsed].iloc[0]["ids"])
        return imageId
            
    def getRepresentations(self, ids):
        idsByClass = self.getIdsByClass(ids)
        for column in idsByClass:
            idsByClass[column] = len(idsByClass[column])
        return idsByClass

    def increaseClassAmountsById(self, classAmounts, imageId):
        filename = "im{}.jpg".format(imageId)
        labels = self.filesAndLabels[self.filesAndLabels["filename"] == filename].iloc[0].values[1:]
        return classAmounts + labels

    def increaseIdAmountUsedById(self, idAmounts, imageId):
        idAmounts.loc[idAmounts[idAmounts["ids"] == imageId].index[0], "amountUsed"] += 1

    def getIdsByClass(self, ids):
        idsByClass = {}
        availableData = self.filesAndLabels.iloc[ids-1]
        for column in self.filesAndLabels.columns[1:]:
            dataWithClass = availableData[availableData[column] == 1]
            imageIds = dataWithClass["filename"].map(lambda x: int(x[2:-4])).values
            idsByClass[column] = imageIds
        return idsByClass