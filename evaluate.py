import river.metrics as rmetrics
from clustpy.metrics import unsupervised_clustering_accuracy, PairCountingScores
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Purity(rmetrics.base.MultiClassMetric):

    @property
    def works_with_weights(self):
        return False

    def get(self):
        purity = 0
        num = 0
        print(self.cm)
        for j in self.cm.classes:  # predict
            maxi = 0
            for i in self.cm.classes:  # label
                num += self.cm[i][j]
                if maxi < self.cm[i][j]:
                    maxi = self.cm[i][j]
            purity += maxi

        return purity / num

def printMetrics(labels, predictions):
    stream_ari = rmetrics.AdjustedRand()
    stream_nmi = rmetrics.NormalizedMutualInfo()
    stream_ami = rmetrics.AdjustedMutualInfo()
    stream_completeness = rmetrics.Completeness()
    stream_fowl = rmetrics.FowlkesMallows()
    stream_homogeneity = rmetrics.Homogeneity()
    stream_purity = Purity()

    for i in range(len(predictions)):
        pred = predictions[i]
        label = labels[i]
        stream_ari.update(label, pred)
        stream_nmi.update(label, pred)
        stream_ami.update(label, pred)
        stream_completeness.update(label, pred)
        stream_fowl.update(label, pred)
        stream_homogeneity.update(label, pred)
        stream_purity.update(label, pred)

    ari = stream_ari.get()
    nmi = stream_nmi.get()
    ami = stream_ami.get()
    completeness = stream_completeness.get()
    fowl = stream_fowl.get()
    homogeneity = stream_homogeneity.get()
    purity = stream_purity.get()
    acc = unsupervised_clustering_accuracy(np.array(labels), np.array(predictions))
    paircounting = PairCountingScores(np.array(labels), np.array(predictions))
    f1 = paircounting.f1()
    prec = paircounting.precision()
    recall = paircounting.recall()

    clunum = len(np.unique(predictions))
    trueclunum = len(np.unique(labels))
    outputstring =  f"Acc: {acc} NMI: {nmi} ARI: {ari} AMI: {ami} Pur.: {purity} Prec.: {prec} Rec.: {recall} F1: {f1} Comp.: {completeness} Fowl.: {fowl} Homo.: {homogeneity} Clu.Num.: {clunum} True Clu.Num {trueclunum}"
    logger.info(
       outputstring)
    logger.info(stream_purity.cm)
    return  f"Acc: {acc} NMI: {nmi} ARI: {ari} AMI: {ami} Pur.: {purity} Prec.: {prec} Rec.: {recall} F1: {f1} Comp.: {completeness} Fowl.: {fowl} Homo.: {homogeneity} Clu.Num.: {clunum} True Clu.Num {trueclunum}"