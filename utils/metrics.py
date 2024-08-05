
import numpy as np


class IOUMetric:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        '''
        a = np.array([0,1,2,2])
        aa = np.array([0,1,2,1])
        a.astype(int)*3:
            array([0, 3, 6, 6])
        a.astype(int)*3+aa:
            array([0, 4, 8, 7])
        np.bincount(a.astype(int)*3+aa).reshape(3,3):
            array([[1, 0, 0],
           [0, 1, 0],
           [0, 1, 1]])
        '''
        mask = (label_true >= 0) & (label_true < self.num_classes)#筛选符合需求的class类别
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + #这一行相当于给每个类别的真实标签开辟了一行计数器
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes) #预测值为类别几就放到真实值那一行的第几个位置
        #因此，每行是真实值，每列是预测值
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        # Precision and Recall,仅mask类别计算precision和recall
        precision = np.diag(self.hist) / self.hist.sum(axis=0)#按列求和，预测为某类的汇总
        recall = np.diag(self.hist) / self.hist.sum(axis=1)#按列求和，某类的真实值汇总

        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return acc, acc_cls, iu, mean_iu, fwavacc, precision, recall, f1


