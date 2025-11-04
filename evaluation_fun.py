import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import auc, precision_recall_curve, matthews_corrcoef, confusion_matrix

class evaluation_index:
    def __init__(self, label, score, ndcg_k):
        self.score = score
        self.label = label
        self.ndcg_k = ndcg_k

    def evaluation(self):
        # AUC
        self.AUC = roc_auc_score(self.label, self.score)

        # AUPR
        Cur_precision, Cur_recall, _thresholds = precision_recall_curve(self.label, self.score)
        self.AUPR = auc(Cur_recall, Cur_precision)
        # NDCG@k
        df = pd.DataFrame(columns=["y_pred", "y_true"], data=np.array([list(self.score), list(self.label)]).T)
        self.ndcg = self.get_ndcg(df, self.ndcg_k)
        # average precision
        self.ap = average_precision_score(self.label, self.score)
        # Reciprocal Rank
        self.rr = self.reciprocal_rank(self.label, self.score)
        # Reciprocal Rank@10
        self.rr10 = self.reciprocal_rank10(self.label, self.score)
        # Hit Ratio
        self.HR = self.hit_ratio(self.label, self.score, 10)
        # ROC
        self.ROC = []
        temp = list(range(1, 22, 1))
        # for i in range(1, 51, 1):
        for i in temp:
            self.ROC.append(self.ROC_x(self.label, self.score, i))

    # ---------------------------------------------------------------------------------------
    # NDCG@k
    def get_dcg(self, y_pred, y_true, k):
        df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})
        df = df.sort_values(by="y_pred", ascending=False)
        df = df.iloc[0:k, :]
        dcg = (2 ** df["y_true"] - 1) / np.log2(np.arange(1, df["y_true"].count() + 1) + 1)
        dcg = np.sum(dcg)
        return dcg

    def get_ndcg(self, df, k):
        dcg = self.get_dcg(df["y_pred"], df["y_true"], k)
        idcg = self.get_dcg(df["y_true"], df["y_true"], k)
        ndcg = dcg / idcg
        return ndcg

    # ---------------------------------------------------------------------------------------
    # reciprocal rank
    def reciprocal_rank(self, label, score):
        descend_score = -np.sort(-score)
        descend_label = label[np.argsort(-score)]

        pos_loc = np.where(descend_label == 1)

        rr = 1/(pos_loc[0][0]+1)
        return rr

    # ---------------------------------------------------------------------------------------
    # reciprocal rank @10
    def reciprocal_rank10(self, label, score):
        descend_score = -np.sort(-score)
        descend_label = label[np.argsort(-score)]

        gt = [1]
        score = 0.0
        for rank, item in enumerate(descend_label[:10]):
            if item in gt:
                score = 1.0 / (rank + 1.0)
                break

        return score
    # ---------------------------------------------------------------------------------------
    # reciprocal rank @10
    def hit_ratio(self, label, score, x):
        descend_score = -np.sort(-score)
        descend_label = label[np.argsort(-score)]

        if x > descend_label.size:
            x = descend_label.size

        pos_num = 0
        for i in range(x):
            if descend_label[i] == 1:
                pos_num += 1

        return pos_num

    # ---------------------------------------------------------------------------------------
    # ROCx
    def ROC_x(self, label, score, x):
        if x > np.where(label == 0)[0].size:
            x = np.where(label == 0)[0].size

        descend_score = -np.sort(-score)
        descend_label = label[np.argsort(-score)]

        roc_value = self.cal_roc_mu_at(x, descend_label)
        return roc_value

    def cal_roc_mu_at(self, level, y_true):
        tp = np.count_nonzero(y_true)
        all_fp = np.count_nonzero(~y_true.astype(bool))
        if all_fp < level:
            fp = all_fp
            yt_level = y_true
        else:
            df = pd.DataFrame(y_true.astype(bool), columns=["label"])
            fp_index = df.loc[~df.loc[:, "label"]].index[level - 1]
            yt_level = df.loc[:fp_index].to_numpy(int).reshape((-1,))
        fp = np.count_nonzero(~yt_level.astype(bool))
        cumsum_yt_level = np.cumsum(yt_level)
        area = cumsum_yt_level[~yt_level.astype(bool)].sum()
        if tp != 0 and fp != 0:
            roc_at_score = area / (tp * fp)
        elif tp != 0 and fp == 0:
            roc_at_score = 1.0
        elif tp == 0 and fp != 0:
            roc_at_score = 0.0
        return roc_at_score

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def evaluation_all(score,label):
    performence=evaluation_index(label,score,20)
    performence.evaluation()

    AUC = round(performence.AUC, 4)
    AUPR = round(performence.AUPR, 4)
    NDCG10 = round(performence.ndcg, 4)
    MAP = round(performence.ap, 4)
    MRR = round(performence.rr, 4)
    MRR10 = round(performence.rr10, 4)
    ROC = np.round(performence.ROC, decimals=4)

    return AUC, AUPR, NDCG10, MAP, MRR, MRR10, ROC

def compute_metrics(score, label):
    precision_1, recall_1,thres1= precision_recall_curve(label, score)
    f1_scores = (2 * precision_1 * recall_1) / (precision_1 + recall_1)
    best_f1_score_1 = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index_1 = np.argmax(f1_scores[np.isfinite(f1_scores)])
    y_predicted1_new = (score > thres1[best_f1_score_index_1])

    y_predicted1_new = y_predicted1_new.astype(int)
    mcc_1 = matthews_corrcoef(label, y_predicted1_new)
    tn,fp,fn,tp = confusion_matrix(label,y_predicted1_new).ravel()
    spe_1 = tn / float(tn + fp)
    sen_1 = tp/ float(tp + fn)
    temp_0 = tp + fp
    if temp_0 == 0:
       pre_1 = 0
    else:
       pre_1 = tp/float(tp + fp) 
    acc_1 = (tp + tn)/float(tp + fp + tn + fn)
    temp = pre_1 +sen_1
    if temp == 0:
       f1_1 = 0
    else:
       f1_1 = (2*pre_1*sen_1)/float(pre_1 +sen_1)
    print('SPE,SEN,PRE,ACC,F1_score for method:')
    print(spe_1,sen_1,pre_1,acc_1,f1_1)
    return spe_1,sen_1,pre_1,acc_1,f1_1
