import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score


if __name__ == "__main__":
    exp_name = "Inference"

    df = pd.read_csv(f"../{exp_name}_Widowed_Multitask.csv", index_col=0) 
    
    tasks = ["home", "mortality", "icu"]
    
    for task in tasks:    
        yt = df[f"ytrue_{task}"].values
        yp = df[f"yprob_{task}"].values

        #if np.unique(yt).shape[0] == 1:
        #    continue

        f1 = f1_score(yt, np.round(yp))

        precision, recall, threshold = precision_recall_curve(yt, yp)
        auc_precision_recall = auc(recall, precision)

        auc_roc = roc_auc_score(yt, yp)
    
        print(f"Evaluation Metrics for '{task}':") 
        print(f"F1 Score: {f1:.5f}")
        print(f"Area under the Precision-Recall Curve: {auc_precision_recall:.5f}")
        print(f"Area under the ROC-AUC Curve: {auc_roc:.5f}\n")

