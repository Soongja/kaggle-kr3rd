import numpy as np
from sklearn.metrics import f1_score
from utils.tools import one_hot_encoding


def batch_f1_score(logits, labels, mixup=False):

    logits = logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # added
    labels[labels>0] = 1

    batch_size = logits.shape[0]
    total_score = 0.0
    for i in range(batch_size):
        if mixup:
            preds = logits[i].argsort()[-2:]
        else:
            preds = np.argmax(logits[i])

        total_score += f1_score(labels[i], one_hot_encoding(preds))

    return total_score / batch_size
