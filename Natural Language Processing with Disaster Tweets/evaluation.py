from sklearn.metrics import accuracy_score ,precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits,labels=eval_pred
    preds=logits.argmax(-1)
    acc=accuracy_score(labels,preds)
    precision,recall,f1,_=precision_recall_fscore_support(labels,preds,average='binary')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

