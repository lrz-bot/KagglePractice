import torch
import pandas as pd
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification,Trainer,TrainingArguments 
from datasets import Dataset,DatasetDict
from sklearn.model_selection import train_test_split
from evaluation import compute_metrics
from tokenizer import tokenize,tokenizer
train_data_path=r"E:\Kaggle\src\Natural Language Processing with Disaster Tweets\Data\train.csv"
test_data_path=r"E:\Kaggle\src\Natural Language Processing with Disaster Tweets\Data\test.csv"

train_data=pd.read_csv(train_data_path)

train_data['input_text']=train_data['keyword'].fillna("unknown")+' '+train_data['location'].fillna("unknown")+" "+train_data['text']

X_train, X_val, y_train, y_val = train_test_split(
    train_data['input_text'], train_data['target'], test_size=0.2, random_state=42, stratify=train_data['target']
)

train_dataset=Dataset.from_dict({'text':X_train.tolist(),'label':y_train.tolist()})
val_dataset=Dataset.from_dict({'text':X_val.tolist(),'label':y_val.tolist()})

dataset=DatasetDict({'train':train_dataset,'val':val_dataset})



tokenized_datasets=dataset.map(tokenize,batched=True)
tokenized_datasets.set_format('torch',columns=['input_ids','attention_mask','label'])

device='cuda'

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

training_args=TrainingArguments(
    output_dir=r"E:\Kaggle\src\Natural Language Processing with Disaster Tweets\Data\res",
    num_train_epochs=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_dir=r"E:\Kaggle\src\Natural Language Processing with Disaster Tweets\Data\log",
    do_train=True,
    do_eval=True,
    eval_steps=50,
    seed=42,


)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics

)

print(" Training started...")

trainer.train()
print(" Training Completed...")

eval_results = trainer.evaluate()
print("\n Validation Results (Accuracy, Precision, Recall, F1):")
print(eval_results)



