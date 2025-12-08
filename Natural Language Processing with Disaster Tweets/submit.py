import pandas as pd
from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification
from datasets import Dataset
from tokenizer import tokenizer,tokenize
import torch
import numpy as np
test_data_path=r"E:\Kaggle\src\Natural Language Processing with Disaster Tweets\Data\test.csv"
model_dir=r"E:\Kaggle\src\Natural Language Processing with Disaster Tweets\Data\res\checkpoint-5715"
submit_data_path=r"E:\Kaggle\src\Natural Language Processing with Disaster Tweets\Data\sample_submission.csv"
device='cuda'
print("--- 开始读取csv文件 ---")
test_data=pd.read_csv(test_data_path)
print("--- 成功读取csv文件 ---")
test_data['input_text']=test_data['keyword'].fillna("unknown")+' '+test_data['location'].fillna("unknown")+" "+test_data['text']
test_dataset=Dataset.from_dict({'text':(test_data['input_text'].tolist())})

test_tokenized=test_dataset.map(tokenize,batched=True)
test_tokenized.set_format('torch',columns=['input_ids','attention_mask'])
model=DistilBertForSequenceClassification.from_pretrained(model_dir)
model=model.to(device)
print("--- 开始推理 ---")
with torch.no_grad():
    input_ids=(test_tokenized['input_ids']).to(device)
    attention_mask=(test_tokenized['attention_mask'].to(device))
    outputs=model(input_ids=input_ids,attention_mask=attention_mask)
    logits=outputs.logits.cpu().detach().numpy()
print("--- 推理完成 ---")
preds=np.argmax(logits,axis=-1)

submit_data=pd.read_csv(submit_data_path)

submit_data['target']=preds

print("--- 即将写入csv文件 ---")
submit_data.to_csv(submit_data_path,index=False)
print("--- 成功写入csv文件 ---")