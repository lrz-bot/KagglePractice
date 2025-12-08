from transformers import DistilBertTokenizerFast

tokenizer=DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'],padding="max_length",truncation=True,max_length=128)