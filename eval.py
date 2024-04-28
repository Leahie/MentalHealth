import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import evaluate


device = torch.device("cuda")

EPOCH = 1 
BATCH = 6 
SEED = 4222
LEARNING_RATE = 1e-5
SAVE_PATH = ".model/bert2"
CHECKPOINT_PATH = ".model/bert_checkpoint"
LOG_PATH = ".model/bert_checkpoint/logs"

import pandas as pd
import numpy as np

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.string_)


df = pd.read_csv(".data/Suicide_Detection_Final_Clean.csv", header=0, names = ['text', 'label'])
df = df.reset_index()
df['label'] = df['label'].map({'suicide':1, 'non-suicide':0})


train, temp = train_test_split(df, random_state=SEED, test_size=0.25, stratify=df['label'])
val, test = train_test_split(temp,random_state=SEED, test_size=0.5, stratify=temp['label'])

# HuggingFace
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_metric

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def dataset_conversion(train, test, val):
  train.reset_index(drop=True, inplace=True)
  test.reset_index(drop=True, inplace=True)
  val.reset_index(drop=True, inplace=True)

  train_dataset = Dataset.from_pandas(train)
  test_dataset = Dataset.from_pandas(test)
  val_dataset = Dataset.from_pandas(val)

  return DatasetDict({"train": train_dataset,
                      "test": test_dataset,
                      "val": val_dataset})

raw_datasets = dataset_conversion(train, test, val)

def tokenize_function(dataset):
    return tokenizer(dataset["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


SAMPLE_SIZE =  3500
train_dataset = tokenized_datasets["train"].shuffle(seed=SEED).select(range(SAMPLE_SIZE))
test_dataset = tokenized_datasets["test"].shuffle(seed=SEED).select(range(SAMPLE_SIZE))
val_dataset = tokenized_datasets["val"].shuffle(seed=SEED).select(range(SAMPLE_SIZE))


# train_dataset = tokenized_datasets["train"]
# test_dataset = tokenized_datasets["test"]
# val_dataset = tokenized_datasets["val"]

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    overwrite_output_dir = True,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCH,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    seed=SEED,
    logging_dir=LOG_PATH,
    save_strategy="steps",
    save_steps=1500
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


model = AutoModelForSequenceClassification.from_pretrained("./.model/bert2", num_labels=2)   


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print(trainer.predict(test_dataset).metrics)