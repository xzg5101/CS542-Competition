from transformers import AutoTokenizer, pipeline, AutoModel
from datasets import load_dataset, ClassLabel
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2")
metric = evaluate.load("accuracy")
labels = ClassLabel(names_file='datasets/labels.json')

def tokenize_function(data):
    return tokenizer(data["question"], padding="max_length", truncation=True, )
# TODO
# Find a way to tokenize the labels
# the labels are strings
def tokenize(batch):
    tokenized_batch = tokenizer(batch['question'], padding=True, truncation=True, max_length=128)
    tokenized_batch['label'] = labels.str2int(batch['label'])
    return tokenized_batch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


data_files={'train': 'datasets/training.json', 'test': 'datasets/testing.json'}
dataset = load_dataset("json", data_files=data_files)



tokenizer.pad_token = tokenizer.eos_token
tokenized_datasets = dataset.map(tokenize, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))


training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

print("flag")