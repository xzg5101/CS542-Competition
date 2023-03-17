from transformers import AutoTokenizer, pipeline, AutoModel
from datasets import load_dataset
from transformers import TrainingArguments
import numpy as np
import evaluate

# transformer pipeline
#tg = pipeline("text-generation", model = 'gpt2')

dataset = load_dataset("json", data_files="training.json")
print(dataset)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
    return tokenizer(examples["question"], padding="max_length", truncation=True, )

#tokenizer = AutoTokenizer.from_pretrained("gpt2")
#dataset.to_json("./myset.csv")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(tokenized_datasets)

model = AutoModel.from_pretrained("gpt2")

training_args = TrainingArguments(output_dir="test_trainer")

metric = evaluate.load("accuracy")