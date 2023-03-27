from transformers import AutoTokenizer, pipeline, AutoModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

# transformer pipeline
#tg = pipeline("text-generation", model = 'gpt2')

train_dataset = load_dataset("json", data_files="datasets/training.json")
test_dataset = load_dataset("json", data_files="datasets/testing.json")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["question"], padding="max_length", truncation=True, )

#tokenizer = AutoTokenizer.from_pretrained("gpt2")
#dataset.to_json("./myset.csv")
tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
tokenized_test_datasets = test_dataset.map(tokenize_function, batched=True)
#print(tokenized_datasets)

model = AutoModel.from_pretrained("gpt2")

training_args = TrainingArguments(output_dir="test_trainer")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_test_datasets,
    compute_metrics=compute_metrics,
)

print("flag")