from transformers import AutoTokenizer, pipeline, AutoModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
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
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
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
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

print("flag")