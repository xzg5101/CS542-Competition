import json
from transformers import AutoTokenizer, pipeline
#from datasets import load_dataset


train_questions = json.load(open('training.json', encoding='utf-8')) # from the Autocast dataset

# use first one as example
q = train_questions[0]['question']+' Please answer a number.'

prompt = train_questions[0]['question'] + ' Here are your options: ' + train_questions[0]['choices'] + ' the correct one is'
print(prompt)

# transformer pipeline
tg = pipeline("text-generation", model = 'gpt2')

#context = train_questions[0]['background']

result = tg(prompt, max_length=100)
print(result)