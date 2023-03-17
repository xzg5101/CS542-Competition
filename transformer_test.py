import json
from transformers import AutoTokenizer, pipeline
#from datasets import load_dataset

# import data
autocast_questions = json.load(open('autocast_questions.json', encoding='utf-8')) # from the Autocast dataset
test_questions = json.load(open('autocast_competition_test_set.json', encoding='utf-8'))
test_ids = [q['id'] for q in test_questions]

# extract training data
train_questions = []
for question in autocast_questions:
    if question['id'] in test_ids: # skipping questions in the competition test set
        continue
    if question['answer'] is None: # skipping questions without answer
        continue
    train_questions.append(question)

# use first one as example
q = train_questions[0]['question']+' Please answer a number.'

# transformer pipeline
tg = pipeline("text-generation", model = 'gpt2')

#context = train_questions[0]['background']

result = tg(q, max_length=100)
print(result)