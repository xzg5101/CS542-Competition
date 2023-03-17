import json

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
    train_questions.append(
        {
            'id':str(question['id']),
            'question':str(question['question']),
            'label': str(question['answer']),
            'answer':str(question['answer']),
            'background': str(question['background']),
            'publish_time':str(question['publish_time']),
            'close_time':str(question['close_time']),
            'tags':str(question['tags']),
            'answer': str(question['answer']),
            'choices': str(question['choices']),
        }
    )

#2797
print(f"{len(train_questions)} training questions found")

# Serializing json
json_object = json.dumps(train_questions, indent=4)
 
# Writing to sample.json
with open("training.json", "w") as outfile:
    outfile.write(json_object)