import json

# import data
autocast_questions = json.load(open('autocast_questions.json', encoding='utf-8')) # from the Autocast dataset
test_set = json.load(open('autocast_competition_test_set.json', encoding='utf-8'))
test_ids = [q['id'] for q in test_set]

# extract training data
train_questions = []
test_questions = []
for question in autocast_questions:
    if question['answer'] is None: # skipping questions without answer
        continue

    if question['id'] in test_ids: # skipping questions in the competition test set
        test_questions.append({
                'id':str(question['id']),
                'question':str(question['question']),
                'label': str(question['answer']),           # the label
                'answer':str(question['answer']),
                'background': str(question['background']),
                'publish_time':str(question['publish_time']),
                'close_time':str(question['close_time']),
                'tags':str(question['tags']),
                'answer': str(question['answer']),
                'choices': str(question['choices']),
            })
    else:
        # convert all useful training data fields into strings
        train_questions.append({
                'id':str(question['id']),
                'question':str(question['question']),
                'label': str(question['answer']),           # the label
                'answer':str(question['answer']),
                'background': str(question['background']),
                'publish_time':str(question['publish_time']),
                'close_time':str(question['close_time']),
                'tags':str(question['tags']),
                'answer': str(question['answer']),
                'choices': str(question['choices']),
            })

#2797
print(f"{len(train_questions)} training questions found")

print(f"{len(test_questions)} test questions found")

# Serializing json
train_object = json.dumps(train_questions, indent=4)
test_object = json.dumps(test_questions, indent=4)

# Writing to json
with open("datasets/training.json", "w") as outfile:
    outfile.write(train_object)

with open("datasets/testing.json", "w") as outfile:
    outfile.write(train_object)