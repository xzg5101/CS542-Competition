import json


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
bool_labels = ['no', 'yes']

def is_letter(string):
    return len(string) == 1 and string in letters


# import data
autocast_questions = json.load(open('autocast_questions.json', encoding='utf-8')) # from the Autocast dataset
test_set = json.load(open('autocast_competition_test_set.json', encoding='utf-8'))
test_ids = [q['id'] for q in test_set]

# extract training data
train_questions = []
test_questions = []
labels = set()
for question in autocast_questions:
    if question['answer'] is None: # skipping questions without answer
        continue

    if is_float(question['answer']):
        label = str(question['answer'])
    elif is_letter(question['answer']):
        label = str(letters.index(question['answer']))
    elif question['answer'] in bool_labels:
        label = str(bool_labels.index(question['answer']))
    else:
        print('unknown label', question['answer'])

    q_obj = {
                'id':str(question['id']),
                'question':str(question['question']) + " The choices are " + str(question['choices']) + ", the correct one is " + str(question['answer']) + ".",
                'label': label,           # the label
                'answer':str(question['answer']),
                'background': str(question['background']),
                'publish_time':str(question['publish_time']),
                'close_time':str(question['close_time']),
                'tags':str(question['tags']),
                'answer': str(question['answer']),
                'choices': str(question['choices']),
            }
    labels.add(str(question['answer']))
    if question['id'] in test_ids: 
        test_questions.append(q_obj)
    else:
        train_questions.append(q_obj)

#2797
print(f"{len(train_questions)} training questions found")

print(f"{len(test_questions)} test questions found")

dataset = {
    'test': test_questions,
    'train': train_questions
}

converted_labels = []

for i in labels:
    if is_float(i):
        converted_labels.append(i)
    elif is_letter(i):
        converted_labels.append(str(letters.index(i)))
    elif i in bool_labels:
        converted_labels.append(str(bool_labels.index(i)))
    else:
        print('unknown label', i)

# Serializing json
train_object = json.dumps(train_questions, indent=4)
test_object = json.dumps(test_questions, indent=4)
data_object = json.dumps(dataset, indent=4)
label_object = json.dumps(list(labels), indent=4)
clabel_object = json.dumps(list(converted_labels), indent=4)
# Writing to json
with open("datasets/training.json", "w") as outfile:
    outfile.write(train_object)

with open("datasets/testing.json", "w") as outfile:
    outfile.write(train_object)

with open("datasets/dataset.json", "w") as outfile:
    outfile.write(data_object)

with open("datasets/labels.json", "w") as outfile:
    outfile.write(label_object)

with open("datasets/converted_labels.json", "w") as outfile:
    outfile.write(clabel_object)