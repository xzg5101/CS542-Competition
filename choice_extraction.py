import json


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

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

    if question['qtype'] == 'mc':
        label = letters.find(question['answer'])
    else:
        continue

    tags = ''
    if len(question['tags']) > 0:
        tags = "This question is about " + ", ". join(question['tags']) + '. '
    bg = str(question['background']).split('(http')[0].rstrip() + '. '

    trimmed_choices = question['choices'][0:26]
    choices_prompt = [i + ":" + str(j) for i, j in zip(letters[0:len(trimmed_choices)], trimmed_choices)]
    true_ans = question['answer']
    false_ans = []
    for choice in letters[0:len(trimmed_choices)]:
        if choice != true_ans:
            false_ans.append(choice)
    q_obj = {
                'id':str(question['id']),
                'question':tags + bg + str(question['question']) + ". You have following choices: " + '; '.join(choices_prompt) + ". The correct choice is " + true_ans + ".",
                # 'label': label,           # the label
                # 'answer':str(question['answer']),
                # 'background': str(question['background']),
                'publish_time':str(question['publish_time']),
                # 'close_time':str(question['close_time']),
                # 'tags':str(question['tags']),
                'answer': str(question['answer']),
                # 'choices': str(question['choices']),
            }
    q_n_obj = {
                'id':str(question['id']),
                'question':tags + bg + str(question['question']) + ". You have following choices: " + '; '.join(choices_prompt) + " The wrong choices are "  + ', '.join(false_ans),
                #'label': 0 if label == 1 else 1,           # the label
                'answer':', '.join(false_ans),
                #'background': str(question['background']).split('(http')[0].rstrip() + '.',
                'publish_time':str(question['publish_time']),
                #'close_time':str(question['close_time']),
                #'tags': "This question is about " + ", ". join(question['tags']),
                #'choices': str(question['choices']),
            }
    labels.add(str(question['answer']))
    if question['id'] in test_ids: 
        test_questions.append(q_obj)
    else:
        train_questions.append(q_obj)
        train_questions.append(q_n_obj)

#2797
print(f"{len(train_questions)} training questions found")

print(f"{len(test_questions)} test questions found")

dataset = {
    'test': test_questions,
    'train': train_questions
}



# Serializing json
train_object = json.dumps(train_questions, indent=4)
test_object = json.dumps(test_questions, indent=4)
data_object = json.dumps(dataset, indent=4)
# Writing to json
with open("datasets/choice_training.json", "w") as outfile:
    outfile.write(train_object)

with open("datasets/choice_testing.json", "w") as outfile:
    outfile.write(train_object)

with open("datasets/choice_dataset.json", "w") as outfile:
    outfile.write(data_object)
