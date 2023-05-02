import json
from utils import clean_background

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

    if question['answer'] in bool_labels:
        label = bool_labels.index(question['answer'])
    else:
        continue
    tags = ''
    if len(question['tags']) > 0:
        tags = "This question is about " + ", ". join(question['tags']) + '. '
    
    bg = clean_background(str(question['background']))

    q_obj = {
                'id':str(question['id']),
                'question':tags + bg + str(question['question']) + " The correct answer is " + str(question['answer']),
                'label': 1,           # the label
                'answer':str(question['answer']),
                #'background': str(question['background']).split('(http')[0].rstrip() + '.',
                'publish_time':str(question['publish_time']),
                #'close_time':str(question['close_time']),
                #'tags': "This question is about " + " ". join(question['tags']),
                #'choices': str(question['choices']),
            }
    if str(question['answer']) == 'yes':
        n_ans = 'no'
    else:
        n_ans = 'yes'
    q_n_obj = {
                'id':str(question['id']),
                'question':tags + bg + str(question['question']) + " The wrong answer is "  + n_ans,
                'label': 0,           # the label
                'answer':n_ans,
                #'background': str(question['background']).split('(http')[0].rstrip() + '.',
                'publish_time':str(question['publish_time']),
                #'close_time':str(question['close_time']),
                #'tags': "This question is about " + ", ". join(question['tags']),
                #'choices': str(question['choices']),
            }
    
    if question['id'] in test_ids: 
        test_questions.append(q_obj)
    else:
        train_questions.append(q_obj)
        #train_questions.append(q_n_obj)

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
with open("datasets/bool_training.json", "w") as outfile:
    outfile.write(train_object)

with open("datasets/bool_testing.json", "w") as outfile:
    outfile.write(train_object)

with open("datasets/bool_dataset.json", "w") as outfile:
    outfile.write(data_object)
