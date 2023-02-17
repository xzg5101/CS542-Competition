import os
import json
import pickle
import numpy as np

autocast_questions = json.load(open('autocast_questions.json', encoding='utf-8')) # from the Autocast dataset
test_questions = json.load(open('autocast_competition_test_set.json', encoding='utf-8'))
test_ids = [q['id'] for q in test_questions]

def random_baseline_model(question):
    if question['qtype'] == 't/f':
        return np.random.random(size=2)
    elif question['qtype'] == 'mc':
        probs = np.random.random(size=len(question['choices']))
        return probs / probs.sum()
    elif question['qtype'] == 'num':
        return np.random.random()


def calibrated_random_baseline_model(question):
    if question['qtype'] == 't/f':
        pred_idx = np.argmax(np.random.random(size=2))
        pred = np.ones(2)
        pred[pred_idx] += 1e-5
        return pred / pred.sum()
    elif question['qtype'] == 'mc':
        pred_idx = np.argmax(np.random.random(size=len(question['choices'])))
        pred = np.ones(len(question['choices']))
        pred[pred_idx] += 1e-5
        return pred / pred.sum()
    elif question['qtype'] == 'num':
        return 0.5

def brier_score(probabilities, answer_probabilities):
    return ((probabilities - answer_probabilities) ** 2).sum() / 2


preds = []
answers = []
qtypes = []
for question in autocast_questions:
    if question['id'] in test_ids: # skipping questions in the competition test set
        continue
    if question['answer'] is None: # skipping questions without answer
        continue
    preds.append(calibrated_random_baseline_model(question))
    if question['qtype'] == 't/f':
        ans_idx = 0 if question['answer'] == 'no' else 1
        ans = np.zeros(len(question['choices']))
        ans[ans_idx] = 1
        qtypes.append('t/f')
    elif question['qtype'] == 'mc':
        ans_idx = ord(question['answer']) - ord('A')
        ans = np.zeros(len(question['choices']))
        ans[ans_idx] = 1
        qtypes.append('mc')
    elif question['qtype'] == 'num':
        ans = float(question['answer'])
        qtypes.append('num')
    answers.append(ans)


tf_results, mc_results, num_results = [],[],[]
for p, a, qtype in zip(preds, answers, qtypes):
    if qtype == 't/f':
        tf_results.append(brier_score(p, a))
    elif qtype == 'mc':
        mc_results.append(brier_score(p, a))
    else:
        num_results.append(np.abs(p - a))

print(f"T/F: {np.mean(tf_results)*100:.2f}, MCQ: {np.mean(mc_results)*100:.2f}, NUM: {np.mean(num_results)*100:.2f}")
print(f"Combined Metric: {(np.mean(tf_results) + np.mean(mc_results) + np.mean(num_results))*100:.2f}")

preds = []
for question in test_questions:
    preds.append(calibrated_random_baseline_model(question))

if not os.path.exists('submission'):
    os.makedirs('submission')

with open(os.path.join('submission', 'predictions.pkl'), 'wb') as f:
    pickle.dump(preds, f, protocol=2)