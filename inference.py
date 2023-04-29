import argparse
import logging

import numpy as np
import torch
import os
import json
import pickle
import numpy as np

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
LENGTH = 50

autocast_questions = json.load(open('autocast_questions.json', encoding='utf-8')) # from the Autocast dataset
test_questions = json.load(open('autocast_competition_test_set.json', encoding='utf-8'))
test_ids = [q['id'] for q in test_questions]

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def ft_init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("output")
    model = GPT2LMHeadModel.from_pretrained("output")
    model.to(device)
    length = adjust_length_to_model(LENGTH, max_sequence_length=model.config.max_position_embeddings)
    return device, tokenizer, model, length

def ft_pred(device, tokenizer, model, length, question):
    prompt_text = question["question"] + " The answer is"
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)
    output_sequences = model.generate(
        input_ids = encoded_prompt,
        max_length = length,
        temperature = 1.0,
        top_k = 0,
        top_p = 0.9,
        repetition_penalty = 1.0,
        pad_token_id = 50256,
    )

    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    #text = text[: text.find(stop_token) if args.stop_token else None]
    
    gen_text = text.replace(prompt_text, '').strip().replace('\n', '')
    gen_ans = 'yes' if gen_text[0:3] == 'yes' else 'no'
    prob_ans = [1, 0] if gen_text[0:3] == 'yes' else [0, 1]
    return prob_ans

def ft_model(question):
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


device, tokenizer, model, length = ft_init()


preds = []
answers = []
qtypes = []
for question in autocast_questions[0:10]:
    if question['id'] in test_ids: # skipping questions in the competition test set
        continue
    if question['answer'] is None: # skipping questions without answer
        continue

    preds.append(calibrated_random_baseline_model(question))
    if question['qtype'] == 't/f':
        ft_ans = ft_pred(device, tokenizer, model, length, question)
        print("\nrandom ans:", calibrated_random_baseline_model(question))
        print("ft_ans:", ft_ans)
        
        ans_idx = 0 if question['answer'] == 'no' else 1
        ans = np.zeros(len(question['choices']))
        ans[ans_idx] = 1
        print("actual_answer:", ans)
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