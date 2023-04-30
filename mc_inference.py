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

mc_model_path = 'choice_fine_tuning'
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
LENGTH = 200

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

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
    tokenizer = GPT2Tokenizer.from_pretrained(mc_model_path)
    model = GPT2LMHeadModel.from_pretrained(mc_model_path)
    model.to(device)
    length = adjust_length_to_model(LENGTH, max_sequence_length=model.config.max_position_embeddings)
    return device, tokenizer, model, length

def ft_pred(device, tokenizer, model, length, question):
    trimmed_choices = question['choices'][0:26]
    choices_prompt = [i + ":" + str(j) for i, j in zip(letters[0:len(trimmed_choices)], trimmed_choices)]
    prompt_text = question["question"] + ". You have following choices: " + '; '.join(choices_prompt) + ". The correct choice is"
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
    # gen_ans = 'yes' if gen_text[0:3] == 'yes' else 'no'
    print("---------The output is " + gen_text)
    gen_ans = gen_text[0:1]

    if gen_text[0:1] in letters:
        gen_ans = gen_text[0:1]
    else:
        print('---------unknown answer:', gen_text)

    prob_ans = np.zeros([1, len(question['choices'])], dtype = np.float64)[0]
    prob_ans[letters.find(gen_ans)] = 1.
    print(prob_ans)

    '''pred_idx = letters.find(gen_ans)
    pred = np.ones(26)
    pred[pred_idx] += 8'''

    return gen_ans, prob_ans

def ft_model(question):
    if question['qtype'] == 't/f':
        pred_idx = np.argmax(np.random.random(size=len(question['choices'])))
        pred = np.ones(len(question['choices']))
        pred[pred_idx] += 1e-5
        return pred / pred.sum()
    elif question['qtype'] == 'mc':
        ft_ans, ft_prob = ft_pred(device, tokenizer, model, length, question)
        return ft_prob
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
for idx, question in enumerate(test_questions):
    print(f"{idx}/{len(test_questions)}")
    preds.append(ft_model(question))

if not os.path.exists('submission'):
    os.makedirs('submission')

with open(os.path.join('submission', 'predictions.pkl'), 'wb') as f:
    pickle.dump(preds, f, protocol=2)