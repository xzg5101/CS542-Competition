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
bool_model_path = "bool_fine_tuning"

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
LENGTH = 100

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
    tokenizer = GPT2Tokenizer.from_pretrained(bool_model_path)
    model = GPT2LMHeadModel.from_pretrained(bool_model_path)
    model.to(device)
    length = adjust_length_to_model(LENGTH, max_sequence_length=model.config.max_position_embeddings)
    return device, tokenizer, model, length

def ft_pred(device, tokenizer, model, length, question):
    prompt_text = question["question"] + " The answer is"
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)
    outputs = model.generate(
        input_ids = encoded_prompt,
        max_length = length,
        temperature = 1.0,
        top_k = 0,
        top_p = 0.9,
        repetition_penalty = 1.0,
        pad_token_id = 50256,
        #max_new_tokens=5,
        #num_beams=4,
        #num_return_sequences=4,
        return_dict_in_generate=True,
        output_scores=True,
    )
    print(outputs)
    output_seq = outputs.to_tuple()
    generated_sequence = output_seq[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    #text = text[: text.find(stop_token) if args.stop_token else None]
    
    #t_score = model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False)
    #print(t_score)
    
    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False)
    print(transition_scores)
    gen_text = text.replace(prompt_text, '').strip().replace('\n', '')
    gen_ans = 'yes' if gen_text[0:3] == 'yes' else 'no'

    if gen_text[0:3] == 'yes':
        gen_ans = 'yes'
    elif gen_text[0:2] == 'no':
        gen_ans = 'no'
    else:
        print('unknown answer:', gen_text)

    prob_ans = np.array([0.0, 1.0]) if gen_text[0:3] == 'yes' else np.array([1.0, 0.0])

    pred_idx = 1 if gen_text[0:3] == 'yes' else 0
    pred = np.ones(2)
    pred[pred_idx] += 8

    return gen_ans, pred / pred.sum()

def ft_model(question):
    if question['qtype'] == 't/f':
        ft_ans, ft_prob = ft_pred(device, tokenizer, model, length, question)
        return ft_prob
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
for idx, question in enumerate(test_questions):
    print(f"{idx}/{len(test_questions)}")
    preds.append(ft_model(question))

if not os.path.exists('submission'):
    os.makedirs('submission')

with open(os.path.join('submission', 'predictions.pkl'), 'wb') as f:
    pickle.dump(preds, f, protocol=2)
