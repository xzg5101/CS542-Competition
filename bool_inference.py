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
bool_model_path = "bool_fine_tuning2"

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
LENGTH = 500

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
    tags = ''
    if len(question['tags']) > 0:
        tags = "This question is about " + ", ". join(question['tags']) + '. '
    bg = str(question['background']).split('(http')[0].rstrip() + '. '

    prompt_text = tags + bg + str(question['question']) + " The correct answer is"
    
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)
    outputs = model.generate(
        input_ids = encoded_prompt,
        max_length = length,
        #temperature = 1.0,
        #top_k = 0,
        #top_p = 0.9,
        #repetition_penalty = 1.0,
        pad_token_id = 50256,
        #max_new_tokens=5,
        #num_beams=4,
        #num_return_sequences=4,
        return_dict_in_generate=True,
        output_scores=True,
    )

    output_seq = outputs.sequences
    generated_sequence = output_seq[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    input_length = encoded_prompt.shape[1]
    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
    generated_tokens = outputs.sequences[:, input_length:]
    #first_token, first_score = generated_tokens[0][0], transition_scores[0][0]
    #print(f"first token | {first_token:5d} | {tokenizer.decode(first_token):8s} | {first_score.numpy():.4f} | {np.exp(first_score.numpy()):.2%}")

    #print(f"|{tokenizer.decode(bool_token):8s} | {bool_score.numpy():.4f} | {np.exp(bool_score.numpy()):.4f}")
    #confident = 0 #np.exp(first_score.numpy())
    gen_ans = 'no'
    meet_answer = False
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        #print(f"word: {tokenizer.decode(tok):8s} | {np.exp(score.numpy()):.4f}")
        if tokenizer.decode(tok).strip() in ['yes', 'no', 'positive', 'negative']:
            #print(f"first token | {first_token:5d} | {tokenizer.decode(first_token):8s} | {first_score.numpy():.4f} | {np.exp(first_score.numpy()):.2%}")
            print(f"|{tokenizer.decode(tok):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.4f}")
            
            confident = np.exp(score.numpy())
            gen_ans = tokenizer.decode(tok).strip()
            print(f"answer: {tokenizer.decode(tok):8s} | {np.exp(score.numpy()):.4f}")
            meet_answer = True
            break
    if gen_ans in ['yes', 'positive']:
        gen_ans = 'yes'
    else:
        gen_ans = 'no'
     
    #gen_text = text.replace(prompt_text, '').strip().replace('\n', '')
    if not meet_answer:
        print('unexpected answer:', text)
        y_pos = float('inf')
        n_pos = float('inf')
        if 'yes' in text:
            y_pos = text.find('yes')
        
        if 'positive' in text and text.find('positive') < y_pos:
            y_pos = text.find('positive')
        
        if 'no' in text:
            n_pos = text.find('no')
        
        if 'negative' in text and text.find('negative') < n_pos:
            n_pos = text.find('negative')
        
        if y_pos == float('inf') and n_pos == float('inf'):
            return np.array([0.5, 0.5])

        elif y_pos < n_pos:
            gen_ans = 'yes'
        else:
            gen_ans = 'no'


    #gen_ans = 'yes' if gen_text[0:3] == 'yes' else 'no'

    #if gen_text[0:3] == 'yes':
    #    gen_ans = 'yes'
    #elif gen_text[0:2] == 'no':
    #    gen_ans = 'no'
    #else:
    #    print('unknown answer:', gen_text)
    #    if 'yes' in gen_text[0:10]:
    #        gen_ans = 'yes'
    #    else:
    #        gen_ans = 'no'
    print('final answer is ', gen_ans)
    pred_idx = 1 if gen_ans == 'yes' else 0
    pred = np.ones(2)
    pred[pred_idx] += confident
    print( pred / pred.sum() )
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
    print(f"\n{idx}/{len(test_questions)}")
    preds.append(ft_model(question))

if not os.path.exists('submission'):
    os.makedirs('submission')

with open(os.path.join('submission', 'predictions.pkl'), 'wb') as f:
    pickle.dump(preds, f, protocol=2)
