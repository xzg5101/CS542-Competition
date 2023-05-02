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

from utils import clean_background
from utils import MAX_LENGTH, LENGTH, ANS_LEN

mc_model_path = 'choice_fine_tuning'


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
    tags = ''
    if len(question['tags']) > 0:
        tags = "This question is about " + ", ". join(question['tags']) + '. '
    bg = clean_background(str(question['background']))
    trimmed_choices = question['choices'][0:26]
    choices_prompt = [i + ":" + str(j) for i, j in zip(letters[0:len(trimmed_choices)], trimmed_choices)]
    prompt_text = tags + bg + question["question"] + ". You have following choices: " + '; '.join(choices_prompt) + ". The correct choice is"
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)
    outputs = model.generate(
        # temperature = 1.0,
        # top_k = 0,
        # top_p = 0.9,
        # repetition_penalty = 1.0,
        input_ids = encoded_prompt,
        max_length = encoded_prompt.shape[1] + ANS_LEN,
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

    confident = 0
    gen_ans = 'unknown'

    meet_answer = False

    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        if tokenizer.decode(tok).strip() in letters:
            print(f"|{tokenizer.decode(tok):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.4f}")
            confident = np.exp(score.numpy())
            gen_ans = tokenizer.decode(tok).strip()
            print(f"answer: {tokenizer.decode(tok):8s} | {confident:.4f}")
            meet_answer = True
            break
    
    if not meet_answer:
        print('unexpected answer:', text)
        even_ans = np.ones(len(question['choices']))
        return even_ans / even_ans.sum()
    
    print('final answer is ', gen_ans)
    pred = np.ones(len(question['choices']))
    pred_idx = letters.find(gen_ans)
    pred[pred_idx] += confident
    print( pred / pred.sum() )
    return gen_ans, pred / pred.sum()

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


device, tokenizer, model, length = ft_init()


preds = []
for idx, question in enumerate(test_questions):
    print(f"{idx}/{len(test_questions)}")
    preds.append(ft_model(question))

if not os.path.exists('submission'):
    os.makedirs('submission')

with open(os.path.join('submission', 'predictions.pkl'), 'wb') as f:
    pickle.dump(preds, f, protocol=2)
