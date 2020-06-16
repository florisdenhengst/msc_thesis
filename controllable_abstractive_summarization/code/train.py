import os
import re
import sys
import math
import time
import random
import statistics

import nltk
import spacy
import torch
import logging
import argparse
import coloredlogs

import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from collections import Counter
from rouge import Rouge
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator

from data_preprocess import anonymize_and_bpe_data
from artificial_data_preprocess import Synthetic
from model import ControllableSummarizer
from generate_sample import preprocess_text

from utils import prepare_batch, calculate_rouge, save_model, summarize_text, \
                    add_tokens_to_vocab, exclude_token, get_lead_3, extract_entities_to_prepend, \
                    count_pads, prepare_summaries, prepare_story_for_control_test

from synthetic import train_synth

from nltk.sentiment.vader import SentimentIntensityAnalyzer, SentiText

EMB_DIM = 340 # from paper 
HID_DIM = 512 # from paper
N_LAYERS = 8 # from paper
KERNEL_SIZE = 3 # from paper
DROPOUT_PROB = 0.2 # from paper 
NO_LEN_TOKENS = 10
BATCH_SIZE = 32

nltk.download('vader_lexicon')
logger = logging.getLogger('Training log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def prepare_batch(batch, txt_field, txt_nonseq_field, sent_end_inds, controls, reinforcement=False):
    summary_to_rouge, summary_to_pass = prepare_summaries(batch.summary, txt_field)
    story = batch.story
    lead_3 = get_lead_3(batch.story, txt_field, sent_end_inds)
    codes = []

    if 'entities' in controls:     
        ent_tensor = extract_entities_to_prepend(lead_3, summary_to_rouge, txt_field)  
        story = prepare_story_for_control_test(story, txt_field, control='entities', ent_tensor=ent_tensor)
        # return story, summary_to_rouge, summary_to_pass, lead_3

    if 'length' in controls:
        if reinforcement:
            codes = get_summary_length_codes(batch.summary, batch.length_tokens, reinforcement)    
        else:
            codes = ['<len' + str(int(len_ind)) + '>' for len_ind in batch.length_tokens]
        
        story = prepare_story_for_control_test(story, txt_field, control='length', control_codes=codes)

    if 'source' in controls:
        if reinforcement:
            codes = get_summary_source_codes(batch.summary, batch.source, txt_field, reinforcement)    
        else:
            codes = ['<' + txt_nonseq_field.vocab.itos[src_ind] + '>' for src_ind in batch.source]
        story = prepare_story_for_control_test(story, txt_field, control='source', control_codes=codes)

    if 'sentiment' in controls:
        codes = get_summary_sentiment_codes(batch.summary, txt_field, reinforcement)
        story = prepare_story_for_control_test(story, txt_field, control='sentiment', control_codes=codes)
        

    return story, summary_to_rouge, summary_to_pass, lead_3, codes

def calculate_rouge(summary_to_rouge, output_to_rouge, rouge, rouge_scores):
    
    try:
        temp_scores = rouge.get_scores(output_to_rouge, summary_to_rouge, avg=True)
    except RecursionError:
        recursion_count += 1
        temp_scores = rouge.get_scores(['a'], ['b'], avg=True)
    if rouge_scores is None:
        rouge_scores = temp_scores
    else: 
        rouge_scores = {key: Counter(rouge_scores[key]) + Counter(temp_scores[key]) for key in rouge_scores.keys()}
        for key in rouge_scores:
            if len(rouge_scores[key]) == 0:
                rouge_scores[key] = {'f': 0.0, 'p': 0.0, 'r': 0.0}
            else:
                rouge_scores[key] = dict(rouge_scores[key]) 
    return rouge_scores, temp_scores

def save_model(model, save_model_path, epoch, save_suffix):
    Path.mkdir(save_model_path, exist_ok=True)
    if Path.exists(Path(save_model_path, 'summarizer_epoch_' + str(epoch-1) + save_suffix + '.model')):
        logger.info('Removing model from previous epoch...')
        Path.unlink(Path(save_model_path, 'summarizer_epoch_' + str(epoch-1) + save_suffix + '.model'))
    logger.info(f'Saving model at epoch {epoch}.')
    torch.save(model.state_dict(), Path(save_model_path, 'summarizer_epoch_' + str(epoch) + save_suffix + '.model'))


def summarize_text(text, field, model, device, bpe_applied=True, desired_length=5, desired_source='cnn', summary=None):
    
    if not bpe_applied:
        with open(Path(data_path, 'cnn_dailymail.bpe'), 'r') as codes:
            bpencoder = BPEncoder(codes)

        def repl(match):
            replaced = match.group(0).replace('@@ ', '')
            return replaced

        pattern = '@{3} enti@{2} ty@{2} (\d+@@ )*\d+(?!@)'
        text = re.sub(pattern, repl, bpencoder.encode(text))

    with model.eval() and torch.no_grad():
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
        text = [tok.lower() for tok in nlp.tokenizer(text)]
        text = ['<sos>'] + text + ['<eos>']
        text = torch.tensor([field.vocab.stoi[token] for token in text]).unsqueeze(0).to(device)

        # lead_3 = get_lead_3(story, txt_field, sent_end_inds) 
        
        len_tensor = torch.tensor([txt_field.vocab.stoi['<len' + str(desired_length) + '>']]).unsqueeze(dim=1)
        src_tensor = torch.tensor([txt_field.vocab.stoi['<' + txt_nonseq_field.vocab.itos[desired_source] + '>']]).unsqueeze(dim=1)
        # ent_tensor = extract_entities_to_prepend(lead_3, summary_to_rouge, txt_field)

        story = torch.cat((len_tensor, src_tensor, story), dim=1) #ent_tensor, len_tensor, src_tensor, story), dim=1)
        output = model.inference(story, 'sos', 'eos')
        logger.info(f'Summary: {[txt_field.vocab.itos[out] for out in output]}')
        if summary is not None:
            logger.info(f'Summary: {summary}')


def add_tokens_to_vocab(txt_field, tokens):
    for token in tokens:
        if token not in txt_field.vocab.stoi:
            txt_field.vocab.stoi[token] = len(txt_field.vocab.stoi)
            txt_field.vocab.itos.append(token)
    return txt_field


def exclude_token(summaries, index, for_rouge=False):
    
    new_summaries = []
    for summ in summaries:
        cutoff = (summ == index).nonzero()
        if len(cutoff) > 1:
            cutoff = cutoff[0]
            new_summaries.append(summ[0:cutoff])
        elif len(cutoff) == 1:
            new_summaries.append(torch.cat((summ[0:cutoff], summ[cutoff+1:])))
        else:
            new_summaries.append(summ)
    
    if for_rouge:
        return new_summaries
    else:
        return torch.stack(new_summaries)

def get_lead_3(story, txt_field, sent_end_inds):    
    lead_3 = []
    for single in story:
        sents, ends = [], 0
        for ind in single:
            if ends < 3:
                if ind == txt_field.vocab.stoi['<eos>']:
                    break
                else:                    
                    sents.append(txt_field.vocab.itos[ind])
            else:
                break
            if ind in sent_end_inds:
                ends += 1 
        lead_3.append(' '.join(sents))
    return lead_3

def extract_entities_to_prepend(lead_3, summary_to_rouge, txt_field):
    lead_3_entities = [re.findall(r"@entity\d+", lead) for lead in lead_3]
    sum_entities = [re.findall(r"@entity\d+", summ) for summ in summary_to_rouge]

    entities_to_prepend = []
    entities_lens = []

    assert len(sum_entities) == len(lead_3_entities)
    for no in range(len(sum_entities)):
        ents = set([item for item in sum_entities[no] if item not in lead_3_entities[no]])
        ent_inds = [txt_field.vocab.stoi[item] for item in ents]        
        entities_lens.append(len(ent_inds))
        entities_to_prepend.append(ent_inds)
    
    assert len(entities_to_prepend) == len(lead_3_entities)
    max_len = max(entities_lens)

    for no in range(len(entities_to_prepend)):
        while len(entities_to_prepend[no]) != max_len:
            entities_to_prepend[no].insert(0, txt_field.vocab.stoi[txt_field.pad_token])
        entities_to_prepend[no] = torch.tensor(entities_to_prepend[no])

    return torch.stack(entities_to_prepend)

def count_pads(train_iter, padding_idx, to_count=True):
    stories_len = []
    summaries_len = []
    st_all_tokens = 0
    sm_all_tokens = 0
    st_pads = 0
    sm_pads = 0
    for batch in train_iter:
        stories_len.append(batch.story.shape[1])
        summaries_len.append(batch.summary.shape[1])
        if to_count:
            st_all_tokens += batch.story.shape[0] * batch.story.shape[1] 
            sm_all_tokens += batch.summary.shape[0] * batch.summary.shape[1] 
            st_pads += sum([sum([1 for ind in st if ind==padding_idx]) for st in batch.story])
            sm_pads += sum([sum([1 for ind in st if ind==padding_idx]) for st in batch.summary])
    if to_count:
        logger.info(f'In stories, pads are {100*st_pads/st_all_tokens} of all tokens.')
        logger.info(f'In summaries, pads are {100*sm_pads/sm_all_tokens} of all tokens.')

    logger.info(f'Maximum length of article: {max(stories_len)}.')

    stories_len.sort(reverse=True)
    logger.info(stories_len[0:5])
    logger.info(f'Maximum length of summary: {max(summaries_len)}.')
    max_len = max([max(stories_len), max(summaries_len)])+10

    if max_len > 1000:
        sys.setrecursionlimit(max_len + 10)
        max_len = 1000
    return max_len



def prepare_summaries(summary, txt_field, output=False):

    summary_to_pass = exclude_token(summary, txt_field.vocab.stoi['<eos>'], for_rouge=output)
    summary_to_rouge = exclude_token(summary_to_pass, txt_field.vocab.stoi['<sos>'], for_rouge=True)
    summary_to_rouge = exclude_token(summary_to_rouge, txt_field.vocab.stoi[txt_field.pad_token], for_rouge=True)
    summary_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in summary]) for summary in summary_to_rouge]
    # if get_length:
        # lengths = [len(summary) for summary in summary_to_rouge]
        # return summary_to_rouge, summary_to_pass, lengths
    # else:
    return summary_to_rouge, summary_to_pass


def prepare_story_for_control_test(stories, txt_field, control, control_codes=None, ent_tensor=None):
    if control != 'entities':
        ctrl_tensor = torch.tensor([txt_field.vocab.stoi[code] for code in control_codes]).unsqueeze(dim=1)
    else:
        ctrl_tensor = ent_tensor
    story = torch.cat((ctrl_tensor, stories), dim=1)
    return story



def test_on_control(model, batch, txt_field, control, control_tokens, device):
    if control == 'length':
        native_controls, flex_controls, control_evl_fn = test_on_length(batch, txt_field, control_tokens)
    elif control == 'sentiment':
        native_controls, flex_controls, control_evl_fn = test_on_sentiment(batch, txt_field, control_tokens[0], control_tokens[1])

    # Inference on input w/o control code
    output = model.inference(batch.story.to(device), txt_field.vocab.stoi['<sos>'], txt_field.vocab.stoi['<eos>'])
    output_to_rouge, _ = prepare_summaries(torch.tensor(output), txt_field, output=True)

    # Keep track of no control inference and control performanec
    no_control_output = output_to_rouge
    no_control_results = control_evl_fn(output_to_rouge, batch.summary, story, txt_field)

    # Inference on input over all control codes
    flex_results = []
    flex_outputs = []
    native_output = [None for i in range(len(batch.story))]
    native_results = [None for i in range(len(batch.story))]

    for flex in flex_controls:

        story = prepare_story_for_control_test(batch.story, txt_field, control=control, control_codes=flex)
        output = model.inference(story.to(device), txt_field.vocab.stoi['<sos>'], txt_field.vocab.stoi['<eos>'])
        output_to_rouge, _ = prepare_summaries(torch.tensor(output), txt_field, output=True)
        
        flex_outputs.append(output_to_rouge)
        # Save results on control performance, per story for all control codes
        flex_results.append(control_evl_fn(output_to_rouge, batch.summary, story, txt_field))

        # Track if the flex control is native for one of the stories in the batch
        # If so, add generated summary and control performance to the respective lists
        native_index = [i for i, x in enumerate(native_controls) if x == flex[0]]
        if len(native_index) > 0:
            for ind in native_index:
                native_output[ind] = output_to_rouge[ind]
                native_results[ind] = {kkey: flex_results[-1][kkey][ind] for kkey in flex_results[-1].keys()}
    
    outputs = (no_control_output, native_output, flex_outputs)
    results = (no_control_results, native_results, flex_results)
    
    control_performance = [sum(flex['output'])/len(flex['output']) for flex in flex_results]
    
    return outputs, control_performance, results


def test_on_length(batch, txt_field, len_tokens):
    native_controls = ['<len' + str(int(len_ind)) + '>' for len_ind in batch.length_tokens]
    flex_controls = []
    for token in len_tokens:
        flex_controls.append([token for i in range(len(batch.summary))])
    return native_controls, flex_controls, evaluate_on_length

def test_on_sentiment(batch, txt_field, sentiment_tokens, sentiment_codes):
    native_controls = sentiment_codes
    flex_controls = []
    for token in sentiment_tokens:
        flex_controls.append([token for i in range(len(batch.summary))])
    return native_controls, flex_controls, evaluate_on_sentiment


def evaluate_on_length(output_to_rouge, summary, story, txt_field):
    summary_to_rouge, _ = prepare_summaries(summary, txt_field)
    length_output = [len(out.split(' ')) for out in output_to_rouge]
    length_summary = [len(summary.split(' ')) for summary in summary_to_rouge]

    story_to_rouge, _ = prepare_summaries(story, txt_field)
    length_story = [len(story.split(' ')) for story in story_to_rouge]

    return {'output': length_output, 'summary': length_summary, 'story': length_story}

def evaluate_on_sentiment(output_to_rouge, summary, story, txt_field):
    sid = SentimentIntensityAnalyzer()

    summary_to_rouge, _ = prepare_summaries(summary, txt_field)
    
    sentiment_output = [sid.polarity_scores(out)['compound'] for out in output_to_rouge]
    sentiment_summary = [sid.polarity_scores(summary)['compound'] for summary in summary_to_rouge]

    story_to_rouge, _ = prepare_summaries(story, txt_field)
    sentiment_story = [sid.polarity_scores(story)['compound'] for story in story_to_rouge]

    return {'output': sentiment_output, 'summary': sentiment_summary, 'story': sentiment_story}


def get_summary_sentiment_codes(summaries, txt_field, reinforcement):
    sid = SentimentIntensityAnalyzer()
    remove_tokens = ['<sos>', '<eos>', '<pad>']
    sentiment_codes = []
    for summary in summaries:
        if args.only_pos:
            sentiment_code = '<pos>'
        if args.only_neg:
            sentiment_code = '<neg>'
        else:
            tmp = []
            for ind in summary:
                if txt_field.vocab.itos[ind] not in remove_tokens:
                    tmp.append(txt_field.vocab.itos[ind])
            summary = ' '.join(tmp).replace('@@ ', '')

            sentiment = sid.polarity_scores(summary)['compound']
            if reinforcement:
                coin = random.random()
                if sentiment > 0.05:
                    sentiment_code = '<neg>' if coin > 0.5 else '<neu>'
                elif sentiment < 0.05:
                    sentiment_code = '<pos>' if coin > 0.5 else '<neu>'
                else:
                    sentiment_code = '<pos>' if coin > 0.5 else '<neg>'
            else:
                if sentiment > 0.05:
                    sentiment_code = '<pos>'
                elif sentiment < -0.05:
                    sentiment_code = '<neg>'
                else:
                    sentiment_code = '<neu>'
        sentiment_codes.append(sentiment_code)
            
    return sentiment_codes

def obtain_reward_sentiment(output_to_rouge, baseline_to_rouge, sentiment_codes, do_rouge=False, summary_to_rouge=None, rouge=None):
    sid = SentimentIntensityAnalyzer()
    rewards = []
    sentiments = []
    baseline_sentiments = []
    if do_rouge:
        temp_scores = rouge.get_scores(output_to_rouge, summary_to_rouge, avg=False)
        output_rouge = [score['rouge-l']['f'] for score in temp_scores]
        temp_scores = rouge.get_scores(baseline_to_rouge, summary_to_rouge, avg=False)
        baseline_rouge = [score['rouge-l']['f'] for score in temp_scores]
    else:
        output_rouge = [0 for i in range(len(output_to_rouge))] 
        baseline_rouge = [0 for i in range(len(output_to_rouge))] 
    for sample, baseline, sentiment, s_rouge, b_rouge in zip(output_to_rouge, baseline_to_rouge, sentiment_codes, output_rouge, baseline_rouge):
        sample = sample.replace('@@ ', '')
        r_sample = sid.polarity_scores(sample)['compound']
        sentiments.append(r_sample)

        baseline = sample.replace('@@ ', '')
        r_baseline = sid.polarity_scores(baseline)['compound']
        baseline_sentiments.append(r_baseline)

        if sentiment == '<pos>':
            # loss = -CE
            # loss_rl = (baseline - sample) * CE
            # loss_in_pytorch = (sample - baseline) * loss
            # minimize
            rewards.append((r_sample - r_baseline) + (s_rouge - b_rouge))
        elif sentiment == '<neg>':
            rewards.append((r_baseline - r_sample) + (s_rouge - b_rouge))
        elif sentiment == '<neu>':
            rewards.append(abs(r_sample - r_baseline) + (s_rouge - b_rouge))
    return torch.tensor(rewards), sentiments, baseline_sentiments



def get_summary_length_codes(summaries, lengths, txt_field):
    length_codes = []
    for summary, length_code in zip(summaries, lengths):
        coin = random.random()
        if int(length_code) < 4:
            # This is the short summary category
            code = '<medium>' if coin > 0.5 else '<long>'
        elif int(length_code) < 8:
            code = '<short>' if coin > 0.5 else '<long>'
        else:
            code = '<medium>' if coin > 0.5 else '<short>'
        length_codes.append(code)
    return length_codes  


def obtain_reward_length(output_to_rouge, baseline_to_rouge, length_codes, do_rouge=False, summary_to_rouge=None, rouge=None):
    rewards = []
    lengths = []
    baseline_lengths = []
    if do_rouge:
        temp_scores = rouge.get_scores(output_to_rouge, summary_to_rouge, avg=False)
        output_rouge = [score['rouge-l']['f'] for score in temp_scores]
        temp_scores = rouge.get_scores(baseline_to_rouge, summary_to_rouge, avg=False)
        baseline_rouge = [score['rouge-l']['f'] for score in temp_scores]
    else:
        output_rouge = [0 for i in range(len(output_to_rouge))] 
        baseline_rouge = [0 for i in range(len(output_to_rouge))] 
    for sample, baseline, length_code, s_rouge, b_rouge in zip(output_to_rouge, baseline_to_rouge, length_codes, output_rouge, baseline_rouge):
        sample = sample.replace('@@ ', '')
        r_sample = len(sample.split(' '))
        lengths.append(r_sample)

        # r_baseline = 
        baseline = baseline.replace('@@ ', '')
        r_baseline = len(baseline.split(' '))
        baseline_lengths.append(r_baseline)

        if length_code == '<long>':
            # loss = -CE
            # loss_rl = (baseline - sample) * CE
            # loss_in_pytorch = (sample - baseline) * loss
            # minimize
            rewards.append((r_sample - r_baseline) + (s_rouge - b_rouge))
        elif length_code == '<short>':
            rewards.append((r_baseline - r_sample) + (s_rouge - b_rouge))
        elif length_code == '<medium>':
            rewards.append(abs(r_sample - r_baseline) + (s_rouge - b_rouge))
    return torch.tensor(rewards), lengths, baseline_lengths



def get_summary_length_codes(summaries, lengths, txt_field):
    length_codes = []
    for summary, length_code in zip(summaries, lengths):
        coin = random.random()
        if int(length_code) < 4:
            # This is the short summary category
            code = '<medium>' if coin > 0.5 else '<long>'
        elif int(length_code) < 8:
            code = '<short>' if coin > 0.5 else '<long>'
        else:
            code = '<medium>' if coin > 0.5 else '<short>'
        length_codes.append(code)
    return length_codes  

# def get_summary_source_codes(summaries, sources, txt_field):
#     source_codes = []
#     for summary, source in zip(summaries, sources):
#         if source == '<cnn>':
#             code = '<dailymail>'
#         elif source == '<dailymail>':
#             code == '<cnn>'
#         else:
#             coin = random.random()
#             code = 'dm' if coin > 0.5 else 'cnn'
#         source_codes.append(code)
#     return source_codes    

# def obtain_reward_source(output_to_rouge, baseline_to_rouge, source_codes, do_rouge=False, summary_to_rouge=None, rouge=None):
#     rewards = []
#     overlaps = []
#     baseline_overlaps = []
#     if do_rouge:
#         temp_scores = rouge.get_scores(output_to_rouge, summary_to_rouge, avg=False)
#         output_rouge = [score['rouge-l']['f'] for score in temp_scores]
#         temp_scores = rouge.get_scores(baseline_to_rouge, summary_to_rouge, avg=False)
#         baseline_rouge = [score['rouge-l']['f'] for score in temp_scores]
#     else:
#         output_rouge = [0 for i in range(len(output_to_rouge))] 
#         baseline_rouge = [0 for i in range(len(output_to_rouge))] 
#     for sample, baseline, source_code, s_rouge, b_rouge in zip(output_to_rouge, baseline_to_rouge, source_codes, output_rouge, baseline_rouge):
#         # r_sample = 
#         overlaps.append(r_sample)

#         # r_baseline = 
#         baseline_overlaps.append(r_baseline)

#         if source_code == '<pos>':
#             # loss = -CE
#             # loss_rl = (baseline - sample) * CE
#             # loss_in_pytorch = (sample - baseline) * loss
#             # minimize
#             rewards.append((r_sample - r_baseline) + (s_rouge - b_rouge))
#         elif source_code == '<neg>':
#             rewards.append((r_baseline - r_sample) + (s_rouge - b_rouge))
#         elif source_code == '<neu>':
#             rewards.append(abs(r_sample - r_baseline) + (s_rouge - b_rouge))
#     return torch.tensor(rewards), overlaps, baseline_overlaps



def train():
    random.seed(args.seed)
    st = random.getstate()

    sys.setrecursionlimit(1500) # handles some issues with Rouge
    
    data_path = Path(Path.cwd(), 'data/')
    save_model_path = Path(Path.cwd(), args.save_model_to)

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

    txt_field = Field(tokenize=lambda x: [tok.text for tok in nlp.tokenizer(x)], 
                        init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
    num_field = Field(sequential=False, use_vocab=False)
    txt_nonseq_field = Field(sequential=False, use_vocab=True)

    train_fields = [('id', txt_nonseq_field), ('story', txt_field), ('length_tokens', num_field),
                    ('length_sentences', num_field), ('source', txt_nonseq_field), 
                    ('entities', None), ('summary', txt_field)]
    
    logger.info(f'Training on {device} ')
    logger.info('Started loading data...')

    start = time.time()
    if args.full_train:
        csv_path = Path(data_path, 'cnn_dailymail.csv')
    else:
        csv_path = Path(data_path, 'cnn_dailymail_test_purposes.csv')
    if not Path.exists(csv_path):
        logger.info(f'creating pre-processed data...')
        anonymize_and_bpe_data(data_path=data_path, sources=['cnn'], cut_off_length=400)
    train_data, val_data, test_data = TabularDataset(path=csv_path, format='csv', 
                                skip_header=True, fields=train_fields).split(split_ratio=[0.922, 0.043, 0.035], random_state=st)
    if args.test:
        train_data, val_data = [], []
        test_iter = BucketIterator(dataset=test_data, batch_size=args.batch_size, 
            sort_key=lambda x:(len(x.story), len(x.summary)), shuffle=True, train=False)
        txt_field.build_vocab(test_data)
        txt_nonseq_field.build_vocab(test_data)
    else:        
        test_data = []
        train_iter = BucketIterator(dataset=train_data, batch_size=args.batch_size, 
            sort_key=lambda x:(len(x.story), len(x.summary)), shuffle=True, train=True)
        val_iter = BucketIterator(dataset=val_data, batch_size=args.batch_size, 
            sort_key=lambda x:(len(x.story), len(x.summary)), shuffle=True, train=False)
        if args.epoch != 0:
            txt_field.build_vocab()
            txt_nonseq_field.build_vocab()
        else:
            txt_field.build_vocab(train_data, val_data)
            txt_nonseq_field.build_vocab(train_data, val_data)

        #Check for consistency of random seed
        sample = next(iter(train_iter))
        logger.info(f'1st train article id is {sample.id}')
        sample = next(iter(val_iter))
        logger.info(f'1st val article id is {sample.id}')
        _ = count_pads(train_iter, txt_field.vocab.stoi[txt_field.pad_token], True)
        
    
    logger.info(f'{len(train_data)} train samples, {len(val_data)} validation samples, {len(test_data)} test samples...', )

    end = time.time()
    logger.info(f'finished in {end-start} seconds.')
    logger.info('Started building vocabs...')
    start = time.time()    
    
    if Path.exists(Path(save_model_path, 'vocab_stoi.pkl')):
        with open(Path(save_model_path, 'vocab_stoi.pkl'), 'rb') as file:
            txt_field.vocab.stoi = pickle.load(file)
        with open(Path(save_model_path, 'vocab_itos.pkl'), 'rb') as file:
            txt_field.vocab.itos = pickle.load(file)
    else:
        with open(Path(save_model_path, 'vocab_stoi.pkl'), 'wb') as file:
            pickle.dump(txt_field.vocab.stoi, file)
        with open(Path(save_model_path, 'vocab_itos.pkl'), 'wb') as file:
            pickle.dump(txt_field.vocab.itos, file)    

    logger.info(f'{len(txt_field.vocab.stoi)} items in vocabulary before adding control codes.')
    
    len_tokens = ['<len' + str(i+1) + '>' for i in range(args.no_len_tokens)]
    len_tokens_rl = ['<long>', '<short>', '<medium>']
    txt_field = add_tokens_to_vocab(txt_field, len_tokens)
    txt_field = add_tokens_to_vocab(txt_field, len_tokens_rl)
    source_tokens = ['<cnn>', '<dailymail>']
    source_tokens_rl = ['<cnnrl>', '<dailymailrl>']
    txt_field = add_tokens_to_vocab(txt_field, source_tokens)
    txt_field = add_tokens_to_vocab(txt_field, source_tokens_rl)
    sentiment_tokens = ['<pos>', '<neg>', '<neu>']
    txt_field = add_tokens_to_vocab(txt_field, sentiment_tokens)

    controls = []
    if args.controls == 0:
        tmp_controls = ['1', '2', '3', '4']
    else:
        tmp_controls = [ctrl for ctrl in str(args.controls)]

    if '1' in tmp_controls:
        controls.append('length')
    if '2' in tmp_controls:
        controls.append('source')
    if '3' in tmp_controls:
        controls.append('entities')
    if '4' in tmp_controls:
        controls.append('sentiment')

    if 'sentiment' in controls:
        control_tokens = sentiment_tokens
    elif 'length' in controls:
        control_tokens = len_tokens_rl if args.reinforcement else len_tokens
    elif 'source' in controls:
        control_tokens = source_tokens_rl if args.reinforcement else source_tokens

    logger.info(f'{len(txt_field.vocab.stoi)} items in vocabulary after adding control codes.')

    padding_idx = txt_field.vocab.stoi[txt_field.pad_token]
    sos_idx = txt_field.vocab.stoi['<sos>']
    eos_idx = txt_field.vocab.stoi['<eos>']
    input_dim = len(txt_field.vocab.itos)
    output_dim = len(txt_field.vocab.itos)
    
    end = time.time()
    logger.info(f'finished in {end-start} seconds.')

    # if args.test:
    max_len = 1000
    # else: 
    save_suffix = ''
    if args.reinforcement:
        save_suffix += '_rl'
    if args.ml_reinforcement:
        save_suffix += '_ml'
    if args.rouge_scaling:
        save_suffix += '_rouge'
    if not args.only_pos:
        if args.only_neg:
            save_suffix += '_onlyneg'
        else:
            save_suffix += '_all'
    

    batch = next(iter(train_iter))
    logger.info(f'Batch length tokens: {batch.length_tokens}')
    logger.info(f'Batch source tokens: {batch.source}')
            
    logger.info(f'Initializing model with:') 
    logger.info(f'Input dim: {input_dim}, output dim: {output_dim}, emb dim: {args.emb_dim} hid dim: {args.hid_dim}, {args.n_layers} layers, {args.kernel_size}x1 kernel, {args.dropout_prob} dropout, sharing weights: {args.share_weights}, maximum length: {max_len}.')

    model = ControllableSummarizer(input_dim=input_dim, output_dim=output_dim, emb_dim=args.emb_dim, 
                                    hid_dim=args.hid_dim, n_layers=args.n_layers, kernel_size=args.kernel_size, 
                                    dropout_prob=args.dropout_prob, device=device, padding_idx=padding_idx, 
                                    share_weights=args.share_weights, max_length=max_len, self_attention=int(args.self_attention)).to(device)
    if args.test:
        model.load_state_dict(torch.load(Path(save_model_path, 'summarizer.model')))
        model.eval()
        metrics = {'test_loss':[], 'test_rouge':[]}
    else:
        if Path.exists(Path(save_model_path, 'summarizer_epoch_' + str(args.epoch) + save_suffix + '.model')):
            model.load_state_dict(torch.load(Path(save_model_path, 'summarizer_epoch_' + str(args.epoch) + save_suffix + '.model')))
        elif Path.exists(Path(save_model_path, 'summarizer.model')):
            model.load_state_dict(torch.load(Path(save_model_path, 'summarizer.model')))
        logger.info(f'Shape of word embeddings: {model.tok_embedding.weight.shape}')
        logger.info(f'Shape of positional embeddings: {model.pos_embedding.weight.shape}')
            
            
        epoch = args.epoch
        metrics = {'train_loss':[], 'train_rouge':[], 'val_loss':[], 'val_rouge':[]}
        if Path.exists(Path(save_model_path, 'metrics_epoch_' + str(args.epoch) + save_suffix + '.pkl')):
            with open(Path(save_model_path, 'metrics_epoch_' + str(args.epoch) + save_suffix + '.pkl'), 'rb') as file:
                metrics  = pickle.load(file)
        elif Path.exists(Path(save_model_path, 'metrics_epoch_' + str(args.epoch) + '.pkl')):
            with open(Path(save_model_path, 'metrics_epoch_' + str(args.epoch) + '.pkl'), 'rb') as file:
                metrics  = pickle.load(file)

        control_performance = {'train': {'performance': [], 'count': []},
                                'val': {'performance': [], 'count': []},
                                'baseline': []}
        if Path.exists(Path(save_model_path, 'control_epoch_' + str(args.epoch) + save_suffix + '.pkl')):
            with open(Path(save_model_path, 'control_epoch_' + str(args.epoch) + save_suffix + '.pkl'), 'rb') as file:
                control_performance = pickle.load(file)


        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        no_params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info(f'{no_params} trainable parameters in the model.')
        
        if args.adam:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, nesterov=True)
        if args.reinforcement:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=0)
    crossentropy = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='none')
    if args.ml_reinforcement:
        gamma = args.gamma
    
    rouge = Rouge()
    sent_end_tokens = ['.', '!', '?']
    sent_end_inds = [txt_field.vocab.stoi[token] for token in sent_end_tokens]
    recursion_count = 0
    stop_condition = True    
    

    if args.timing and args.reinforcement and not args.full_train:
        timings = {'no_grad': [], 'grad': [], 'teacherforced': []}

        with model.eval() and torch.no_grad():
            for batch in train_iter:
                start = time.time()

                story, summary_to_rouge, summary_to_pass, lead_3, sentiment_codes = prepare_batch(batch, txt_field, txt_nonseq_field, sent_end_inds, controls, reinforcement=args.reinforcement)
                sample_output, output_tokens, baseline_tokens = model.rl_inference(story.to(device), sos_idx, eos_idx)
                
                baseline_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in summ]) for summ in baseline_tokens]
                output_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in summ]) for summ in output_tokens]
                
                sample_output = sample_output.contiguous().view(-1, sample_output.shape[-1])
                sample_to_loss = output_tokens[:,1:].contiguous().view(-1)

                loss = crossentropy(sample_output, sample_to_loss).contiguous().view(output_tokens.shape[0], -1)
                rewards, sentiments, _ = obtain_reward_sentiment(output_to_rouge, baseline_to_rouge, sentiment_codes)
                rewards = rewards.to(device)
                loss = torch.mul(rewards.unsqueeze(1), loss)
                loss = loss.mean()

                end = time.time()
                timings['no_grad'].append(end-start)

        logger.info(f'Without gradients, {sum(timings["no_grad"])} seconds taken to go over {len(train_iter)} batches of size {args.batch_size}')
        logger.info(f'That is {sum(timings["no_grad"]) / len(timings["no_grad"])} seconds per batch on average. Standard deviation: {statistics.stdev(timings["no_grad"])}')
        logger.info(f'Minimum time taken: {min(timings["no_grad"])} seconds.')
        logger.info(f'Maximum time taken: {max(timings["no_grad"])} seconds.')
        for batch in train_iter:
                start = time.time()

                story, summary_to_rouge, summary_to_pass, lead_3, sentiment_codes = prepare_batch(batch, txt_field, txt_nonseq_field, sent_end_inds, controls, reinforcement=args.reinforcement)
                sample_output, output_tokens, baseline_tokens = model.rl_inference(story.to(device), sos_idx, eos_idx)
                
                baseline_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in summ]) for summ in baseline_tokens]
                output_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in summ]) for summ in output_tokens]
                
                sample_output = sample_output.contiguous().view(-1, sample_output.shape[-1])
                sample_to_loss = output_tokens[:,1:].contiguous().view(-1)

                loss = crossentropy(sample_output, sample_to_loss).contiguous().view(output_tokens.shape[0], -1)
                rewards, sentiments, _ = obtain_reward_sentiment(output_to_rouge, baseline_to_rouge, sentiment_codes)
                rewards = rewards.to(device)
                loss = torch.mul(rewards.unsqueeze(1), loss)
                loss = loss.mean()

                end = time.time()
                timings['grad'].append(end-start)
        logger.info(f'Without gradients, {sum(timings["grad"])} seconds taken to go over {len(train_iter)} batches of size {args.batch_size}')
        logger.info(f'That is {sum(timings["grad"]) / len(timings["grad"])} seconds per batch on average. Standard deviation: {statistics.stdev(timings["grad"])}')
        logger.info(f'Minimum time taken: {min(timings["grad"])} seconds.')
        logger.info(f'Maximum time taken: {max(timings["grad"])} seconds.')

        for batch in train_iter:
                start = time.time()

                story, summary_to_rouge, summary_to_pass, lead_3, sentiment_codes = prepare_batch(batch, txt_field, txt_nonseq_field, sent_end_inds, controls, reinforcement=args.reinforcement)
                output, _ = model(story.to(device), summary_to_pass.to(device)) # second output is attention 
                output_tokens = torch.argmax(output, dim=2)
                summary = batch.summary[:,1:].contiguous().view(-1)
                output = output.contiguous().view(-1, output.shape[-1])
                loss = crossentropy(output, summary.to(device)).mean()

                end = time.time()
                timings['teacherforced'].append(end-start)
        logger.info(f'Without gradients, {sum(timings["teacherforced"])} seconds taken to go over {len(train_iter)} batches of size {args.batch_size}')
        logger.info(f'That is {sum(timings["teacherforced"]) / len(timings["teacherforced"])} seconds per batch on average. Standard deviation: {statistics.stdev(timings["teacherforced"])}')
        logger.info(f'Minimum time taken: {min(timings["teacherforced"])} seconds.')
        logger.info(f'Maximum time taken: {max(timings["teacherforced"])} seconds.')




    elif args.generate:
        text_paths = os.listdir(Path(Path.cwd(), 'sample_docs/'))
        batch = []
        logger.info(text_paths)
        for no, text_path in enumerate(text_paths):
            text = preprocess_text(text_path)        
            batch.extend(text)
            if len(batch) == 2 or no == len(text_paths) - 1:
                batch = txt_field.process(batch)
                print(f'How many OOV tokens per text: {[sum([oov == 0 for oov in txt]) for txt in batch]}')
                print(f'{batch.shape} shape')
                output = model.inference(batch.to(device), txt_field.vocab.stoi['<sos>'], txt_field.vocab.stoi['<eos>'])
                summaries, _ = prepare_summaries(torch.tensor(output), txt_field, output=False)
                logger.info(summaries)
                batch = []
        

    elif args.test:
        test_rouge = None
        no_control_rouge = None
        batch_count = 0

        control_results = []
        control_performance = [0 for i in range(len(control_tokens))]
        rouge_for_all = [None for i in range(len(control_tokens))]
        

        with model.eval() and torch.no_grad():
            for batch in test_iter:
                batch_count += 1
                
                if 'sentiment' in controls:
                    story, summary_to_rouge, summary_to_pass, lead_3, sentiment_codes = prepare_batch(batch, txt_field, txt_nonseq_field, sent_end_inds, controls, reinforcement=args.reinforcement)
                    outputs, batch_control_performance, results = test_on_control(model, batch, txt_field, controls[0], (sentiment_tokens, sentiment_codes), device)
                elif 'length' in controls:
                    story, summary_to_rouge, summary_to_pass, lead_3, length_codes = prepare_batch(batch, txt_field, txt_nonseq_field, sent_end_inds, controls, reinforcement=args.reinforcement)
                    outputs, batch_control_performance, results = test_on_control(model, batch, txt_field, controls[0], len_tokens, device)
                    

                start = time.time()
                
                
                end = time.time()
                logger.info(f'finished one control test in {end-start} seconds.')

                control_performance = [all_len+ind_len for all_len, ind_len in zip(control_performance, batch_control_performance)]

                no_control_rouge, _  = calculate_rouge(summary_to_rouge, outputs[0], rouge, no_control_rouge)
                test_rouge, temp_scores = calculate_rouge(summary_to_rouge, outputs[1], rouge, test_rouge)

                for i, r in enumerate(rouge_for_all):
                    rouge_scores, _ = calculate_rouge(summary_to_rouge, outputs[2][i], rouge, r)
                    rouge_for_all[i] = rouge_scores  
                if batch_count % 10 == 0:
                    logger.info(f'Processed {batch_count} batches.')
                    logger.info(f'True summary: {summary_to_rouge[0]}')
                    for i, lt in enumerate(control_tokens):
                        logger.info(f'Control category {lt}, output: {outputs[2][i][0]}')
                    logger.info(f'Batch control performance: {batch_control_performance}')
                    logger.info(f'Batch ROUGE: {temp_scores}.')

            logger.info(f'Done testing.')
            for i, r in enumerate(rouge_for_all):
                rouge_for_all[i] = {key: {metric: float(r[key][metric]/batch_count) for metric in r[key].keys()} for key in r.keys()}
                logger.info(f'Rouge on test set, controls {control_tokens[i]}: {rouge_for_all[i]}.')

            test_rouge = {key: {metric: float(test_rouge[key][metric]/batch_count) for metric in test_rouge[key].keys()} for key in test_rouge.keys()}
            no_control_rouge = {key: {metric: float(no_control_rouge[key][metric]/batch_count) for metric in no_control_rouge[key].keys()} for key in no_control_rouge.keys()}
            logger.info(f'Rouge on test set, native controls: {test_rouge}.')
            logger.info(f'Rouge on test set, no controls: {no_control_rouge}.')

            control_performance = [perf / batch_count for perf in control_performance]
            logger.info(f'Control performance: {control_performance}')
            # with open(Path(save_model_path, 'metrics_epoch_' + str(epoch) + '.pkl'), 'wb') as file:
                # pickle.dump(metrics, file)

    else:

        while stop_condition:
            logger.info(f'Current learning rate is: {optimizer.param_groups[0]["lr"]}')
            epoch += 1
            no_samples = 0
            epoch_loss = 0
            val_epoch_loss = 0
            rouge_scores = None
            val_rouge_scores = None
            batch_count = 0
            val_batch_count = 0

            train_controls = [[] for i in range(len(control_tokens))]
            baseline_controls = []
            val_controls = [[] for i in range(len(control_tokens))]

            model.train()


            start = time.time()
            logger.info(f'Training, epoch {epoch}.')

            # Train epoch
            for batch in train_iter:
                if batch.story.shape[1] > max_len:
                    continue
                if batch.summary.shape[1] > max_len:
                    continue

                batch_count += 1

                # Prepare inputs for forward pass

                story, summary_to_rouge, summary_to_pass, lead_3, codes = prepare_batch(batch, txt_field, txt_nonseq_field, sent_end_inds, controls, reinforcement=args.reinforcement)
                
                if 'sentiment' in controls:
                    reward_fn = obtain_reward_sentiment
                elif 'length' in controls:
                    reward_fn = obtain_reward_length
                elif 'source' in controls:
                    reward_fn = obtain_reward_source
                if args.reinforcement:
                    if args.ml_reinforcement:
                        output, sample_output, output_tokens, baseline_tokens = model.ml_rl_inference(story.to(device), sos_idx, eos_idx)
                    else:
                        sample_output, output_tokens, baseline_tokens = model.rl_inference(story.to(device), sos_idx, eos_idx)
                    baseline_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in summ]) for summ in baseline_tokens]
                else:
                    output, _ = model(story.to(device), summary_to_pass.to(device)) # second output is attention 
                    output_tokens = torch.argmax(output, dim=2)

                output_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in summ]) for summ in output_tokens]
                
                rouge_scores, output_rouge = calculate_rouge(summary_to_rouge, output_to_rouge, rouge, rouge_scores)
                
                if args.reinforcement:
                    sample_output = sample_output.contiguous().view(-1, sample_output.shape[-1])
                    sample_to_loss = output_tokens[:,1:].contiguous().view(-1)

                    loss = crossentropy(sample_output, sample_to_loss).contiguous().view(output_tokens.shape[0], -1)
                    rewards, control_perf, baseline_perf = reward_fn(output_to_rouge, baseline_to_rouge, codes, 
                                                                                        do_rouge=args.rouge_scaling, summary_to_rouge=summary_to_rouge, rouge=rouge)

                    for ind, group in enumerate(codes):
                        if group == '<pos>' or group == '<long>' or group == '<cnn>':
                            train_controls[0].append(control_perf[ind])
                        elif group == '<neg>' or group == '<short>' or group == '<dailymail>':
                            train_controls[1].append(control_perf[ind])
                        elif group == '<neu>' or group == '<medium>':
                            train_controls[2].append(control_perf[ind])
                    
                    baseline_controls.extend(baseline_perf)

                    rewards = rewards.to(device)
                    loss = torch.mul(rewards.unsqueeze(1), loss)
                    loss = loss.mean()
                    if args.ml_reinforcement:
                        summary = batch.summary[:,1:].contiguous().view(-1)
                        output = output.contiguous().view(-1, output.shape[-1])
                        ml_loss = crossentropy(output, summary.to(device)).mean()
                        loss = gamma * loss + (1 - gamma) * ml_loss

                else:
                    summary = batch.summary[:,1:].contiguous().view(-1)
                    output = output.contiguous().view(-1, output.shape[-1])
                    loss = crossentropy(output, summary.to(device)).mean()
                    

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                no_samples += len(batch.story)
                if args.full_train:
                    print_condition = (batch_count - 1) % 500 == 0
                else:
                    print_condition = True

                if print_condition:
                    logger.info(f'Batch {batch_count}, processed {no_samples} stories.')
                    logger.info(summary_to_rouge[0])
                    logger.info(output_to_rouge[0])
                    logger.info(baseline_to_rouge[0])
                    logger.info(f'Average loss: {epoch_loss / batch_count}.')
                    logger.info(f'Latest ROUGE: {output_rouge}.')

                    for n, score in enumerate(train_controls):                        
                        try:
                            # We substract 1 because we intialized the control performance list with zeros
                            logger.info(f'{control_tokens[n]} performance: {sum(score) / (len(score) - 1)}.')
                        except ZeroDivisionError:
                            logger.info(f'Cannot show {control_tokens[n]} performance yet.')

                    end = time.time()
                    logger.info(f'Epoch {epoch} running already for {end-start} seconds.')

            # Validation epoch
            with model.eval() and torch.no_grad():
                for batch in val_iter:
                    val_batch_count += 1

                    story, summary_to_rouge, summary_to_pass, lead_3, codes = prepare_batch(batch, txt_field, txt_nonseq_field, sent_end_inds, controls, reinforcement=args.reinforcement)
                    
                    
                    if args.reinforcement:
                        if args.ml_reinforcement:
                            output, sample_output, output_tokens, baseline_tokens = model.ml_rl_inference(story.to(device), sos_idx, eos_idx)
                        else:
                            sample_output, output_tokens, baseline_tokens = model.rl_inference(story.to(device), sos_idx, eos_idx)
                        baseline_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in summ]) for summ in baseline_tokens]
                    else:
                        output, _ = model(story.to(device), summary_to_pass.to(device)) # second output is attention 
                        output_tokens = torch.argmax(output, dim=2)

                    output_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in summ]) for summ in output_tokens]
                    
                    val_rouge_scores, temp_scores = calculate_rouge(summary_to_rouge, output_to_rouge, rouge, val_rouge_scores)
                    
                    if args.reinforcement:
                        sample_output = sample_output.contiguous().view(-1, sample_output.shape[-1])
                        sample_to_loss = output_tokens[:,1:].contiguous().view(-1)
                        loss = crossentropy(sample_output, sample_to_loss).contiguous().view(output_tokens.shape[0], -1)
                        rewards, control_perf, _ = obtain_reward_sentiment(output_to_rouge, baseline_to_rouge, codes, 
                                                                    do_rouge=args.rouge_scaling, summary_to_rouge=summary_to_rouge, rouge=rouge)

                        for ind, group in enumerate(codes):
                            if group == '<pos>' or group == '<long>' or group == '<cnn>':
                                val_controls[0].append(control_perf[ind])
                            elif group == '<neg>' or group == '<short>' or group == '<dailymail>':
                                val_controls[1].append(control_perf[ind])
                            elif group == '<neu>' or group == '<medium>':
                                val_controls[2].append(control_perf[ind])


                        rewards = rewards.to(device)
                        loss = torch.mul(rewards.unsqueeze(1), loss)
                        loss = loss.mean()
                        if args.ml_reinforcement:
                            summary = batch.summary[:,1:].contiguous().view(-1)
                            output = output.contiguous().view(-1, output.shape[-1])
                            ml_loss = crossentropy(output, summary.to(device)).mean()
                            loss = gamma * loss + (1 - gamma) * ml_loss

                    else:
                        summary = batch.summary[:,1:].contiguous().view(-1)
                        output = output.contiguous().view(-1, output.shape[-1])
                        loss = crossentropy(output, summary.to(device)).mean()

                    if val_batch_count % 100 == 0:
                        logger.info(f'Greedy prediction: {baseline_to_rouge[0]}')
                        logger.info(f'True summary: {summary_to_rouge[0]}')
                    
                    val_epoch_loss += loss.item()
                # logger.info(f'Control performance: {[score / count for score, count in zip(val_controls, len_val_controls)]}.')

            if args.reinforcement:
                scheduler.step()
            else:
                scheduler.step(val_epoch_loss / val_batch_count)
            
            # Averaging ROUGE scores
            rouge_scores = {key: {metric: float(rouge_scores[key][metric]/batch_count) for metric in rouge_scores[key].keys()} for key in rouge_scores.keys()}
            val_rouge_scores = {key: {metric: float(val_rouge_scores[key][metric]/val_batch_count) for metric in val_rouge_scores[key].keys()} for key in val_rouge_scores.keys()}

            metrics['val_loss'].append(val_epoch_loss / val_batch_count)
            metrics['val_rouge'].append(val_rouge_scores)
            metrics['train_loss'].append(epoch_loss / batch_count)
            metrics['train_rouge'].append(rouge_scores)

            train_performance, train_count, val_performance, val_count = [], [], [], []
            for train_score, val_score in zip(train_controls, val_controls):   
                count = len(train_score) if len(train_score) > 0 else 1
                std = statistics.stdev(train_score) if len(train_score) > 0 else 0
                train_performance.append([sum(train_score)/count, std])
                train_count.append(len(train_score))
                count = len(val_score) if len(val_score) > 0 else 1
                std = statistics.stdev(train_score) if len(train_score) > 0 else 0
                val_performance.append([sum(val_score)/count, std])
                val_count.append(len(val_score))
                
            control_performance['train']['performance'].append(train_performance)
            control_performance['train']['count'].append(train_count)
            control_performance['val']['performance'].append(val_performance)
            control_performance['val']['count'].append(val_count)
            
            control_performance['baseline'].append([sum(baseline_controls) / len(baseline_controls), statistics.stdev(baseline_controls)])

            # Saving model if validation loss decreasing
            logger.info(metrics)
            logger.info(control_performance)

            if epoch > 1:
                if args.reinforcement:
                    save_model(model, save_model_path, epoch, save_suffix)    
                elif metrics['val_loss'][-1] < metrics['val_loss'][-2]:
                    save_model(model, save_model_path, epoch, save_suffix)    
            else:
                save_model(model, save_model_path, epoch, save_suffix)    


            
            with open(Path(save_model_path, 'metrics_epoch_' + str(epoch) + save_suffix + '.pkl'), 'wb') as file:
                pickle.dump(metrics, file)
            with open(Path(save_model_path, 'control_epoch_' + str(epoch) + save_suffix + '.pkl'), 'wb') as file:
                pickle.dump(control_performance, file)

            logger.info(f'Recursion error count at {recursion_count}.')

            end = time.time()
            logger.info(f'Epoch {epoch} took {end-start} seconds.')

            if args.reinforcement:
                stop_condition = epoch < (args.epoch + args.max_epoch)
            else:
                stop_condition = optimizer.param_groups[0]['lr'] > 1e-5





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='batch size')
    parser.add_argument('--emb_dim', type=int, default=EMB_DIM,
                        help='size of embedding layer')
    parser.add_argument('--hid_dim', type=int, default=HID_DIM,
                        help='size of hidden layer')
    parser.add_argument('--n_layers', type=int, default=N_LAYERS,
                        help='number of layers in encoder and decoder')
    parser.add_argument('--kernel_size', type=int, default=KERNEL_SIZE,
                        help='size of kernel in convolution')    
    parser.add_argument('--dropout_prob', type=int, default=DROPOUT_PROB,
                        help='dropout probability')    
    parser.add_argument('--lr', type=float, default=0.2,
                        help='predictor learning rate')
    parser.add_argument('--self_attention', action='store_true',
                        help='Whether to use self_attention')
    parser.add_argument('--share_weights', action='store_true',
                        help='Share weights between encoder and decoder as per Fan')


    parser.add_argument('--seed', type=int, default=42,
                        help='Train with a fixed seed')
    parser.add_argument('--epoch', type=int, default=0,
                        help='Epoch number (if cont training)')
    parser.add_argument('--max_epoch', type=int, default=20,
                        help='Max epoch number (if reinforcement)')
    parser.add_argument('--test', action='store_true',
                        help='Use test set')
    parser.add_argument('--full_train', action='store_true',
                        help='Train full model')

    parser.add_argument('--debug', action='store_true',
                        help='Debug for CUDA or not')
    parser.add_argument('--count_pads', action='store_true',
                        help='Count what % paddings in batches are or not')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU for training')

    parser.add_argument('--save_model_to', type=str, default="saved_models/",
                        help='Output path for saved model')
    parser.add_argument('--controls', type=int, default=0,
                        help='Specification for control codes. \
                        0 is all, 1 is length, 2 is source style, 3 is entities, 4 is sentiment, -1 is none. ')
    parser.add_argument('--no_len_tokens', type=int, default=NO_LEN_TOKENS,
                        help='Number of bins for summary lengths in terms of tokens.')

    parser.add_argument('--reinforcement', action='store_true',
                        help='Optimize with reinforcement')
    parser.add_argument('--ml_reinforcement', action='store_true',
                        help='Optimize with ml and rl objectives')
    parser.add_argument('--adam', action='store_true',
                        help='Use Adam optimization')
    parser.add_argument('--gamma', type=float, default=0.9984,
                        help='weight for rl loss (weight for ml loss is 1 - gamma)')
    parser.add_argument('--rouge_scaling', action='store_true',
                        help='Whether to scale reward by rouge for rl experiment')
    parser.add_argument('--sentiment', action='store_true',
                        help='Whether to scale reward by rouge for rl experiment')
    parser.add_argument('--only_pos', action='store_true',
                        help='Whether to include only positive sentiment control')
    parser.add_argument('--only_neg', action='store_true',
                        help='Whether to include only negative sentiment control')

    parser.add_argument('--synth', action='store_true',
                        help='Whether to use on synthetic data')
    parser.add_argument('--timing', action='store_true',
                        help='Whether to use time the rl experiment')
    parser.add_argument('--generate', action='store_true',
                        help='Demo')

    args = parser.parse_args()
    logger.info(args)

    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synth:
        train_synth()
    else:
        train()
