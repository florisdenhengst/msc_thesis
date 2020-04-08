import os
import re
import sys
import math
import time
import random

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

from data_preprocess import anonymize_and_bpe_data, MyIterator, batch_size_fn
from artificial_data_preprocess import Synthetic
from model import ControllableSummarizer

EMB_DIM = 340 # from paper 
HID_DIM = 512 # from paper
N_LAYERS = 8 # from paper
KERNEL_SIZE = 3 # from paper
DROPOUT_PROB = 0.2 # from paper 
NO_LEN_TOKENS = 10
BATCH_SIZE = 32


logger = logging.getLogger('Training log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

def count_pads(train_iter, padding_idx):
    stories_len = []
    summaries_len = []
    st_all_tokens = 0
    sm_all_tokens = 0
    st_pads = 0
    sm_pads = 0
    for batch in train_iter:
        stories_len.append(batch.stories.shape[1])
        summaries_len.append(batch.summary.shape[1])
        if args.count_pads:
            st_all_tokens += batch.stories.shape[0] * batch.stories.shape[1] 
            sm_all_tokens += batch.summary.shape[0] * batch.summary.shape[1] 
            st_pads += sum([sum([1 for ind in st if ind==padding_idx]) for st in batch.stories])
            sm_pads += sum([sum([1 for ind in st if ind==padding_idx]) for st in batch.summary])
    if args.count_pads:
        logger.info(f'In stories, pads are {100*st_pads/st_all_tokens} of all tokens.')
        logger.info(f'In summaries, pads are {100*sm_pads/sm_all_tokens} of all tokens.')

    logger.info(f'Maximum length of article: {max(stories_len)}.')

    stories_len.sort(reverse=True)
    logger.info(stories_len[0:5])
    logger.info(f'Maximum length of summary: {max(summaries_len)}.')
    max_len = max([max(stories_len), max(summaries_len)])

    if max_len > 1000:
        sys.setrecursionlimit(max_len + 10)
        max_len = 1000
    return max_len



def prepare_summaries(batch, txt_field, output=False):
    if output:
        summary = batch
    else:
        summary = batch.summary

    summary_to_pass = exclude_token(summary, txt_field.vocab.stoi['<eos>'])
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


def test_on_control(model, batch, txt_field, native_controls, flex_controls, control, control_evl_fn, device):
    story = prepare_story_for_control_test(batch.stories, txt_field, control=control, control_codes=native_controls)
    output = model.inference(story.to(device), txt_field.vocab.stoi['<sos>'], txt_field.vocab.stoi['<eos>'])
    output_to_rouge, _ = prepare_summaries(output, txt_field, output=True)
    native_results = control_evl_fn(output, batch.summary, story, txt_field)

    flex_results = []
    for flex in flex_controls:
        story = prepare_story_for_control_test(batch.stories, txt_field, control=control, control_codes=flex)
        output = model.inference(story.to(device), txt_field.vocab.stoi['<sos>'], txt_field.vocab.stoi['<eos>'])
        # output_argmax = [[ind for ind in torch.argmax(summary, dim=1)] for summary in output]        
        flex_results.append(control_evl_fn(output, batch.summary, story, txt_field))
    return output_to_rouge, native_results, flex_results

def evalutate_on_length(output, summary, story, txt_field):
    sos_idx = txt_field.vocab.stoi['<sos>']
    eos_idx = txt_field.vocab.stoi['<eos>']
    length_outputs = []
    for out in output:
        length = 0
        for ind in output:
            if ind == sos_idx:
                continue
            if ind == eos_idx:
                break
            length += 1
        length_outputs.append(length)
    length_summary = []
    for summ in summary:
        length = 0
        for ind in summ:
            if ind == sos_idx:
                continue
            if ind == eos_idx:
                break
            length += 1
        length_summary.append(length)
    return {'output': length_outputs, 'summary': length_summary}

def test_on_length(model, batch, txt_field, len_tokens, device):
    native_controls = ['<len' + str(int(len_ind)) + '>' for len_ind in batch.length_tokens]
    flex_controls = []
    for token in len_tokens:
        flex_controls.append([token for i in range(len(batch.length_tokens))])
    output_to_rouge, native_results, flex_results = test_on_control(model, batch, txt_field, native_controls, flex_controls, 'length', evalutate_on_length, device)
    length_performance = [sum(flex['output'])/len(flex['output']) for flex in flex_results]

    return output_to_rouge, length_performance



def prepare_batch(batch, txt_field, txt_nonseq_field, sent_end_inds):
    lead_3 = get_lead_3(batch.stories, txt_field, sent_end_inds) 
    
    summary_to_rouge, summary_to_pass = prepare_summaries(batch, txt_field)
    
    ent_tensor = extract_entities_to_prepend(lead_3, summary_to_rouge, txt_field)  
    story = prepare_story_for_control_test(batch.stories, txt_field, control='entities', ent_tensor=ent_tensor)

    len_codes = ['<len' + str(int(len_ind)) + '>' for len_ind in batch.length_tokens]
    story = prepare_story_for_control_test(batch.stories, txt_field, control='length', control_codes=len_codes)

    src_codes = ['<' + txt_nonseq_field.vocab.itos[src_ind] + '>' for src_ind in batch.source]
    story = prepare_story_for_control_test(batch.stories, txt_field, control='source', control_codes=src_codes)

    return story, summary_to_rouge, summary_to_pass, lead_3

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

def save_model(model, save_model_path, epoch):
    Path.mkdir(save_model_path, exist_ok=True)
    if Path.exists(Path(save_model_path, 'summarizer_epoch_' + str(epoch-1) + '.model')):
        logger.info('Removing model from previous epoch...')
        Path.unlink(Path(save_model_path, 'summarizer_epoch_' + str(epoch-1) + '.model'))
    logger.info(f'Saving model at epoch {epoch}.')
    torch.save(model.state_dict(), Path(save_model_path, 'summarizer_epoch_' + str(epoch) + '.model'))


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

    train_fields = [('id', txt_nonseq_field), ('stories', txt_field), ('length_tokens', num_field),
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

    if args.test:
        _, _, test_data = TabularDataset(path=csv_path, format='csv', 
                                skip_header=True, fields=train_fields).split(split_ratio=[0.922, 0.043, 0.035], random_state=st)
        train_data, val_data = [], []
        test_iter = BucketIterator(dataset=test_data, batch_size=args.batch_size, 
            sort_key=lambda x:(len(x.stories), len(x.summary)), shuffle=True, train=False)
        txt_field.build_vocab(test_data)
        txt_nonseq_field.build_vocab(test_data)
    else:
        train_data, val_data, _ = TabularDataset(path=csv_path, format='csv', 
                                skip_header=True, fields=train_fields).split(split_ratio=[0.922, 0.043, 0.035], random_state=st)
        test_data = []
        train_iter = BucketIterator(dataset=train_data, batch_size=args.batch_size, 
            sort_key=lambda x:(len(x.stories), len(x.summary)), shuffle=True, train=True)
        val_iter = BucketIterator(dataset=val_data, batch_size=args.batch_size, 
            sort_key=lambda x:(len(x.stories), len(x.summary)), shuffle=True, train=False)
        txt_field.build_vocab(train_data, val_data)
        txt_nonseq_field.build_vocab(train_data, val_data)

        #Check for consistency of random seed
        sample = next(iter(train_iter))
        logger.info(f'1st train article id is {sample.id}')
        sample = next(iter(val_iter))
        logger.info(f'1st val article id is {sample.id}')
        
    
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

    len_tokens = ['<len' + str(i+1) + '>' for i in range(args.no_len_tokens)]
    source_tokens = ['<cnn>', '<dailymail>']
    txt_field = add_tokens_to_vocab(txt_field, len_tokens+source_tokens)

    padding_idx = txt_field.vocab.stoi[txt_field.pad_token]
    sos_idx = txt_field.vocab.stoi['<sos>']
    eos_idx = txt_field.vocab.stoi['<eos>']
    input_dim = len(txt_field.vocab.itos)
    output_dim = len(txt_field.vocab.itos)
    
    end = time.time()
    logger.info(f'finished in {end-start} seconds.')

    if args.test:
        max_len = 1000
    else: 
        max_len = count_pads(train_iter, padding_idx)
            
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
        
        if Path.exists(Path(save_model_path, 'summarizer_epoch_' + str(args.epoch) + '.model')):
            model.load_state_dict(torch.load(Path(save_model_path, 'summarizer_epoch_' + str(args.epoch) + '.model')))
        epoch = args.epoch
        metrics = {'train_loss':[], 'train_rouge':[], 'val_loss':[], 'val_rouge':[]}
        if Path.exists(Path(save_model_path, 'metrics_epoch_' + str(args.epoch) + '.pkl')):
            with open(Path(save_model_path, 'metrics_epoch_' + str(args.epoch) + '.pkl'), 'rb') as file:
                metrics  = pickle.load(file)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        no_params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info(f'{no_params} trainable parameters in the model.')
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=0)
    crossentropy = nn.CrossEntropyLoss(ignore_index=padding_idx)
    
    rouge = Rouge()
    sent_end_tokens = ['.', '!', '?']
    sent_end_inds = [txt_field.vocab.stoi[token] for token in sent_end_tokens]
    recursion_count = 0    

    if args.test:
        rouge_scores = None
        batch_count = 0
        length_performance = [0 for i in range(len(len_tokens))]
        with model.eval() and torch.no_grad():
            for no, batch in enumerate(test_iter):
                batch_count += 1
                
                story, summary_to_rouge, _, lead_3 = prepare_batch(batch, txt_field, txt_nonseq_field, sent_end_inds)
                
                # start = time.time()
                # output = model.inference(story.to(device) , sos_idx, eos_idx)
                # end = time.time()
                # logger.info(f'finished one pass in {end-start} seconds.')

                # output = torch.tensor([output['beam_' + str(abs(i))][b] for b, i in enumerate(beams)])
                # output_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in summ]) for summ in output]

                start = time.time()
                output_to_rouge, batch_lengths = test_on_length(model, batch, txt_field, len_tokens, device)
                end = time.time()
                logger.info(f'finished one length test in {end-start} seconds.')

                length_performance = [all_len+ind_len for all_len, ind_len in zip(length_performance, batch_lengths)]
                total_length_performance = [l/batch_count for l in length_performance]
                rouge_scores, temp_scores = calculate_rouge(summary_to_rouge, output_to_rouge, rouge, rouge_scores)
                
                if no % 50 == 0:
                    logger.info(f'Processed {no+1} batches.')
                    logger.info(summary_to_rouge)
                    logger.info(output_to_rouge)
                    logger.info(f'Length performance: {total_length_performance}')
                    # logger.info(f'Average loss: {epoch_loss / no}.')
                    logger.info(f'Latest ROUGE: {temp_scores}.')
                    
            rouge_scores = {key: {metric: float(rouge_scores[key][metric]/batch_count) for metric in rouge_scores[key].keys()} for key in rouge_scores.keys()}
            logger.info(f'Test rouge: {rouge_scores}.')
            logger.info(f'Length performance: {total_length_performance}')

    else:
        while optimizer.param_groups[0]['lr'] > 1e-5:
            logger.info(f'Current learning rate is: {optimizer.param_groups[0]["lr"]}')
            epoch += 1
            no_samples = 0
            epoch_loss = 0
            val_epoch_loss = 0
            rouge_scores = None
            val_rouge_scores = None
            batch_count = 0
            val_batch_count = 0

            model.train()


            start = time.time()
            logger.info(f'Training, epoch {epoch}.')

            # Train epoch
            for no, batch in enumerate(train_iter):
                if batch.stories.shape[1] > max_len:
                    continue

                batch_count += 1

                # Prepare inputs for forward pass
                story, summary_to_rouge, summary_to_pass, lead_3 = prepare_batch(batch, txt_field, txt_nonseq_field, sent_end_inds)
                output, _ = model(story.to(device), summary_to_pass.to(device)) # second output is attention 
                output_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in torch.argmax(summ, dim=1)]) for summ in output]        
                
                rouge_scores, temp_scores = calculate_rouge(summary_to_rouge, output_to_rouge, rouge, rouge_scores)
                
                output = output.contiguous().view(-1, output.shape[-1])
                summary = batch.summary[:,1:].contiguous().view(-1)
                loss = crossentropy(output, summary.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                no_samples += len(batch.stories)
                if no % 500 == 0 and no != 0:
                    logger.info(f'Batch {no}, processed {no_samples} stories.')
                    logger.info(summary_to_rouge[0])
                    logger.info(output_to_rouge[0])
                    logger.info(f'Average loss: {epoch_loss / no}.')
                    logger.info(f'Latest ROUGE: {temp_scores}.')
                    end = time.time()
                    logger.info(f'Epoch {epoch} running already for {end-start} seconds.')

            # Validation epoch
            with model.eval() and torch.no_grad():
                for batch in val_iter:
                    val_batch_count += 1
                    story, summary_to_rouge, summary_to_pass, lead_3 = prepare_batch(batch, txt_field, txt_nonseq_field, sent_end_inds)
                    output, _ = model(story.to(device), summary_to_pass.to(device))
                    
                    if val_batch_count % 100 == 0:
                        output_greedy, _ = model.greedy_inference(story.to(device) , sos_idx, eos_idx)
                        output_greedy = [' '.join([txt_field.vocab.itos[ind] for ind in summ]) for summ in output_greedy]
                        logger.info(f'Greedy prediction: {output_greedy[0]}')
                        logger.info(f'True summary: {summary_to_rouge[0]}')
                    output_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in torch.argmax(summ, dim=1)]) for summ in output]
                    val_rouge_scores, temp_scores = calculate_rouge(summary_to_rouge, output_to_rouge, rouge, val_rouge_scores)

                    output = output.contiguous().view(-1, output.shape[-1])
                    summary = batch.summary[:,1:].contiguous().view(-1)
                    val_loss = crossentropy(output, summary.to(device))
                    val_epoch_loss += val_loss.item()
                scheduler.step(val_epoch_loss / val_batch_count)
            
            # Averaging ROUGE scores
            rouge_scores = {key: {metric: float(rouge_scores[key][metric]/batch_count) for metric in rouge_scores[key].keys()} for key in rouge_scores.keys()}
            val_rouge_scores = {key: {metric: float(val_rouge_scores[key][metric]/val_batch_count) for metric in val_rouge_scores[key].keys()} for key in val_rouge_scores.keys()}

            metrics['val_loss'].append(val_epoch_loss / val_batch_count)
            metrics['val_rouge'].append(val_rouge_scores)
            metrics['train_loss'].append(epoch_loss / batch_count)
            metrics['train_rouge'].append(rouge_scores)

            # Saving model if validation loss decreasing
            logger.info(metrics)
            if epoch > 1:
                if metrics['val_loss'][-1] < metrics['val_loss'][-2]:
                    save_model(model, save_model_path, epoch)    
            else:
                save_model(model, save_model_path, epoch)    

            with open(Path(save_model_path, 'metrics_epoch_' + str(epoch) + '.pkl'), 'wb') as file:
                pickle.dump(metrics, file)

            logger.info(f'Recursion error count at {recursion_count}.')

            end = time.time()
            logger.info(f'Epoch {epoch} took {end-start} seconds.')


def train_synth():
    padding_idx = 0
    sos_idx = 1
    eos_idx = 2

    data = Synthetic(batch_size=32, vocab_size=100, max_in=50, max_out=10, min_in=20, min_out=5,
                    padding_idx=padding_idx, sos_idx=sos_idx, eos_idx=eos_idx)
    test_data = Synthetic(batch_size=5, vocab_size=100, max_in=50, max_out=10, min_in=20, min_out=5,
                    padding_idx=padding_idx, sos_idx=sos_idx, eos_idx=eos_idx)
    
    input_dim = data.vocab_size + 10    # control length codes
    output_dim = data.vocab_size + 10   # control length codes
    max_len = max(data.max_in_len, data.max_out_len) + 2
    
    model = ControllableSummarizer(input_dim=input_dim, output_dim=output_dim, emb_dim=args.emb_dim, 
                                    hid_dim=args.hid_dim, n_layers=args.n_layers, kernel_size=args.kernel_size, 
                                    dropout_prob=args.dropout_prob, device=device, padding_idx=padding_idx, 
                                    share_weights=args.share_weights, max_length=max_len, 
                                    self_attention=int(args.self_attention)).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    no_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f'{no_params} trainable parameters in the model.')

    crossentropy = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.99, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, last_epoch=-1)

    epoch_loss = 0
    for n, batch in enumerate(data):
        optimizer.zero_grad()
        story = batch['input']
        
        summary_to_pass = exclude_token(batch['output'], int(data.eos_idx))
        
        output, _ = model(story.to(device), summary_to_pass.to(device))
        
        output_to_rouge = [[int(ind) for ind in torch.argmax(summ, dim=1)] for summ in output]
        output = output.contiguous().view(-1, output.shape[-1])
        summary = batch['output'][:,1:].contiguous().view(-1)
        loss = crossentropy(output, summary.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        epoch_loss += loss.item()

        if n % 2000 == 0:
            logger.info(f'Batch {n+1}, loss: {epoch_loss / (n+1)}.')
            logger.info
            logger.info(story[0])
            logger.info(summary_to_pass[0])
            logger.info(output_to_rouge[0])
        if n % 2000 == 0:
            if  n != 0:
                scheduler.step()
            if args.test:
                batch_count = 0
                with model.eval() and torch.no_grad():
                    for no, batch in enumerate(test_data):
                        batch_count += 1
                        story = batch['input']
                        summary_to_pass = exclude_token(batch['output'], int(data.eos_idx))
                        output = model.inference(story.to(device) , sos_idx, eos_idx)
                        # logger.info(f'Beams before selection: {output}')
                        # output = torch.tensor([output['beam_' + str(abs(i))][b] for b, i in enumerate(beams)])
                        # print(output)
                        
                        logger.info(f'Processed {no} stories.')
                        logger.info(f'True: {summary_to_pass}')
                        logger.info(f'Beam: {output}')
                            # logger.info(f'Greedy: {greedy_output}')
                        if no % 1 == 0 and no != 0:
                            break





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
    parser.add_argument('--seed', type=int, default=42,
                        help='Train with a fixed seed')
    parser.add_argument('--epoch', type=int, default=0,
                        help='Epoch number (if cont training)')
    parser.add_argument('--test', action='store_true',
                        help='Use test set')
    parser.add_argument('--full_train', action='store_true',
                        help='Train full model')
    parser.add_argument('--share_weights', action='store_true',
                        help='Share weights between encoder and decoder as per Fan')
    parser.add_argument('--debug', action='store_true',
                        help='Debug for CUDA or not')
    parser.add_argument('--count_pads', action='store_true',
                        help='Count what % paddings in batches are or not')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU for training')
    parser.add_argument('--save_model_to', type=str, default="saved_models/",
                        help='Output path for saved model')
    parser.add_argument('--no_len_tokens', type=int, default=NO_LEN_TOKENS,
                        help='Number of bins for summary lengths in terms of tokens.')
    parser.add_argument('--synth', action='store_true',
                        help='Whether to use on synthetic data')
    parser.add_argument('--self_attention', action='store_true',
                        help='Whether to use self_attention')

    args = parser.parse_args()

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