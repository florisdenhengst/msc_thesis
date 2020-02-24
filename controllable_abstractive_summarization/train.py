import math
import os
import time
import spacy
import argparse
import random
import torch
import logging
import coloredlogs
import torch.nn as nn
import numpy as np
import re

from collections import Counter
from rouge import Rouge
from data_preprocess import anonymize_and_bpe_data, MyIterator, batch_size_fn
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator

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


def exclude_token(summaries, eos):
    
    new_summaries = []
    for summ in summaries:
        eos_idx = (summ == eos).nonzero()
        new_summaries.append(torch.cat((summ[0:eos_idx], summ[eos_idx+1:])))
    
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


def train():
    data_path = os.path.join(os.getcwd(), 'data/')

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
        csv_path = os.path.join(data_path, 'cnn_dailymail.csv')
    else:
        csv_path = os.path.join(data_path, 'cnn_dailymail_test_purposes.csv')
    if not os.path.exists(csv_path):
        logger.info(f'creating pre-processed data...')
        anonymize_and_bpe_data(data_path=data_path, sources=['cnn'], cut_off_length=400)

    random.seed(42)
    st = random.getstate()
    train_data, val_data, _ = TabularDataset(path=csv_path, format='csv', skip_header=True, fields=train_fields).split(split_ratio=[0.922, 0.043, 0.035], random_state=st)

    # train_data, val_data = torch.utils.data.random_split(dataset, 
    #     [math.ceil(len(dataset)*0.92), len(dataset)-math.ceil(len(dataset)*0.92)])
    # if args.test:
    #     val_data, test_data = torch.utils.data.random_split(val_data, 
    #         [math.ceil(len(val_data)*0.55), len(val_data)-math.ceil(len(val_data)*0.55)])    
        
    #     test_iter = BucketIterator(dataset=test_data, batch_size=32, 
    #         sort_key=lambda x:(len(x.stories), len(x.summary)), shuffle=True, train=False)
    # else:
    #     val_data, _ = torch.utils.data.random_split(val_data, 
    #         [math.ceil(len(val_data)*0.55), len(val_data)-math.ceil(len(val_data)*0.55)])    
    test_data = []
    
    logger.info(f'{len(train_data)} train samples, {len(val_data)} validation samples, {len(test_data)} test samples...', )
    
    # train_iter = BucketIterator(dataset=train_data, batch_size=args.batch_size, 
    #         sort_key=lambda x:(len(x.stories), len(x.summary)), shuffle=True, train=True)
    batch_tokens = 800 * args.batch_size
    train_iter = MyIterator(dataset=train_data, batch_size=batch_tokens, 
        sort_key= lambda x:(len(x.stories), len(x.summary)),
        batch_size_fn=batch_size_fn, train=True, shuffle=True)

    val_iter = BucketIterator(dataset=val_data, batch_size=256, 
            sort_key=lambda x:(len(x.stories), len(x.summary)), shuffle=True, train=False)


    end = time.time()
    
    logger.info(f'finished in {end-start} seconds.')
    
    logger.info('Started building vocabs...')
    
    start = time.time()
    txt_field.build_vocab(train_data, val_data)
    txt_nonseq_field.build_vocab(train_data, val_data)

    sample = next(iter(train_iter))
    logger.info(f'1st train article id is {sample.id}')
    sample = next(iter(val_iter))
    logger.info(f'1st val article id is {sample.id}')
   
    
    len_tokens = ['<len' + str(i+1) + '>' for i in range(args.no_len_tokens)]
    source_tokens = ['<cnn>', '<dailymail>']
    txt_field = add_tokens_to_vocab(txt_field, len_tokens+source_tokens)

    padding_idx = txt_field.vocab.stoi[txt_field.pad_token]
    sos_idx = txt_field.vocab.stoi['<sos>']
    eos_idx = txt_field.vocab.stoi['<eos>']
    input_dim = len(txt_field.vocab)
    output_dim = len(txt_field.vocab)
    end = time.time()
    
    logger.info(f'finished in {end-start} seconds.')

    
    stories_len = []
    summaries_len = []
    st_all_tokens = 0
    sm_all_tokens = 0
    st_pads = 0
    sm_pads = 0
    
    for batch in train_iter:
        stories_len.append(batch.stories.shape[1])
        summaries_len.append(batch.summary.shape[1])
        if if args.count_pads:
            st_all_tokens += batch.stories.shape[0] * batch.stories.shape[1] 
            sm_all_tokens += batch.summary.shape[0] * batch.summary.shape[1] 
            st_pads += sum([sum([1 for ind in st if ind==padding_idx]) for st in batch.stories])
            sm_pads += sum([sum([1 for ind in st if ind==padding_idx]) for st in batch.summary])

    if args.count_pads:
        logger.info(f'In stories, pads are {100*st_pads/st_all_tokens} of all tokens.')
        logger.info(f'In summaries, pads are {100*sm_pads/sm_all_tokens} of all tokens.')
    max_len = max([max(stories_len), max(summaries_len)])
    

    logger.info(f'Initializing model with:') 
    logger.info(f'Input dim: {input_dim}, output dim: {output_dim}, emb dim: {args.emb_dim} hid dim: {args.hid_dim}, {args.n_layers} layers, {args.kernel_size}x1 kernel, {args.dropout_prob} dropout, sharing weights: {args.share_weights}, maximum length: {max_len}.')


    model = ControllableSummarizer(input_dim=input_dim, output_dim=output_dim, emb_dim=args.emb_dim, 
                                    hid_dim=args.hid_dim, n_layers=args.n_layers, kernel_size=args.kernel_size, 
                                    dropout_prob=args.dropout_prob, device=device, padding_idx=padding_idx, 
                                    share_weights=args.share_weights, max_length=max_len).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    no_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f'{no_params} trainable parameters in the model.')
    crossentropy = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.99, nesterov=True)
    
    rouge = Rouge()
    if val_iter is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, last_epoch=-1)
    

    
    sent_end_tokens = ['.', '!', '?']
    sent_end_inds = [txt_field.vocab.stoi[token] for token in sent_end_tokens]
    
    epoch = 0
    metrics = {'train_loss':[], 'train_rouge':[], 'val_loss':[], 'val_rouge':[]}
    
    
    logger.info(f'Current learning rate is: {optimizer.param_groups[0]["lr"]}')


    while optimizer.param_groups[0]['lr'] > 1e-5:
        epoch += 1
        no_samples = 0
        epoch_loss = 0
        val_epoch_loss = 0
        rouge_scores = None
        val_rouge_scores = None
        batch_count = 0
        val_batch_count = 0

        model.train()
        logger.info(f'Training, epoch {epoch}.')
        for no, batch in enumerate(train_iter):
            batch_count += 1

            story, summary = batch.stories, batch.summary

            lead_3 = get_lead_3(story, txt_field, sent_end_inds) 
            summary_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in summ]) for summ in summary]
            summary_to_pass = exclude_token(summary, eos_idx)
            len_tensor = torch.tensor([txt_field.vocab.stoi['<len' + str(int(len_ind)) + '>'] for len_ind in batch.length_tokens]).unsqueeze(dim=1)
            src_tensor = torch.tensor([txt_field.vocab.stoi['<' + txt_nonseq_field.vocab.itos[src_ind] + '>'] for src_ind in batch.source]).unsqueeze(dim=1)
            ent_tensor = extract_entities_to_prepend(lead_3, summary_to_rouge, txt_field)

            story = torch.cat((ent_tensor, len_tensor, src_tensor, story), dim=1)

            optimizer.zero_grad()

            output, _ = model(story.to(device), summary_to_pass.to(device)) # second output is attention 

            output_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in torch.argmax(summ, dim=1)]) for summ in output]        

            if rouge_scores is None:
                rouge_scores = rouge.get_scores(summary_to_rouge, output_to_rouge, avg=True)
            else: 
                temp_scores = rouge.get_scores(summary_to_rouge, output_to_rouge, avg=True)
                rouge_scores = {key: Counter(rouge_scores[key]) + Counter(temp_scores[key]) for key in rouge_scores.keys()}
                for key in rouge_scores:
                    if len(rouge_scores[key]) == 0:
                        rouge_scores[key] = {'f': 0.0, 'p': 0.0, 'r': 0.0}
                    else:
                        rouge_scores[key] = dict(rouge_scores[key]) 
            
            output = output.contiguous().view(-1, output.shape[-1])
            summary = summary[:,1:].contiguous().view(-1)
            loss = crossentropy(output, summary.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            epoch_loss += loss.item()
            no_samples += len(batch.stories)
            if no % 10:
                logger.info(f'Batch {no}, processed {no_samples} stories.')
                logger.info(f'Average loss: {epoch_loss / no}.')
                logger.info(f'Latest ROUGE: {temp_scores}.')
                logger.info('Output sample:')
                logger.info(f'{output_to_rouge[0]}')
                logger.info('Ground truth:')
                logger.info(f'{summary_to_rouge[0]}')

        
        if val_iter is not None:
            
            with model.eval() and torch.no_grad():
                for batch in val_iter:
                    val_batch_count += 1
                    story, summary = batch.stories, batch.summary

                    lead_3 = get_lead_3(story, txt_field, sent_end_inds) 
                    summary_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in summ]) for summ in summary]
                    summary_to_pass = exclude_token(summary, eos_idx)
                    len_tensor = torch.tensor([txt_field.vocab.stoi['<len' + str(int(len_ind)) + '>'] for len_ind in batch.length_tokens]).unsqueeze(dim=1)
                    src_tensor = torch.tensor([txt_field.vocab.stoi['<' + txt_nonseq_field.vocab.itos[src_ind] + '>'] for src_ind in batch.source]).unsqueeze(dim=1)
                    ent_tensor = extract_entities_to_prepend(lead_3, summary_to_rouge, txt_field)

                    story = torch.cat((ent_tensor, len_tensor, src_tensor, story), dim=1)

                    output, _ = model(batch.stories.to(device), summary_to_pass.to(device))
                    output_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in torch.argmax(summ, dim=1)]) for summ in output]

                    if val_rouge_scores is None:
                        val_rouge_scores = rouge.get_scores(summary_to_rouge, output_to_rouge, avg=True)
                    else: 
                        temp_scores = rouge.get_scores(summary_to_rouge, output_to_rouge, avg=True)
                        val_rouge_scores = {key: Counter(val_rouge_scores[key]) + Counter(temp_scores[key]) for key in val_rouge_scores.keys()}
                        for key in val_rouge_scores:
                            if len(val_rouge_scores[key]) == 0:
                                val_rouge_scores[key] = {'f': 0.0, 'p': 0.0, 'r': 0.0}
                            else:
                                val_rouge_scores[key] = dict(val_rouge_scores[key]) 

                    output = output.contiguous().view(-1, output.shape[-1])
                    summary = summary[:,1:].contiguous().view(-1)
                    
                    val_loss = crossentropy(output, summary.to(device))
                    val_epoch_loss += val_loss.item()
                    scheduler.step(val_loss)
                    
                    
                    
        else:
            scheduler.step()
        
        logger.info(f'Current learning rate is: {optimizer.param_groups[0]["lr"]}')
        rouge_scores = {key: {metric: float(rouge_scores[key][metric]/batch_count) for metric in rouge_scores[key].keys()} for key in rouge_scores.keys()}
        val_rouge_scores = {key: {metric: float(val_rouge_scores[key][metric]/batch_count) for metric in val_rouge_scores[key].keys()} for key in val_rouge_scores.keys()}
        metrics['val_loss'].append(val_epoch_loss / val_batch_count)
        metrics['val_rouge'].append(val_rouge_scores)
        
        metrics['train_loss'].append(epoch_loss / batch_count)
        metrics['train_rouge'].append(rouge_scores)
        logger.info(metrics)
        logger.info('Output sample:')
        logger.info(f'{output_to_rouge}')
        logger.info('Ground truth:')
        logger.info(f'{summary_to_rouge}')

        os.makedirs(args.save_model_to, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_model_to, 'summarizer_epoch_' + str(epoch) + '.model'))





            



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
    parser.add_argument('--lr', type=float, default=0.001,
                        help='predictor learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Train with a fixed seed')
    parser.add_argument('--test', action='store_true',
                        help='Use test set')
    parser.add_argument('--full_train', action='store_true',
                        help='Train full model')
    parser.add_argument('--share_weights', action='store_true',
                        help='Share weights between encoder and decoder as per Fan')
    parser.add_argument('--debug', action='store_true',
                        help='Debug for CUDA or not')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU for training')
    parser.add_argument('--save_model_to', type=str, default="saved_models/",
                        help='Output path for saved model')
    parser.add_argument('--no_len_tokens', type=int, default=NO_LEN_TOKENS,
                        help='Number of bins for summary lengths in terms of tokens.')

    args = parser.parse_args()

    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train()