import os
import time
import spacy
import argparse
import torch
import torch.nn as nn
import numpy as np

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

def add_tokens_to_vocab(txt_field, tokens):
    for token in tokens:
        txt_field.vocab.stoi[token] = len(txt_field.vocab.stoi)
        txt_field.vocab.itos.append(token)
    return txt_field


def exclude_token(summaries, eos):
    
    new_summaries = []
    for summ in summaries:
        eos_idx = (summ == eos).nonzero()
        new_summaries.append(torch.cat((summ[0:eos_idx], summ[eos_idx+1:])))
    
    return torch.stack(new_summaries)

def train():
    data_path = os.path.join(os.getcwd(), 'data/')

    nlp = spacy.load("en", disable=["tagger", "parser", "ner"])
    txt_field = Field(tokenize=lambda x: [tok.text for tok in nlp.tokenizer(x)], 
                        init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
    num_field = Field(sequential=False, use_vocab=False)
    txt_nonseq_field = Field(sequential=False, use_vocab=True)

    train_fields = [('id', None), ('stories', txt_field), ('length_tokens', num_field),
                    ('length_sentences', num_field), ('source', txt_nonseq_field), 
                    ('entities', None), ('summary', txt_field)]
    
    print('Started loading data...', end='')

    start = time.time()
    csv_path = os.path.join(data_path, 'cnn.csv')
    if not os.path.exists(csv_path):
        print(f'creating pre-processed data...', end='')
        anonymize_and_bpe_data(data_path=data_path, sources=['cnn'], cut_off_length=400)

    dataset = TabularDataset(path=csv_path, format='csv', skip_header=True, fields=train_fields)
    train_iter = BucketIterator(dataset=dataset, batch_size=32, 
            sort_key=lambda x:(len(x.stories), len(x.summary)), shuffle=True, train=True)

    end = time.time()
    
    print(f'finished in {end-start} seconds.')
    
    print('Started building vocabs...', end='')
    
    start = time.time()
    txt_field.build_vocab(dataset)
    txt_nonseq_field.build_vocab(dataset)

    len_tokens = ['<len' + str(i+1) + '>' for i in range(args.no_len_tokens)]
    source_tokens = ['<cnn>', '<dailymail>']
    txt_field = add_tokens_to_vocab(txt_field, len_tokens+source_tokens)

    padding_idx = txt_field.vocab.stoi[txt_field.pad_token]
    sos_idx = txt_field.vocab.stoi['<sos>']
    eos_idx = txt_field.vocab.stoi['<eos>']
    input_dim = len(txt_field.vocab)
    output_dim = len(txt_field.vocab)
    end = time.time()
    
    print(f'finished in {end-start} seconds.')

    # train_iter = MyIterator(dataset, batch_size=20000, device=0, repeat=False, 
    #                     sort_key= lambda x:(len(x.stories), len(x.summary)),
    #                     batch_size_fn=batch_size_fn, train=True, shuffle=True)
    if args.val is 0:
        val_iter = None
    else:
        val_dataset = TabularDataset(path=os.path.join(data_path, 'cnn_val.csv'), format='csv', skip_header=True, fields=train_fields)
        val_iter = BucketIterator(dataset=val_dataset, batch_size=len(val_dataset), 
            sort_key=lambda x:(len(x.stories), len(x.summary)), shuffle=True, train=False)

    model = ControllableSummarizer(input_dim=input_dim, output_dim=output_dim, emb_dim=args.emb_dim, 
                                    hid_dim=args.hid_dim, n_layers=args.n_layers, kernel_size=args.kernel_size, 
                                    dropout_prob=args.dropout_prob, device=device, padding_idx=padding_idx, share_weights=True)
    
    crossentropy = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.99, nesterov=True)
    
    rouge = Rouge()

    if val_iter is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, last_epoch=-1)

    
    no_samples = 0
    epoch_loss = 0
    rouge_scores = None
    rouge_count = 0
    for no, batch in enumerate(train_iter):
        story, summary = batch.stories, batch.summary
        
        len_tensor = torch.tensor([txt_field.vocab.stoi['<len' + str(int(len_ind)) + '>'] for len_ind in batch.length_tokens]).unsqueeze(dim=1)
        src_tensor = torch.tensor([txt_field.vocab.stoi['<' + txt_nonseq_field.vocab.itos[src_ind] + '>'] for src_ind in batch.source]).unsqueeze(dim=1)
        story = torch.cat((len_tensor, src_tensor, story), dim=1)


        return


        model.train()
        optimizer.zero_grad()
        summary_to_pass = exclude_token(summary, eos_idx)
        
        output, _ = model(story, summary_to_pass) # second output is attention 

        summary_to_rouge = [' '.join([txt_field.vocab.itos[ind] for ind in summ]) for summ in summary]
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
        rouge_count += 1

        
        output = output.contiguous().view(-1, output.shape[-1])
        summary = summary[:,1:].contiguous().view(-1)
        # print(output.shape)
        # print(summary.shape)

        loss = crossentropy(output, summary)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        epoch_loss += loss.item()
        no_samples += len(batch.stories)
        if no % 10:
            print(f'Batch {no}, processed {no_samples} stories.')
            print(f'Average loss: {epoch_loss / no}.')
        
        if val_iter is not None:
            # FIX THIS 10000
            if no_samples % 10000 == 0 and no_samples != 0:
                with model.eval() and torch.no_grad():
                    batch = next(iter(val_iter))
                    output, _ = model(batch.stories)
                    loss = crossentropy(output, batch.summary)
                    scheduler.step(val_loss)
        else:
            scheduler.step()
    rouge_scores = {key: {metric: float(rouge_scores[key][metric]/rouge_count) for metric in rouge_scores[key].keys()} for key in rouge_scores.keys()}
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=20000,
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
    parser.add_argument('--seed', type=int, default=None,
                        help='Train with a fixed seed')
    parser.add_argument('--val', type=int, default=0,
                        help='Use validation set')
    parser.add_argument('--save_model_to', type=str, default="saved_models/",
                        help='Output path for saved model')
    parser.add_argument('--no_len_tokens', type=int, default=NO_LEN_TOKENS,
                        help='Number of bins for summary lengths in terms of tokens.')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train()