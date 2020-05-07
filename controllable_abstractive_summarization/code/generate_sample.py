import os
import re
import spacy
import pickle
import argparse


from pathlib import Path
from torchtext.data import Field

from data_preprocess import BPEncoder

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

def preprocess_text(text_path, max_len=998):
    
    data_path = Path(Path.cwd(), 'data/')
    samples_path = Path(Path.cwd(), 'sample_docs/')
    with open(Path(data_path, 'cnn_dailymail.bpe'), 'r') as codes:
        bpencoder = BPEncoder(codes)
    with open(Path(samples_path, text_path), 'r') as file:
        text = file.read()

    

    unicode_detector =  re.compile('[^\x00-\x7F]+')
    bad_punctuation_detector =  re.compile(r'(.)\1{3,}|\n')
    text = unicode_detector.sub(' ', text)
    text = bad_punctuation_detector.sub(' ', text.lower())
    # print(text)
    text = bpencoder.encode(text)
    text = [tok.text for tok in nlp.tokenizer(text)]
    if len(text) > max_len:
        text = [text[n*max_len:n*max_len+max_len] for n in range(len(text) // max_len + 1)]
        
    else:
        text = [text]
    return text    


if __name__ == '__main__':
    txt_field = Field(tokenize=lambda x: [tok.text for tok in nlp.tokenizer(x)], 
                        init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
    txt_field.build_vocab()
    save_model_path = Path(Path.cwd(), 'saved_models/')
    with open(Path(save_model_path, 'vocab_stoi.pkl'), 'rb') as file:
        txt_field.vocab.stoi = pickle.load(file)
    with open(Path(save_model_path, 'vocab_itos.pkl'), 'rb') as file:
        txt_field.vocab.itos = pickle.load(file)


    text_paths = os.listdir(Path(Path.cwd(), 'sample_docs/'))
    texts = []
    for text_path in text_paths:
        text = preprocess_text(text_path)
        texts.extend(text)
        print(len(texts))    
    num = txt_field.process(texts)
    print([sum([n == 0 for n in nn]) for nn in num])
    print(num.shape)


