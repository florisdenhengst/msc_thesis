# coding=utf-8
# coding: utf-8
import logging
import coloredlogs

import os
import numpy as np
from torch.utils.data import Dataset
import torch

logger = logging.getLogger('Training log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Synthetic(Dataset):
    def __init__(self, batch_size, vocab_size, max_in=100, max_out=20, min_in=20, min_out=5, 
                        padding_idx=0, sos_idx=1, eos_idx=2):
        super(Synthetic).__init__()
        self.batch_size = batch_size
        self.count = 0
        
        self.padding_idx = padding_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        
        self.vocab_size = vocab_size
        self.max_in_len = max_in
        self.max_out_len = max_out
        self.possible_input_lens = np.random.randint(min_in, self.max_in_len, size=25)

        self.possible_output_lens = np.random.randint(min_out, self.max_out_len, size=25) 

    def __len__(self):

        return self.count

    def __getitem__(self, idx):
        
        item = {'input': [], 'output': [], 'output_len': []}
        for i in range(self.batch_size):
            self.count += 1
            
            input_len = self.possible_input_lens[np.random.randint(0, len(self.possible_input_lens))]
            item_input = np.append([self.sos_idx], np.random.randint(3, self.vocab_size+1, size=input_len))
            item_input = torch.tensor(np.append(item_input, [self.eos_idx]), dtype=torch.int64)

            output_len = self.possible_output_lens[np.random.randint(0, len(self.possible_output_lens))]
            item_output = torch.cat((item_input[0:int(np.round(output_len/2))], item_input[(input_len-(output_len-int(np.round(output_len/2)))):]))
            assert len(item_output) == output_len + 2
            item['output_len'].append(torch.tensor(int(10*np.around(output_len/self.max_out_len, decimals=1))-1))

            if len(item_input) < self.max_in_len + 2:
                pads = torch.zeros(self.max_in_len + 2 - len(item_input), dtype=torch.int64)
                item['input'].append(torch.cat((item_input, pads)))
            if len(item_output) < self.max_out_len + 2:
                pads = torch.zeros(self.max_out_len + 2 - len(item_output), dtype=torch.int64)
                item['output'].append(torch.cat((item_output, pads)))
        item['input'] = torch.stack(item['input'], dim=0)
        item['output'] = torch.stack(item['output'], dim=0)

        return item
