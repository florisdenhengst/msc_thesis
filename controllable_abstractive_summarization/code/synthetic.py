import logging
import coloredlogs

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np


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
