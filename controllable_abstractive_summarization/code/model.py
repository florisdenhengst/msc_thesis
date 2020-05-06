# import fairseq
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

# controllable abstractive summarization
# conv seq2seq with intra-attention and BPE encoding for word embeddings

# https://pypi.org/project/bpemb/
# https://fairseq.readthedocs.io/en/latest/
# https://arxiv.org/abs/1705.03122
# https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb





class ControllableSummarizer(nn.Module):

    def __init__(self, input_dim, output_dim, emb_dim, hid_dim, n_layers, 
                        kernel_size, dropout_prob, device, padding_idx, share_weights=False, max_length=1500, self_attention=1):
        super(ControllableSummarizer, self).__init__()
        self.device = device
        self.padding_idx = padding_idx
        self.max_length = max_length
        self.max_sample_length = int(max_length / 5)

        # original paper shares word embeddings between encoder and decoder
        if share_weights:
            self.tok_embedding = nn.Embedding(input_dim, emb_dim)
            nn.init.normal_(self.tok_embedding.weight, 0, 0.1)
            nn.init.constant_(self.tok_embedding.weight[padding_idx], 0)

            self.pos_embedding = nn.Embedding(max_length, emb_dim)
            nn.init.normal_(self.pos_embedding.weight, 0, 0.1)
            nn.init.constant_(self.pos_embedding.weight[padding_idx], 0)

            self.hid2emb = nn.Linear(hid_dim, emb_dim)
        else:
            self.tok_embedding = None
            self.pos_embedding = None
            self.hid2emb = None
        
        self.encoder = ConvEncoder(input_dim, emb_dim, hid_dim, n_layers, 
                                    kernel_size, dropout_prob, device, padding_idx, 
                                    max_length, self.tok_embedding, self.pos_embedding, self.hid2emb)
        self.decoder = ConvDecoder(output_dim, emb_dim, hid_dim, n_layers, 
                                    kernel_size, dropout_prob, device, padding_idx, 
                                    self_attention, self.max_length, self.max_sample_length, self.tok_embedding, self.pos_embedding, self.hid2emb)

    def forward(self, src_tokens, trg_tokens):
        conved, combined = self.encoder(src_tokens)
        output, attention = self.decoder(trg_tokens, conved, combined)
        return output, attention


    def rl_inference(self, src_tokens, sos_idx, eos_idx):
        conved, combined = self.encoder(src_tokens)
        _, baseline_tokens = self.decoder.forward_sample(conved, combined, sos_idx, eos_idx, greedy=True)            
        output, sample_tokens = self.decoder.forward_sample(conved, combined, sos_idx, eos_idx, sample=True)            
        return output, sample_tokens, baseline_tokens

    def ml_rl_inference(self, src_tokens, trg_tokens, sos_idx, eos_idx):
        conved, combined = self.encoder(src_tokens)
        _, baseline_tokens = self.decoder.forward_sample(conved, combined, sos_idx, eos_idx, greedy=True)            
        sample_output, sample_tokens = self.decoder.forward_sample(conved, combined, sos_idx, eos_idx, sample=True)            
        output, _ = self.decoder(trg_tokens, conved, combined)
        return output, sample_output, sample_tokens, baseline_tokens

    def greedy_inference(self, src_tokens, sos_idx, eos_idx):
        conved, combined = self.encoder(src_tokens)
        output, greedy_tokens = self.decoder.forward_sample(conved, combined, sos_idx, eos_idx, greedy=True)            
        return greedy_tokens

    def inference(self, src_tokens, sos_idx, eos_idx, beam_width=5):

        conved, combined = self.encoder(src_tokens)
        batch_size = conved.shape[0]

        trg_idx = {'beam_' + str(i): [[sos_idx] for j in range(batch_size)] for i in range(beam_width)}
        beam_probs = {'beam_' + str(i): torch.FloatTensor([[0 for k in range(beam_width)] for j in range(batch_size)]).to(self.device) for i in range(beam_width)}
        trigrams = {'beam_' + str(i): [[] for j in range(batch_size)] for i in range(beam_width)}
        
        batch_complete = [False for b in range(batch_size)]
        beam_for_batch = [0 for b in range(batch_size)]

        for i in range(self.max_sample_length): 
            
            iter_tokens = []
            iter_probs = []

            for j in range(beam_width):
                beam = 'beam_' + str(j)

                trg_tokens = torch.LongTensor(trg_idx[beam]).to(self.device)
                
                output, _ = self.decoder(trg_tokens, conved, combined, inference=True)

                next_probs = torch.topk(output, k=beam_width, dim=2)[0].squeeze().squeeze().to(self.device)

                next_tokens = torch.topk(output, k=beam_width, dim=2)[1].squeeze().squeeze().to(self.device)
                if i != 0:
                    next_probs = next_probs[:,-1,:]
                    next_tokens = next_tokens[:,-1,:]

                iter_tokens.append(next_tokens)
                iter_probs.append(beam_probs[beam] + torch.log(next_probs))

            iter_probs = torch.stack(iter_probs)
            tmp_probs = torch.stack([torch.stack([iter_probs[i][j] for i in range(iter_probs.shape[0])]).flatten() for j in range(iter_probs.shape[1])])
            iter_idx = torch.topk(tmp_probs, k=beam_width, dim=1)[1]
            x, y = [], []
            for i, line_prob in enumerate(tmp_probs):
               unique_probs, unique_ids = torch.unique(line_prob, return_inverse=True)
               topk_probs, topk_ids = torch.topk(unique_probs, k=beam_width)
               x.append([idx // beam_width for idx in unique_ids[topk_ids]])
               y.append([idx % beam_width for idx in unique_ids[topk_ids]])

            # x = [idx // beam_width for idx in iter_idx]
            # y = [idx % beam_width for idx in iter_idx]
            
            tmp_idx = copy.deepcopy(trg_idx)
            tmp_trigrams = copy.deepcopy(trigrams)

            for b in range(batch_size):
                if not batch_complete[b]:
                    for j in range(beam_width):
                        trigrams['beam_' + str(j)][b], beam_continue = check_for_trigram(int(iter_tokens[x[b][j]][b,y[b][j]]), tmp_trigrams['beam_' + str(int(x[b][j]))][b])
                        trg_idx['beam_' + str(j)][b] = tmp_idx['beam_' + str(int(x[b][j]))][b] + [int(iter_tokens[x[b][j]][b,y[b][j]])]
                        beam_probs['beam_' + str(j)][b] = iter_probs[x[b][j]][b, y[b][j]] - 100*(1-int(beam_continue))
                        if int(iter_tokens[x[b][j]][b,y[b][j]]) == eos_idx:
                            batch_complete[b] = True
                            beam_for_batch[b] = j
                else:
                    for j in range(beam_width):
                        trg_idx['beam_' + str(j)][b] = trg_idx['beam_' + str(j)][b] + [self.padding_idx]
            if sum(batch_complete) == len(batch_complete):
                # print('finished early')
                return_idx = [trg_idx['beam_' + str(beam_for_batch[i])][i] for i in range(batch_size)]
                return return_idx
        return_idx = [trg_idx['beam_' + str(beam_for_batch[i])][i] for i in range(batch_size)]
        return return_idx


def check_for_trigram(new_token, trigrams):
        if len(trigrams) == 0:
                trigrams.append([new_token])  
        elif len(trigrams[0]) < 3:
            trigrams[0] = trigrams[0] + [new_token]
        else:
            trigrams.append(trigrams[-1][1:] + [new_token])
            if trigrams.count(trigrams[-1]) > 1:
                return trigrams, False
        return trigrams, True

class ConvEncoder(nn.Module):
    """
    inspired by
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb
    """
    def __init__(self, 
                 input_dim, 
                 emb_dim, 
                 hid_dim, 
                 n_layers, 
                 kernel_size, 
                 dropout_prob, 
                 device,
                 padding_idx,
                 max_length=1500, 
                 tok_embedding=None, 
                 pos_embedding=None, 
                 hid2emb=None):
        super(ConvEncoder, self).__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        if tok_embedding is None and pos_embedding is None and hid2emb is None:
            self.tok_embedding = nn.Embedding(input_dim, emb_dim)
            nn.init.normal_(self.tok_embedding.weight, 0, 0.1)
            nn.init.constant_(self.tok_embedding.weight[padding_idx], 0)

            self.pos_embedding = nn.Embedding(max_length, emb_dim)
            nn.init.normal_(self.pos_embedding.weight, 0, 0.1)
            nn.init.constant_(self.pos_embedding.weight[padding_idx], 0)

            self.hid2emb = nn.Linear(hid_dim, emb_dim)
        else:
            self.tok_embedding = tok_embedding
            self.pos_embedding = pos_embedding
            self.hid2emb = hid2emb

        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size, 
                                              padding = (kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        
        # for conv in self.convs:
        #     std = math.sqrt((4 * (1.0 - dropout_prob)) / (conv.kernel_size[0] * hid_dim))
        #     conv.weight.data.normal_(mean=0, std=std)
        #     conv.bias.data.zero_()    

    def forward(self, src_tokens):
        
        src_lengths = src_tokens.shape[1]
        batch_size = src_tokens.shape[0]

        pos = self.pos_embedding(torch.arange(0, src_lengths).unsqueeze(0).repeat(batch_size, 1).to(self.device))
        tok = self.tok_embedding(src_tokens)                    #tok = pos = [batch size, src len, emb dim]

        x = self.dropout(tok + pos)                             #x = [batch size, src len, emb dim]

        conv_input = self.emb2hid(x)                            #conv_input = [batch size, src len, hid dim]
        conv_input = conv_input.permute(0, 2, 1)                #conv_input = [batch size, hid dim, src len]
        
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))             #conved = [batch size, 2 * hid dim, src len]
            conved = F.glu(conved, dim = 1)                     #conved = [batch size, hid dim, src len]
            conved = (conved + conv_input) * self.scale         #conved = [batch size, hid dim, src len]
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))          #conved = [batch size, src len, emb dim]
        combined = (conved + x) * self.scale             #combined = [batch size, src len, emb dim]
        
        return conved, combined

class ConvDecoder(nn.Module):
    """
    inspired by
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb
    """
    def __init__(self, 
                 output_dim, 
                 emb_dim, 
                 hid_dim, 
                 n_layers, 
                 kernel_size, 
                 dropout_prob, 
                 device,
                 padding_idx, 
                 self_attention_heads=1,
                 max_length=1500, 
                 max_sample_length=150,
                 tok_embedding=None, 
                 pos_embedding=None, 
                 hid2emb=None):
        super(ConvDecoder, self).__init__()
        
        self.kernel_size = kernel_size
        self.max_sample_length = max_sample_length
        self.padding_idx = padding_idx
        self.hid_dim = hid_dim
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        if tok_embedding is None and pos_embedding is None:
            self.tok_embedding = nn.Embedding(output_dim, emb_dim)
            nn.init.normal_(self.tok_embedding.weight, 0, 0.1)
            nn.init.constant_(self.tok_embedding.weight[padding_idx], 0)

            self.pos_embedding = nn.Embedding(max_length, emb_dim)
            nn.init.normal_(self.pos_embedding.weight, 0, 0.1)
            nn.init.constant_(self.pos_embedding.weight[padding_idx], 0)
            self.hid2emb = nn.Linear(hid_dim, emb_dim)
        else:
            self.tok_embedding = tok_embedding
            self.pos_embedding = pos_embedding
            self.hid2emb = hid2emb

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size)
                                    for i in range(n_layers)])
        
        
        if self_attention_heads != 0:
            self.attns = nn.ModuleList([Attention(hid_dim, emb_dim, self_attention=False, device=self.device) if i % 2 == 0 else SelfAttention(hid_dim, emb_dim, self.device)
                                    for i in range(n_layers)])
        else:
            self.attns = nn.ModuleList([Attention(hid_dim, emb_dim, self_attention=False, device=self.device) if i % 2 == 0 else None
                                    for i in range(n_layers)])

        self.fc_out = nn.Linear(emb_dim, output_dim)
        
        
        
    def forward(self, trg_tokens, encoder_conved, encoder_combined, inference=False):
        
        batch_size = trg_tokens.shape[0]
        pos = self.pos_embedding(torch.arange(0, trg_tokens.shape[1]).unsqueeze(0).repeat(batch_size, 1).to(self.device))
        tok = self.tok_embedding(trg_tokens)                        #tok = pos = [batch size, trg len, emb dim]
        x = self.dropout(tok + pos)                                 #x = [batch size, trg len, emb dim]
        conv_input = self.emb2hid(x)                            #conv_input = [batch size, trg len, hid dim]
        conv_input = conv_input.permute(0, 2, 1)                    #conv_input = [batch size, hid dim, trg len]
        
        for i, conv in enumerate(self.convs):
            attn = self.attns[i]
            conv_input = self.dropout(conv_input)            
            padding = torch.zeros(batch_size, 
                                      self.hid_dim, 
                                      self.kernel_size - 1).fill_(self.padding_idx).to(self.device)   
            padded_conv_input = torch.cat((padding, 
                                        conv_input), dim = 2)           #padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]
            conved = conv(padded_conv_input)                        #conved = [batch size, 2 * hid dim, trg len]
            conved = F.glu(conved, dim = 1)                         #conved = [batch size, hid dim, trg len]
            if i % 2 == 0:
                attention, conved = attn(conved,
                                    encoder_conved, 
                                    encoder_combined,
                                    x, self.scale)                              #attention = [batch size, trg len, src len]            
            else:
                if attn is not None:
                    conved = attn(conved)
                conved = conved.permute(0, 2, 1) * self.scale
                
            conved = (conved + conv_input) * self.scale             #conved = [batch size, hid dim, trg len]
            conv_input = conved
            
        conved = self.hid2emb(conved.permute(0, 2, 1))              #conved = [batch size, trg len, emb dim]
            
        output = self.fc_out(self.dropout(conved))                  #output = [batch size, trg len, output dim]
            
        return output, attention

    def forward_sample(self, encoder_conved, encoder_combined, sos_idx, eos_idx, sample=False, greedy=False):
        batch_size = encoder_conved.shape[0]
        trg_tokens = torch.LongTensor([[sos_idx] for j in range(batch_size)]).to(self.device) 
        batch_complete = [False for b in range(batch_size)]

        for i in range(self.max_sample_length):
            #tok = pos = [batch size, trg len, emb dim]
            pos = self.pos_embedding(torch.arange(i, i+1).unsqueeze(0).repeat(batch_size, 1).to(self.device))
            tok = self.tok_embedding(trg_tokens[:,-1].unsqueeze(1))   
            if i != 0:
                pos = torch.cat((previous_pos, pos), dim=1)
                tok = torch.cat((previous_tok, tok), dim=1)

            x = self.dropout(tok + pos)                                 #x = [batch size, trg len, emb dim]
            conv_input = self.emb2hid(x)                            #conv_input = [batch size, trg len, hid dim]
            conv_input = conv_input.permute(0, 2, 1)                    #conv_input = [batch size, hid dim, trg len]
            
            for i, conv in enumerate(self.convs):
                attn = self.attns[i]
                conv_input = self.dropout(conv_input)            
                padding = torch.zeros(batch_size, 
                                          self.hid_dim, 
                                          self.kernel_size - 1).fill_(self.padding_idx).to(self.device)   
                padded_conv_input = torch.cat((padding, 
                                            conv_input), dim = 2)           #padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]

                conved = conv(padded_conv_input)                        #conved = [batch size, 2 * hid dim, trg len]
                conved = F.glu(conved, dim = 1)                         #conved = [batch size, hid dim, trg len]
                if i % 2 == 0:
                    attention, conved = attn(conved,
                                        encoder_conved, 
                                        encoder_combined,
                                        x, self.scale)                              #attention = [batch size, trg len, src len]            
                else:
                    if attn is not None:
                        conved = attn(conved)
                        conved = conved.permute(0, 2, 1) * self.scale

                conved = (conved + conv_input) * self.scale             #conved = [batch size, hid dim, trg len]
                conv_input = conved
                
            conved = self.hid2emb(conved.permute(0, 2, 1))              #conved = [batch size, trg len, emb dim]
                
            output = self.fc_out(self.dropout(conved))                  #output = [batch size, trg len, output dim]

            if sample:
                out_tokens = torch.multinomial(F.softmax(output[:, -1, :], dim=1), 1) 
            elif greedy:
                out_tokens = torch.argmax(output, dim=2)[:, -1].unsqueeze(1)

            trg_tokens = torch.cat((trg_tokens, out_tokens), dim=1)
            
            previous_pos = pos
            previous_tok = tok

            for j, ind in enumerate(out_tokens):
                if ind.tolist()[0] == eos_idx:
                    batch_complete[j] = True
            if sum(batch_complete) == len(batch_complete):
                print('stop early')
                break
            
        return output, trg_tokens        

class Attention(nn.Module):
    def __init__(self, out_channels, emb_dim, self_attention=False, device='cuda'):
        super(Attention, self).__init__()
        self.self_attention = self_attention
        self.device = device
        if not self_attention:
            self.attn_hid2emb = nn.Linear(out_channels, emb_dim)
            self.attn_emb2hid = nn.Linear(emb_dim, out_channels)

    def get_mask(self, trg_len):
        mask = (torch.tril(torch.ones(trg_len, trg_len)) == 1).to(self.device)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, conved, encoder_conved, encoder_combined, x=None, scale=None):
        

        if self.self_attention:
            conved_combined = conved           
        else:
            conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))     #conved_emb = [batch size, trg len, emb dim]
            conved_combined = (conved_emb + x) * scale             #combined = [batch size, trg len, emb dim]
            

        encoder_conved = encoder_conved.permute(0, 2, 1)    
        energy = torch.matmul(conved_combined,
                        encoder_conved)            #energy = [batch size, trg len, src len]

        if self.self_attention:        
            mask = self.get_mask(energy.shape[2])
            energy = energy + mask

        attention = F.softmax(energy, dim=2)                        #attention = [batch size, trg len, src len]
        attended_encoding = torch.matmul(attention, 
                        encoder_combined)                           #attended_encoding = [batch size, trg len, emd dim]
        if self.self_attention:
            return attended_encoding
        else:
            attended_encoding = self.attn_emb2hid(attended_encoding)    #attended_encoding = [batch size, trg len, hid dim]
            attended_combined = (conved + attended_encoding.permute(0, 2, 1)) #attended_combined = [batch size, hid dim, trg len]
            return attention, attended_combined
            
        
        

class SelfAttention(nn.Module):
    """
    inspired by
    https://github.com/pytorch/fairseq/tree/master/fairseq/models/fconv_self_att.py
    """
    def __init__(self, out_channels, embed_dim, device):
        super(SelfAttention, self).__init__()
        self.attention = Attention(out_channels, embed_dim, self_attention=True, device=device)
        self.in_proj_q = nn.Linear(out_channels, embed_dim)
        self.in_proj_k = nn.Linear(out_channels, embed_dim)
        self.in_proj_v = nn.Linear(out_channels, embed_dim)
        self.s_attn_emb2hid = nn.Linear(embed_dim, out_channels)
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x):
        residual = x.permute(0, 2, 1)
        query = self.in_proj_q(residual)
        key = self.in_proj_k(residual)
        value = self.in_proj_v(residual)
        encoding = self.attention(query, key, value)
        x_attn = self.s_attn_emb2hid(encoding)
        return self.ln(x_attn + residual)
