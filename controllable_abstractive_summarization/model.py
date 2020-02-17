# import fairseq
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# controllable abstractive summarization
# conv seq2seq with intra-attention and BPE encoding for word embeddings

# https://pypi.org/project/bpemb/
# https://fairseq.readthedocs.io/en/latest/
# https://arxiv.org/abs/1705.03122
# https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb





class ControllableSummarizer(nn.Module):

    def __init__(self, input_dim, output_dim, emb_dim, hid_dim, n_layers, 
                        kernel_size, dropout_prob, device, padding_idx, share_weights=False, max_length=1500):
        super(ControllableSummarizer, self).__init__()
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

        self.encoder = ConvEncoder(input_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout_prob, device, padding_idx, self.tok_embedding, self.pos_embedding, self.hid2emb)
        self.decoder = ConvDecoder(output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout_prob, device, padding_idx, self.tok_embedding, self.pos_embedding, self.hid2emb)

    def forward(self, src_tokens, trg_tokens):
        # print(src_tokens.shape)
        # print(trg_tokens.shape)
        conved, combined = self.encoder(src_tokens)
        # print(conved.shape)
        # print(combined.shape)
        output, attention = self.decoder(trg_tokens, conved, combined)
        return output, attention



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
        
        for conv in self.convs:
            std = math.sqrt((4 * (1.0 - dropout_prob)) / (conv.kernel_size[0] * hid_dim))
            conv.weight.data.normal_(mean=0, std=std)
            conv.bias.data.zero_()    

    def forward(self, src_tokens):
        
        src_lengths = src_tokens.shape[1]
        batch_size = src_tokens.shape[0]

        pos = self.pos_embedding(torch.arange(0, src_lengths).unsqueeze(0).repeat(batch_size, 1).to(self.device))
        tok = self.tok_embedding(src_tokens)                    #tok = pos = [batch size, src len, emb dim]

        x = self.dropout(tok + pos)                             #x = [batch size, src len, emb dim]

        #pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(x)                            #conv_input = [batch size, src len, hid dim]

        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)                #conv_input = [batch size, hid dim, src len]
        
        for i, conv in enumerate(self.convs):
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))             #conved = [batch size, 2 * hid dim, src len]

            #pass through GLU activation function
            conved = F.glu(conved, dim = 1)                     #conved = [batch size, hid dim, src len]

            #apply residual connection
            conved = (conved + conv_input) * self.scale         #conved = [batch size, hid dim, src len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        #permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))          #conved = [batch size, src len, emb dim]
        
        #elementwise sum output (conved) and input (embedded) to be used for attention
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
                 tok_embedding=None, 
                 pos_embedding=None, 
                 hid2emb=None):
        super(ConvDecoder, self).__init__()
        
        self.kernel_size = kernel_size
        self.padding_idx = padding_idx
        self.hid_dim = hid_dim
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        if tok_embedding is None and pos_embedding is None:
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
        self.dropout = nn.Dropout(dropout_prob)

        
        self.attention = Attention(hid_dim, emb_dim)
        self.self_attention = SelfAttention(hid_dim, emb_dim)

        self.fc_out = nn.Linear(emb_dim, output_dim)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size)
                                    for _ in range(n_layers)])
        
    def forward(self, trg_tokens, encoder_conved, encoder_combined):
        
        #trg_tokens = [batch size, trg len]
        #encoder_conved = encoder_combined = [batch size, src len, emb dim]
                
        batch_size = trg_tokens.shape[0]
            
        #create position tensor
        pos = self.pos_embedding(torch.arange(0, trg_tokens.shape[1]).unsqueeze(0).repeat(batch_size, 1).to(self.device))
        tok = self.tok_embedding(trg_tokens)                        #tok = pos = [batch size, trg len, emb dim]
        
        #combine embeddings by elementwise summing
        x = self.dropout(tok + pos)                                 #x = [batch size, trg len, emb dim]
        
        #pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(x)                            #conv_input = [batch size, trg len, hid dim]
        
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)                    #conv_input = [batch size, hid dim, trg len]
        
        for i, conv in enumerate(self.convs):
            #apply dropout
            conv_input = self.dropout(conv_input)
        
            #need to pad so decoder can't "cheat"
            padding = torch.zeros(batch_size, 
                                  self.hid_dim, 
                                  self.kernel_size - 1).fill_(self.padding_idx).to(self.device)                
            padded_conv_input = torch.cat((padding, 
                                    conv_input), dim = 2)           #padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]
        
            #pass through convolutional layer
            conved = conv(padded_conv_input)                        #conved = [batch size, 2 * hid dim, trg len]
            
            #pass through GLU activation function
            conved = F.glu(conved, dim = 1)                         #conved = [batch size, hid dim, trg len]
            
            #calculate attention
            attention, conved = self.attention(conved,
                                    encoder_conved, 
                                    encoder_combined,
                                    x, self.scale)                              #attention = [batch size, trg len, src len]
            
            _, conved = self.self_attention(conved)

            #apply residual connection
            conved = (conved.permute(0, 2, 1) + conv_input) * self.scale             #conved = [batch size, hid dim, trg len]
            
            #set conv_input to conved for next loop iteration
            conv_input = conved
            
        conved = self.hid2emb(conved.permute(0, 2, 1))              #conved = [batch size, trg len, emb dim]
            
        output = self.fc_out(self.dropout(conved))                  #output = [batch size, trg len, output dim]
            
        return output, attention

class Attention(nn.Module):
    def __init__(self, out_channels, emb_dim, self_attention=False):
        super(Attention, self).__init__()
        
        if not self_attention:
            self.attn_hid2emb = nn.Linear(out_channels, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, out_channels)

    def forward(self, conved, encoder_conved, encoder_combined, x=None, scale=None):
        
        #embedded = [batch size, trg len, emb dim]
        #conved = [batch size, hid dim, trg len]
        #encoder_conved = encoder_combined = [batch size, src len, emb dim]
        
        #permute and convert back to emb dim
        if x is None:
            conved_combined = conved
        else:
            conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))     #conved_emb = [batch size, trg len, emb dim]
                
            conved_combined = (conved_emb + x) * scale             #combined = [batch size, trg len, emb dim]
            
                
        energy = torch.matmul(conved_combined,
                        encoder_conved.permute(0, 2, 1))            #energy = [batch size, trg len, src len]
        
        attention = F.softmax(energy, dim=2)                        #attention = [batch size, trg len, src len]
            
        attended_encoding = torch.matmul(attention, 
                        encoder_combined)                           #attended_encoding = [batch size, trg len, emd dim]
        
        #convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)    #attended_encoding = [batch size, trg len, hid dim]
        
        #apply residual connection
        if x is not None:
            attended_combined = (conved + attended_encoding.permute(0, 2, 1)) 
            #attended_combined = [batch size, hid dim, trg len]
        else:
            attended_combined = attended_encoding
        
        return attention, attended_combined

class SelfAttention(nn.Module):
    """
    inspired by
    https://github.com/pytorch/fairseq/tree/master/fairseq/models/fconv_self_att.py
    """
    def __init__(self, out_channels, embed_dim):
        super(SelfAttention, self).__init__()
        self.attention = Attention(out_channels, embed_dim, self_attention=True)
        self.in_proj_q = nn.Linear(out_channels, embed_dim)
        self.in_proj_k = nn.Linear(out_channels, embed_dim)
        self.in_proj_v = nn.Linear(out_channels, embed_dim)
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x):
        residual = x.permute(0, 2, 1)
        query = self.in_proj_q(residual)
        key = self.in_proj_k(residual)
        value = self.in_proj_v(residual)
        attn, x_attn = self.attention(query, key, value)
        return attn, self.ln(x_attn + residual)
