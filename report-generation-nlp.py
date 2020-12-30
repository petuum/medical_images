import dgl
import dgl.function as fn
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
import copy
import torch.optim as optim
import pickle
from torch.distributions import Categorical



# Texar Library
import texar

from texar.torch.core.cell_wrappers import RNNCell, LSTMCell

from texar.torch.modules import UnidirectionalRNNEncoder, WordEmbedder, BasicRNNDecoder

from texar.torch import HParams

from texar.torch.losses import sequence_sparse_softmax_cross_entropy

from texar.torch.core import BahdanauAttention


# Co-Attention in Pytorch 

# It takes in as input the V visual features and the a semantic features along with the hidden vector of the 
# previous time step of the sentence LSTM layer 1 in the 2 layer hierarchical LSTM

class Co_Attention(nn.Module):
    
    def __init__(self, num_units, visual_units, semantic_units, hidden_size, N, M, batch_size = 1):
        
        super(Co_Attention, self).__init__()
        
        # initialise the variables
        
        # assuming hidden state and input to lstm have the same dimension
        # the hidden vector and input size of the sentence LSTM
        self.hidden_size = hidden_size        
        self.batch_size = batch_size
        
        # attention parameters
        self.num_units = num_units# intermediate number of nodes for the BahdanauAttention attention  calculation       
        
        # dimension of visual and semantic features
        self.visual_units = visual_units        
        self.semantic_units = semantic_units
        
        # As per the On the Automatic Generation of Medical Imaging Reports paper
        self.N = N # Number of Visual features
        self.M = M # Number of Semantic features
        
        # The attention layer
        
        self.visual_attn = BahdanauAttention(self.num_units, decoder_output_size = self.hidden_size, 
                                             encoder_output_size = self.visual_units)
        
        self.semantic_attn = BahdanauAttention(self.num_units, decoder_output_size = self.hidden_size, 
                                               encoder_output_size = self.semantic_units)
        
        # Context calculation layer
        
        self.layer = nn.Linear(self.N + self.M, self.hidden_size)
        
    def forward(self, v, a, hidden_state):
        
        # v has dimension [Batch size, max_time_steps = N_v, hidden_state = visual_units]
        
        # a has dimension [Batch size, max_time_steps = N_a, hidden_state = semantic_units]
        
        state_v = torch.rand(self.batch_size,self.N)
        
        visual_alignments, _ = visual_attn(hidden_state, state = state_v, memory = v)

        state_a = torch.rand(self.batch_size,self.M)
        
        semantic_alignments, _ = semantic_attn(hidden_state, state = state_a, memory = a)
        
        # v_attn has dimension [batch size, N_v]
        
        # a_attn has dimension [batch size, N_a]
        
        v_attn = torch.bmm(visual_alignments.view(self.batch_size,1,-1), v).squeeze(1)
        
        a_attn = torch.bmm(semantic_alignments.view(self.batch_size,1,-1), a).squeeze(1)
        
        cat_attn = torch.cat([v_attn.view(self.batch_size, -1), a_attn.view(self.batch_size, -1)], 1)
        
        ctx = self.layer(cat_attn)
        
        return ctx, visual_alignments, semantic_alignments


### 2 layered Hierarchical LSTM in TEXAR-TORCH

# This is an implementation of 2 layered hierarchical LSTM implementation in Texar. In this particular 
# application the first tier takes in the input which here is from co-attention and outputs a hidden state vector. 
# At each time step of the tier-1 LSTM, the hidden state vector is fed into the tier 2 LSTM as a state tuple 
# to output a sentence.  

    
# ---------------------- Hierarchical LSTM ----------------------------

# NOTE: Run the LSTM sentence in a for loop range [0 max time steps] till termination
# Because at each time step we calculate the visual and semantic attention (stack at the end of the max 
# timesteps to find the alpha and beta)
# Output the t needed for the Word lstm 
# At each time step we produce a 0 or 1 to continue or stop

# This is run at every time step of Sentence LSTM as stated above
# It is Bernoulli variable
# p_pred shape [Batch size, 2]
# p_target shape [Batch size]


#################################################################
# Sentence generation and Loss for sentence generation
#################################################################
def sentence_loss(p_pred, p_target):
    
    sentence_lossfn = nn.CrossEntropyLoss()
    
    return sentence_lossfn(p_pred, p_target)

# 1st in the hierarchy
class LSTM_sentence(nn.Module):
    # hidden_size the same as GNN output
    def __init__(self, hidden_size, num_units, visual_units, semantic_units, N, M, seq_len = 1, batch_size = 1):
        super(LSTM_sentence, self).__init__()
        
        # initialise the variables

        # Here the input size is equal to the hidden size used as the input from co-attention
        self.input_size = hidden_size
        
        # the output size
        # LSTM parameters
        # the hidden vector and input size of the sentence LSTM
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size

        # attention parameters
        self.num_units = num_units# intermediate number of nodes for the BahdanauAttention attention  calculation       
        # dimension of visual and semantic features
        self.visual_units = visual_units        
        self.semantic_units = semantic_units
        
        # As per the On the Automatic Generation of Medical Imaging Reports paper
        self.N = N # Number of Visual features
        self.M = M # Number of Semantic features

        # The Co_Attention module
        # observe that the input to the LSTM and hidden vector have the same dimension
        # If not add a new parameter and make changes accordingly
        self.co_attn = Co_Attention(self.num_units, self.visual_units, self.semantic_units, 
        	self.hidden_size, self.N, self.M, batch_size = self.batch_size)
        
        # LSTM layer -- batch_first = true
        
        enc_hparams = {'rnn_cell': {'type': 'LSTMCell', 'kwargs': {'num_units': self.hidden_size}}}
        
        default_hparams = UnidirectionalRNNEncoder.default_hparams()

        hparams_ = HParams(enc_hparams, default_hparams)
        
        self.lstm = UnidirectionalRNNEncoder(input_size=self.input_size, hparams=hparams_.todict())

    def forward(self, v, a, hidden):
        
        (h_0, c_0) = hidden

        inp_lstm, visual_alignments, semantic_alignments = self.co_attn(v, a, h_0)
        
        output, hidden = self.lstm(inp_lstm, initial_state=
                                   (h_0.view(self.batch_size, self.hidden_size), 
                                    c_0.view(self.batch_size, self.hidden_size)))
        

        # return the visual_alignments, semantic_alignments for the loss function calculation
        # stack the visual_alignments, semantic_alignments at each time step of the sentence LSTM to 
        # obtain the alpha (visual_alignments) beta (semantic_alignments)
        return output, hidden, visual_alignments, semantic_alignments 

    def initHidden(self):
        return (torch.zeros(self.seq_len, self.batch_size, self.hidden_size), 
                torch.zeros(self.seq_len, self.batch_size, self.hidden_size))


# 2 nd in the hierarchy
class LSTM_word(nn.Module):
    # hidden_size the same as GNN output
    def __init__(self, hidden_size, output_size, seq_len = 1, batch_size = 1):
        super(LSTM_word, self).__init__()
        
        # initialise the variables

        # Here the input size is equal to the hidden size used as the input from co-attention
        self.input_size = hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        # Embedding layer
        
        self.embedding = WordEmbedder(vocab_size=self.output_size, hparams={'dim': self.hidden_size})
        
        # LSTM layer -- batch_first = true
        
        enc_hparams = {'rnn_cell': {'type': 'LSTMCell', 'kwargs': {'num_units': self.hidden_size}}}
        
        default_hparams = BasicRNNDecoder.default_hparams()

        hparams_ = HParams(enc_hparams, default_hparams)
        
        self.decoder = BasicRNNDecoder(input_size = self.hidden_size, token_embedder = self.embedding, 
                                    vocab_size=self.output_size, hparams=hparams_.todict())

    def forward(self, hidden, train, inp= None, sentence_len = None):
        
        if train:
            (output, final_state, sequence_lengths) = self.decoder( decoding_strategy='train_greedy', inputs=inp, sequence_length=sentence_len, initial_state = hidden)
        else:
            # Create helper
            helper = self.decoder.create_helper(decoding_strategy='infer_greedy',start_tokens=torch.Tensor([CLS_token]*self.batch_size).long(),
                                           end_token=SEP_token, embedding=self.embedding)

            # Inference sample
            # here sentence length is the max_decoding_length
            output, _, _ = self.decoder(helper=helper, initial_state = hidden, max_decoding_length=sentence_len)
        
        return output

    def initHidden(self):
        return (torch.zeros(self.seq_len, self.batch_size, self.hidden_size), 
                torch.zeros(self.seq_len, self.batch_size, self.hidden_size))



###################################################
# Phrase generation and Loss for phrase generation
###################################################
# here the phrase_decoder is an instance of the class LSTM_word
# This is a function that is executed at every time step S of sentence LSTM
# new_node_embedding is the hidden statevector of time step S from sentence LSTM
def generate_phrase(phrase_decoder, new_node_embedding, index = None, train = True):
    
    phrase_loss = 0
    
    if train:
        phrase_decoder_input = torch.Tensor(tokenizer(index2phrase[index])['input_ids']).view(batch_size,-1).long().to(device)
    
        sentence_len = torch.Tensor([len(phrase_decoder_input[0,1:])]).long()
    
    state = new_node_embedding
    
    (state_h_0, state_c_0) = state

    # 1 because the number of stacked lstm layers are 1 pytorch input specification

    state_h_0 = (state_h_0).view(batch_size, hidden_size)

    state_c_0 = (state_c_0).view(batch_size, hidden_size)

    phrase_decoder_hidden = (state_h_0, state_c_0)
    
    if train:
        output = phrase_decoder(phrase_decoder_hidden, train, inp = phrase_decoder_input, 
                                sentence_len = sentence_len)
    
        # per time step S of sentence LSTM
        phrase_loss = sequence_sparse_softmax_cross_entropy( labels=phrase_decoder_input[:, 1:], logits=output.logits, 
                                                 sequence_length=sentence_len)
    else:
        
        output = phrase_decoder(phrase_decoder_hidden, train, sentence_len = 5)
        
    
    predict = list(np.array(output.sample_id.view(-1)))
    
    if train:
        return phrase_loss, tokenizer.decode(predict), phrase_decoder_input
    else:
        return phrase_loss, tokenizer.decode(predict)

#################################################
# Loss for attention and tags
#################################################

# Apply the weighting lambdas in the main function this is just a loss without lambda weights
def tag_loss(p_pred, p_target):
    # p_target is [Batch_size, hidden]   
    # p_pred   is [Batch_size, hidden]
    # assuming p_target is a normalised vector of distribution l/||l||_1 of tags
    # assuming is the output of the  multi-label classification layer without the softmax we apply softmax here
    # pl,pred is before the application of softmax layer
    # We obtain the predicted distriubtion here
    logsoftmax = nn.LogSoftmax(dim=1)
    
    return torch.mean(torch.sum(- p_target * logsoftmax(p_pred), 1))

# Attention loss 
# weigh it in the main loop as per lambda
def attn_loss(alpha, beta):
    # alpha is [Batch_size, N, S]   
    # beta is [Batch_size, M, S]
    # N is the number of visual features, M is the number of semantic features
    # S is the number of time steps in Sentence LSTM
    visual_attn_loss   = torch.sum(((1 - torch.sum(alpha, -1))**2), -1) 
    
    semantic_attn_loss = torch.sum(((1 - torch.sum( beta, -1))**2), -1) 

    return torch.mean(visual_attn_loss, semantic_attn_loss)


    

