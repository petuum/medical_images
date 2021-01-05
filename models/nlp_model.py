# Copyright 2020 Petuum Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model classes for nlp models in medical report generation
"""

import torch
import torch.nn as nn

# Texar Library
import texar
from texar.torch.modules import UnidirectionalRNNEncoder, WordEmbedder, BasicRNNDecoder
from texar.torch import HParams
from texar.torch.core import BahdanauAttention


# Co-Attention in Pytorch
class CoAttention(nn.Module):
    """
    It takes in as input the V visual features and the a semantic features along with the hidden
    vector of the previous time step of the sentence LSTM layer 1 in the 2 layer hierarchical LSTM
    """
    def __init__(self, num_units, visual_units, semantic_units, hidden_size, N, M, batch_size=1):
        """ Initialize function

        Args:
            num_units (int): intermediate number of nodes for the BahdanauAttention
            attention calculation
            visual_units (int): Dimension of visual unit
            semantic_units (int): Dimension of semantic unit
            hidden_size (int): assuming hidden state and input to lstm have the same dimension,
            the hidden vector and input size of the sentence LSTM
            N (int): Number of Visual features
            M (int): Number of Semantic features
            batch_size (int): Batch size
        """
        super(CoAttention, self).__init__()

        # initialise the variables
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # attention parameters
        self.num_units = num_units

        self.visual_units = visual_units
        self.semantic_units = semantic_units

        # As per the On the Automatic Generation of Medical Imaging Reports paper
        self.num_visual = N  # Number of Visual features
        self.num_semantic = M  # Number of Semantic features

        # The attention layer
        self.visual_attn = BahdanauAttention(self.num_units, decoder_output_size=self.hidden_size,
                                             encoder_output_size=self.visual_units)

        self.semantic_attn = BahdanauAttention(self.num_units, decoder_output_size=self.hidden_size,
                                               encoder_output_size=self.semantic_units)

        # Context calculation layer
        self.layer = nn.Linear(self.num_visual + self.num_semantic, self.hidden_size)

    def forward(self, v, a, hidden_state):
        """

        Args:
            v (torch.Tensor): Dimension [Batch size, max_time_steps = N_v, hidden_state =
            visual_units]
            a (torch.Tensor): Dimension [Batch size, max_time_steps = N_a, hidden_state =
            semantic_units]
            hidden_state ():

        Returns:
            ctx (torch.Tensor):
            visual_alignments (torch.Tensor):
            senmantic_alignments (torch.Tensor):
        """
        state_v = torch.rand(self.batch_size, self.num_visual)

        visual_alignments, _ = self.visual_attn(hidden_state, state=state_v, memory=v)

        state_a = torch.rand(self.batch_size, self.num_semantic)

        semantic_alignments, _ = self.semantic_attn(hidden_state, state=state_a, memory=a)

        # v_attn has dimension [batch size, N_v]

        # a_attn has dimension [batch size, N_a]

        v_attn = torch.bmm(visual_alignments.view(self.batch_size, 1, -1), v).squeeze(1)

        a_attn = torch.bmm(semantic_alignments.view(self.batch_size, 1, -1), a).squeeze(1)

        cat_attn = torch.cat(
            [v_attn.view(self.batch_size, -1),
             a_attn.view(self.batch_size, -1)],
            1)

        ctx = self.layer(cat_attn)

        return ctx, visual_alignments, semantic_alignments


### 2 layered Hierarchical LSTM in TEXAR-TORCH

# This is an implementation of 2 layered hierarchical LSTM implementation in Texar. In this
# particular application the first tier takes in the input which here is from co-attention and
# outputs a hidden state vector. At each time step of the tier-1 LSTM, the hidden state vector is
# fed into the tier 2 LSTM as a state tuple to output a sentence.

# ---------------------- Hierarchical LSTM ----------------------------

# NOTE: Run the LSTM sentence in a for loop range [0 max time steps] till termination
# Because at each time step we calculate the visual and semantic attention (stack at the end of the
# max timesteps to find the alpha and beta)
# Output the t needed for the Word lstm
# At each time step we produce a 0 or 1 to continue or stop

# This is run at every time step of Sentence LSTM as stated above
# It is Bernoulli variable
# p_pred shape [Batch size, 2]
# p_target shape [Batch size]


#################################################################
# Sentence generation and Loss for sentence generation
#################################################################

class LSTMSentence(nn.Module):
    """

    """
    def __init__(self, hidden_size, num_units, visual_units, semantic_units, N, M, seq_len=1,
                 batch_size=1):
        """1st in thehierarchy

        Args:
            hidden_size (int): Hidden_size the same as GNN output
            num_units (int): Intermediate number of nodes for the BahdanauAttention attention
            calculation
            visual_units (int): Dimension of visual features
            semantic_units (int): Dimension of semantic features
            N (int): Number of visual features
            M (int): Number of semantic features
            seq_len (int): Sequence length for sentence LSTM
            batch_size (int): Batch size
        """
        super(LSTMSentence, self).__init__()

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
        self.num_units = num_units

        # dimension of visual and semantic features
        self.visual_units = visual_units
        self.semantic_units = semantic_units

        # As per the On the Automatic Generation of Medical Imaging Reports paper
        self.N = N  # Number of Visual features
        self.M = M  # Number of Semantic features

        # The Co_Attention module
        # observe that the input to the LSTM and hidden vector have the same dimension
        # If not add a new parameter and make changes accordingly
        self.co_attn = CoAttention(self.num_units, self.visual_units, self.semantic_units,
                                    self.hidden_size, self.N, self.M, batch_size=self.batch_size)

        # LSTM layer -- batch_first = true

        enc_hparams = {'rnn_cell': {'type': 'LSTMCell', 'kwargs': {'num_units': self.hidden_size}}}

        default_hparams = UnidirectionalRNNEncoder.default_hparams()

        hparams_ = HParams(enc_hparams, default_hparams)

        self.lstm = UnidirectionalRNNEncoder(input_size=self.input_size, hparams=hparams_.todict())

    def forward(self, v, a, hidden):
        """

        Args:
            v (torch.Tensor): Visual features
            a (torch.Tensor): Average features
            hidden ():

        Returns:

        """
        (h_0, c_0) = hidden

        inp_lstm, visual_alignments, semantic_alignments = self.co_attn(v, a, h_0)

        output, hidden = self.lstm(inp_lstm, initial_state=
        (h_0.view(self.batch_size, self.hidden_size),
         c_0.view(self.batch_size, self.hidden_size)))

        # return the visual_alignments, semantic_alignments for the loss function calculation
        # stack the visual_alignments, semantic_alignments at each time step of the sentence LSTM to
        # obtain the alpha (visual_alignments) beta (semantic_alignments)
        return output, hidden, visual_alignments, semantic_alignments

    def init_hidden(self):
        return (torch.zeros(self.seq_len, self.batch_size, self.hidden_size),
                torch.zeros(self.seq_len, self.batch_size, self.hidden_size))


# 2 nd in the hierarchy
class LSTMWord(nn.Module):
    # hidden_size the same as GNN output
    CLS_token = 1
    SEP_token = 2
    def __init__(self, hidden_size, output_size, seq_len=1, batch_size=1):
        """

        Args:
            hidden_size (int): hidden_size the same as GNN output
            output_size (int):
            seq_len (int): sequence length for word lstm
            batch_size (int): batch size
        """
        super(LSTMWord, self).__init__()

        # initialise the variables

        # Here the input size is equal to the hidden size used as the input from co-attention
        self.input_size = hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.batch_size = batch_size

        # Embedding layer
        self.embedding = \
            WordEmbedder(vocab_size=self.output_size, hparams={'dim': self.hidden_size})

        # LSTM layer -- batch_first = true

        enc_hparams = {'rnn_cell': {'type': 'LSTMCell', 'kwargs': {'num_units': self.hidden_size}}}

        default_hparams = BasicRNNDecoder.default_hparams()

        hparams_ = HParams(enc_hparams, default_hparams)

        self.decoder = BasicRNNDecoder(input_size=self.hidden_size, token_embedder=self.embedding,
                                       vocab_size=self.output_size, hparams=hparams_.todict())

    def forward(self, hidden, train, inp=None, sentence_len=None):
        """

        Args:
            hidden ():
            train ():
            inp ():
            sentence_len ():

        Returns:

        """

        if train:
            (output, final_state, sequence_lengths) = self.decoder(
                decoding_strategy='train_greedy',
                inputs=inp,
                sequence_length=sentence_len,
                initial_state=hidden)
        else:
            # Create helper
            helper = self.decoder.create_helper(
                decoding_strategy='infer_greedy',
                start_tokens=torch.Tensor([self.CLS_token] * self.batch_size).long(),
                end_token=self.SEP_token,
                embedding=self.embedding)

            # Inference sample
            # here sentence length is the max_decoding_length
            output, _, _ = self.decoder(
                helper=helper,
                initial_state=hidden,
                max_decoding_length=sentence_len)

        return output

    def init_hidden(self):
        return (torch.zeros(self.seq_len, self.batch_size, self.hidden_size),
                torch.zeros(self.seq_len, self.batch_size, self.hidden_size))
