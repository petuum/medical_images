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
from typing import Dict, Any

import torch
import torch.nn as nn

# Texar Library
from texar.torch.modules import UnidirectionalRNNEncoder, WordEmbedder, BasicRNNDecoder
from texar.torch import HParams
from texar.torch.core import BahdanauAttention
from texar.torch import ModuleBase


CLS_TOKEN = 0
SEP_TOKEN = 1


class CoAttention(ModuleBase):
    r"""It takes in as input the V visual features and the a semantic features along with the hidden
    vector of the previous time step of the sentence LSTM layer 1 in the 2 layer hierarchical LSTM

    Args:
        hparams (dict or HParams, optional): LSTMSentence hyperparameters. Missing
            hyperparameters will be set to default values. See :meth:`default_hparams` for the
            hyperparameter structure and default values.
                * num_units (int): intermediate number of nodes for the BahdanauAttention
                attention calculation
                * visual_units (int): Dimension of visual unit
                * semantic_units (int): Dimension of semantic unit
                * hidden_size (int): Assuming hidden state and input to lstm have the same
                dimension, the hidden vector and input size of the sentence LSTM
                * num_visual (int): Number of Visual features
                * num_semantic (int): Number of Semantic features
                * batch_size (int): Batch size
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        hidden_size = self.hparams.hidden_size
        self.batch_size = self.hparams.batch_size

        # Attention parameters
        num_units = self.hparams.num_units

        visual_units = self.hparams.visual_units
        semantic_units = self.hparams.semantic_units

        # As per the On the Automatic Generation of Medical Imaging Reports paper
        self.num_visual = self.hparams.num_visual  # Number of Visual features
        self.num_semantic = self.hparams.num_semantic  # Number of Semantic features

        # The attention layer
        self.visual_attn = BahdanauAttention(
            num_units,
            decoder_output_size=hidden_size,
            encoder_output_size=visual_units)

        self.semantic_attn = BahdanauAttention(
            num_units,
            decoder_output_size=hidden_size,
            encoder_output_size=semantic_units)

        # Context calculation layer
        self.layer = nn.Linear(self.num_visual + self.num_semantic, hidden_size)


    def forward(self, v, a, hidden_state):
        # TODO: Fill in docstring: visual_alignment and semantic_alignment
        r"""

        Args:
            v (torch.Tensor): Dimension [Batch size, max_time_steps = N_v, hidden_state =
            visual_units]
            a (torch.Tensor): Dimension [Batch size, max_time_steps = N_a, hidden_state =
            semantic_units]
            hidden_state (torch.Tensor): Hidden state for lstm

        Returns:
            ctx (torch.Tensor): Joint context vector
            visual_alignments (torch.Tensor):
            semantic_alignments (torch.Tensor):
        """
        state_v = torch.rand(self.batch_size, self.num_visual)

        visual_alignments, _ = self.visual_attn(hidden_state, state=state_v, memory=v)

        state_a = torch.rand(self.batch_size, self.num_semantic)

        semantic_alignments, _ = self.semantic_attn(hidden_state, state=state_a, memory=a)

        v_attn = torch.bmm(visual_alignments.view(self.batch_size, 1, -1), v).squeeze(1)

        a_attn = torch.bmm(semantic_alignments.view(self.batch_size, 1, -1), a).squeeze(1)

        cat_attn = torch.cat(
            [v_attn.view(self.batch_size, -1),
             a_attn.view(self.batch_size, -1)],
            1)

        ctx = self.layer(cat_attn)

        return ctx, visual_alignments, semantic_alignments

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
            'hidden_size': 512,
            'batch_size': 1,
            'num_units': 512,
            'visual_units': 2048,
            'semantic_units': 512,
            'num_visual': 1,
            'num_semantic': 1,
        }

    @property
    def output_size(self):
        r"""The feature size of :meth:`forward` output tensor(s),
        usually it is equal to the last dimension value of the output
        tensor size.
        """
        return torch.Size([self.batch_size, self.hparams.hidden_size])


class LstmSentence(ModuleBase):
    r"""This is an implementation of 1st in the hierarchy for 2 layered hierarchical LSTM
    implementation in Texar. In this particular application the first tier takes in the input
    which here is from co-attention and outputs a hidden state vector. At each time step of the
    tier-1 LSTM, the hidden state vector is fed into the tier 2 LSTM as a state tuple to output
    a sentence.

    NOTE: Run the LSTM sentence in a for loop range [0 max time steps] till termination
    Because at each time step we calculate the visual and semantic attention (stack at the end of
    the max timesteps to find the alpha and beta)
    Output the t needed for the Word lstm
    At each time step we produce a 0 or 1 to continue or stop

    This is run at every time step of Sentence LSTM as stated above
    It is Bernoulli variable
    p_pred shape [Batch size, 2]
    p_target shape [Batch size]

    Args:
        hparams (dict or HParams, optional): LstmSentence hyperparameters. Missing
            hyperparameters will be set to default values. See :meth:`default_hparams` for the
            hyperparameter structure and default values.
                * hidden_size (int): Hidden_size the same as GNN output
                * num_units (int): Intermediate number of nodes for the BahdanauAttention attention
                calculation
                * visual_units (int): Dimension of visual features
                * semantic_units (int): Dimension of semantic features
                * num_visual (int): Number of visual features
                * num_semantic (int): Number of semantic features
                * seq_len (int): Sequence length for sentence LSTM
                * batch_size (int): Batch size
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        # Here the input size is equal to the hidden size used as the input from co-attention
        input_size = self.hparams.input_size

        # LSTM parameters
        # the hidden vector and input size of the sentence LSTM
        self.hidden_size = self.hparams.hidden_size
        self.seq_len = self.hparams.seq_len
        self.batch_size = self.hparams.batch_size

        # Attention parameters
        num_units = self.hparams.num_units

        # Dimension of visual and semantic features
        visual_units = self.hparams.visual_units
        semantic_units = self.hparams.semantic_units

        # As per the On the Automatic Generation of Medical Imaging Reports paper
        num_visual = self.hparams.num_visual
        num_semantic = self.hparams.num_semantic

        # The Co_Attention module
        # Observe that the input to the LSTM and hidden vector have the same dimension
        # If not add a new parameter and make changes accordingly
        hparams_coattn = {
            "num_units": num_units,
            "visual_units": visual_units,
            "semantic_units": semantic_units,
            "hidden_size": self.hidden_size,
            "num_visual": num_visual,
            "num_semantic": num_semantic,
            "batch_size": self.batch_size
        }
        self.co_attn = CoAttention(hparams_coattn)

        enc_hparams = {
            'rnn_cell': {
                'type': 'LSTMCell',
                'kwargs': {
                    'num_units': self.hidden_size
                }
            }
        }

        default_hparams = UnidirectionalRNNEncoder.default_hparams()

        hparams_rnn = HParams(enc_hparams, default_hparams)

        self.lstm = UnidirectionalRNNEncoder(input_size=input_size, hparams=hparams_rnn.todict())

    def forward(self, v, a, hidden):
        # TODO: Fill in return docstring
        r"""
        Return the visual_alignments, semantic_alignments for the loss function calculation
        Stack the visual_alignments, semantic_alignments at each time step of the sentence LSTM to
        obtain the alpha (visual_alignments) beta (semantic_alignments)
        Args:
            v (torch.Tensor): Visual features of image patches
            a (torch.Tensor): Semantic features. Word embeddings of predicted disease tags
            hidden (torch.Tensor): Previous hidden state in LSTM

        Returns:

        """
        (h_0, c_0) = hidden

        inp_lstm, visual_alignments, semantic_alignments = self.co_attn(v, a, h_0)
        # TODO: BUG HERE! inp_lstm needs to have size [batch, time, depth], got 2 dims only here

        output, hidden = self.lstm(
            inp_lstm.view(1, 1, -1),
            initial_state=(h_0.view(self.batch_size, self.hidden_size),
                           c_0.view(self.batch_size, self.hidden_size)))

        return output, hidden, visual_alignments, semantic_alignments

    def init_hidden(self):
        # TODO: self.seq_len can only be 1 here since h_0.view(self.batch_size, self.hidden_size)
        #  Need to change either hidden init here or h_0.view/c_0.view above
        r"""Initialize hidden tensor

        Returns:
            Tuple[torch.Tensor]: Tuple of tensors with size [seq_len, batch_size, hidden_size]

        """
        return (torch.zeros(self.seq_len, self.batch_size, self.hidden_size),
                torch.zeros(self.seq_len, self.batch_size, self.hidden_size))

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
            "input_size": 512,
            "hidden_size": 512,
            "num_units": 121,
            "visual_units": 2048,
            "semantic_units": 512,
            "seq_len": 1,
            "num_visual": 2048,
            "num_semantic": 512,
            "batch_size": 1
        }

    @property
    def output_size(self):
        return self.lstm.output_size


class LstmWord(ModuleBase):
    """This is an implementation of 2nd in the hierarchy for 2 layered hierarchical LSTM
    implementation in Texar. In this particular application the first tier takes in the input
    which here is from co-attention and outputs a hidden state vector. At each time step of the
    tier-1 LSTM, the hidden state vector is fed into the tier 2 LSTM as a state tuple to output
    a sentence.

    NOTE: Run the LSTM sentence in a for loop range [0 max time steps] till termination
    Because at each time step we calculate the visual and semantic attention (stack at the end of
    the max timesteps to find the alpha and beta)
    Output the t needed for the Word lstm
    At each time step we produce a 0 or 1 to continue or stop

    This is run at every time step of Sentence LSTM as stated above
    It is Bernoulli variable
    p_pred shape [Batch size, 2]
    p_target shape [Batch size]

    Args:
        hparams (dict or HParams, optional): LstmWord hyperparameters. Missing
            hyperparameters will be set to default values. See :meth:`default_hparams` for the
            hyperparameter structure and default values.
        hidden_size (int): hidden_size the same as GNN output
        output_size (int):
        seq_len (int): sequence length for word lstm
        batch_size (int): batch size
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        self.hidden_size = self.hparams.hidden_size
        output_size = self.hparams.output_size
        self.seq_len = self.hparams.seq_len
        self.batch_size = self.hparams.batch_size

        # Embedding layer
        self.embedding = \
            WordEmbedder(vocab_size=output_size, hparams={'dim': self.hidden_size})

        enc_hparams = {
            'rnn_cell': {
                'type': 'LSTMCell',
                'kwargs': {
                    'num_units': self.hidden_size
                }
            }
        }

        default_hparams = BasicRNNDecoder.default_hparams()

        hparams_rnn = HParams(enc_hparams, default_hparams)

        self.decoder = BasicRNNDecoder(input_size=self.hidden_size, token_embedder=self.embedding,
                                       vocab_size=output_size, hparams=hparams_rnn.todict())

    def forward(self, hidden, train, inp=None, sentence_len=None):
        """

        Args:
            hidden (torch.Tensor): Hidden state from LstmSentence
            train (bool): If in training
            inp (torch.Tensor): Groundtruth tokens in a sentence. Only used in training
            sentence_len (int): Number of token in a sentence

        Returns:
            output (torch.Tensor): Generated output

        """

        if train:
            output, _, _ = self.decoder(
                decoding_strategy='train_greedy',
                inputs=inp,
                sequence_length=sentence_len,
                initial_state=hidden)
        else:
            # Create helper
            helper = self.decoder.create_helper(
                decoding_strategy='infer_greedy',
                start_tokens=torch.Tensor([CLS_TOKEN] * self.batch_size).long(),
                end_token=SEP_TOKEN,
                embedding=self.embedding)

            # Inference sample
            # here sentence length is the max_decoding_length
            output, _, _ = self.decoder(
                helper=helper,
                initial_state=hidden,
                max_decoding_length=sentence_len)

        return output

    def init_hidden(self):
        r"""Initialize hidden tensor

        Returns:
            Tuple[torch.Tensor]: Tuple of tensors with size [seq_len, batch_size, hidden_size]

        """
        return (torch.zeros(self.seq_len, self.batch_size, self.hidden_size),
                torch.zeros(self.seq_len, self.batch_size, self.hidden_size))

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
            "hidden_size": 512,
            "output_size": 512,
            "seq_len": 30,
            "batch_size": 1
        }

    @property
    def output_size(self):
        r"""The feature size of forward output
        """
        return self.decoder.output_size
