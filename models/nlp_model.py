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
import torch.nn.functional as F

# Texar Library
from texar.torch import ModuleBase, HParams
from texar.torch.modules import WordEmbedder, BasicRNNDecoder

from models.utils import InferenceHelper


class CoAttention(ModuleBase):
    r"""It takes as input the visual features and the semantic features
    along with the hidden vector of the previous time step of the sentence
    LSTM (layer 1 in the 2 layer hierarchical LSTM), and generates the context
    for the sentence LSTM.

    Args:
        hparams (dict or HParams, optional): LSTMSentence hyperparameters.
        Missing hyperparameters will be set to default values.
        See :meth:`default_hparams` for the hyperparameter structure
        and default values.
            * visual_dim (int): Dimension of visual features
            * hidden_size (int): Dimension of semantic features, hidden states,
                and input to the lstm
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        visual_dim = self.hparams.visual_dim
        semantic_dim = self.hparams.hidden_size
        hidden_size = self.hparams.hidden_size

        # Visual attention
        # Notation from Equation 2 of the paper

        # As per the wingspan project, we set bias = False for W_v and W_a
        # https://gitlab.int.petuum.com/shuxin.yao/image_report_generation/blob/master/implementation/CoAtt/CoAtt.py
        self.W_v = nn.Linear(
            in_features=visual_dim,
            out_features=hidden_size,
            bias=False)

        self.W_v_h = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size)

        self.W_v_att = nn.Linear(
            in_features=hidden_size,
            out_features=1)

        # Semantic attention
        # Notation from Equation 3 of the paper
        self.W_a = nn.Linear(
            in_features=semantic_dim,
            out_features=hidden_size,
            bias=False)

        self.W_a_h = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size)

        self.W_a_att = nn.Linear(
            in_features=hidden_size,
            out_features=1)

        # Context calculation layer
        self.W_fc = nn.Linear(
            in_features=visual_dim + semantic_dim,
            out_features=hidden_size)

        self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        r"""Initialize the weights for each module
        """
        # As per the wingspan project
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_v_h.weight)
        nn.init.xavier_uniform_(self.W_v_att.weight)
        self.W_v_att.bias.data.fill_(0)
        self.W_v_h.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.xavier_uniform_(self.W_a_h.weight)
        nn.init.xavier_uniform_(self.W_a_att.weight)
        self.W_a_att.bias.data.fill_(0)
        self.W_a_h.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.W_fc.weight)
        self.W_fc.bias.data.fill_(0)

    def forward(self, visual_feature, semantic_feature, prev_state):
        r"""
        Calculate the context vectors from the input visual feature and
        semantic feature

        Args:
            visual_feature (torch.Tensor): dimension
                [batch size, num_visual_features, visual_dim]
            semantic_feature (torch.Tensor): dimension
                [batch size, num_semantic_features, semantic_dim]
            prev_state Tuple(torch.Tensor):
                Previous state of sentence_lstm
        Returns:
            context (torch.Tensor): Joint context vector,
                dimension [batch size, visual_dim + semantic_dim]
            visual_align (torch.Tensor):
                dimension [batch size, num_visual_features, 1]
            semantic_align (torch.Tensor):
                dimension [batch size, num_semantic_features, 1]
        """
        # Visual attention
        # Equation 2 of the paper
        W_v = self.W_v(visual_feature)
        W_v_h = self.W_v_h(prev_state[0]).unsqueeze(1)

        visual_score = self.W_v_att(self.tanh(W_v + W_v_h))
        visual_align = F.softmax(visual_score, dim=1)
        visual_att = torch.sum(visual_align * visual_feature, dim=1)

        # Semantic attention
        # Equation 3 of the paper
        W_a = self.W_a(semantic_feature)
        W_a_h = self.W_a_h(prev_state[0]).unsqueeze(1)

        semantic_score = self.W_a_att(self.tanh(W_a_h + W_a))
        semantic_align = F.softmax(semantic_score, dim=1)
        semantic_att = torch.sum(semantic_align * semantic_feature, dim=1)

        # Calculate the context
        # Equation 4 of the paper
        # visual_att = torch.sum(visual_feature, 1) / visual_feature.size(1)
        # semantic_att = torch.zeros_like(semantic_att)
        cat_att = torch.cat([visual_att, semantic_att], dim=1)
        context = self.W_fc(cat_att)

        return context, visual_align, semantic_align

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
            'hidden_size': 512,
            'visual_dim': 1024,
        }

class SemanticTagGenerator(ModuleBase):
    r"""It takes as input the probabilities of active tags and generates
    the sementic features.

    Args:
        hparams (dict or HParams, optional): SemanticTagGenerator hyperparams.
        Missing hyperparameters will be set to default values.
        See :meth:`default_hparams` for the hyperparameter structure
        and default values.
            * num_tags (int): Number of total tags
            * k (int): Number of tags to be selected to produce
                the semantic features
            * hidden_size (int): Dimension of tag embeddings
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        self.num_tags = self.hparams.num_tags
        self.embed = nn.Embedding(self.num_tags, self.hparams.hidden_size)
        self.k = self.hparams.top_k_for_semantic

        # As per the wingspan project
        nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, soft_scores):
        r"""
        Obtained the embeddings for the top k tags in
        terms of probabilities

        Args:
            soft_scores (torch.Tensor): Probabilities for each tags

        Return (torch.Tensor): Embeddings for the top k tags. Dimension
            [batch_size, k, hidden_size]
        """
        return self.embed(
            torch.topk(soft_scores, self.k)[1])

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
            'hidden_size': 512,
            "num_tags": 210,
            "top_k_for_semantic": 10,
        }

class LstmSentence(ModuleBase):
    r"""This is an implementation of 1st in the hierarchy for 2 layered
    hierarchical LSTM implementation in Texar. In this particular application,
    the first tier takes in the input which here is from co-attention and
    outputs a hidden state vector. At each time step of the tier-1 LSTM,
    the hidden state vector is fed into the tier 2 LSTM as a state tuple
    to output a sentence.

    NOTE: Run the Sentence LSTM in a for loop (range [0, max time steps]) till
    termination. At each time step, we calculate the visual and semantic
    attention (stack at the end of the max timesteps to find the visual_ailgn
    and semantic_align). Output the Topic needed for the Word lstm.

    At each time step we produce a 0 or 1 to continue or stop. This is run at
    every time step of Sentence LSTM as stated above. It is Bernoulli variable
    p_pred shape [Batch size, 2]
    p_target shape [Batch size]

    Args:
        hparams (dict or HParams, optional): LstmSentence hyperparameters.
        Missing hyperparameters will be set to default values.
        See :meth:`default_hparams` for the
        hyperparameter structure and default values.
            * hidden_size (int): Dimension of semantic features,
                hidden states, and input to the lstm
            * visual_dim (int): Dimension of visual features
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)
        hidden_size = self.hparams.hidden_size
        self.hidden_size = hidden_size

        # The Co_Attention module
        hparams_coattn = {
            "visual_dim": self.hparams.visual_dim,
            "hidden_size": hidden_size,
        }
        self.co_attn = CoAttention(hparams_coattn)

        # LSTM cell
        self.lstm = nn.LSTMCell(
            input_size=hidden_size,
            hidden_size=hidden_size)

        # As per the wingspan project, we set bias = False
        # for W_t_h and W_stop_s_1
        # https://gitlab.int.petuum.com/shuxin.yao/image_report_generation/blob/master/implementation/CoAtt/CoAtt.py

        # Topic generation
        # Notation from Equation 5 of the paper
        self.W_t_h = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False)

        self.W_t_ctx = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size)

        # Stop control
        # Notation from Equation 6 of the paper
        self.W_stop_s_1 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False)

        self.W_stop_s = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size)

        self.W_stop = nn.Linear(
            in_features=hidden_size,
            out_features=2)

        # Miscellaneous
        self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        # As per the wingspan project
        # Topic generation
        nn.init.xavier_uniform_(self.W_t_h.weight)
        nn.init.xavier_uniform_(self.W_t_ctx.weight)
        self.W_t_ctx.bias.data.fill_(0)

        # Stop control
        nn.init.xavier_uniform_(self.W_stop_s_1.weight)
        nn.init.xavier_uniform_(self.W_stop_s.weight)
        nn.init.xavier_uniform_(self.W_stop.weight)
        self.W_stop.bias.data.fill_(0)
        self.W_stop_s.bias.data.fill_(0)

    def forward(self, visual_feature, semantic_feature, prev_state):
        r"""
        Obtain the context vector by inputing the visual_feature
        and semantic_feature to the CoAttention module.
        The produced visual_align, semantic_align are used to
        calculate the regularization loss function.
        Obtain the topic vector according to Equation 5 in the paper.
        Obtain the stop control prediction according to Equation 6 in the paper.
        Args:
            visual_feature (torch.Tensor): Visual features of image patches
                dimension [batch size, num_visual_features, visual_dim]
            semantic_feature (torch.Tensor): Semantic features. Word embeddings
                of predicted disease tags
                dimension [batch size, num_semantic_features, semantic_dim]
            prev_state Tuple(torch.Tensor): Previous states of the Sentence LSTM

        Returns:
            state Tuple(torch.Tensor): Current states of
                the Sentence LSTM
            topic (torch.Tensor): Generated topics for the Sentence LSTM
            p_stop (torch.Tensor): predicted stop probability
            visual_align (torch.Tensor):
                dimension [batch size, num_visual_features, 1]
            semantic_align (torch.Tensor):
                dimension [batch size, num_semantic_features, 1]
            topic_var (torch.float): averaged variance of the topics across
                differnt sample in the current batch
        """
        context, visual_align, semantic_align = self.co_attn(
            visual_feature, semantic_feature, prev_state)
        state = self.lstm(context, prev_state)

        # Equation 5 in the paper
        topic = self.tanh(
            self.W_t_h(state[0]) + self.W_t_ctx(context))
        topic_var = torch.std(topic, dim=0).mean()

        # Equation 6 in the paper
        p_score = self.W_stop(self.tanh(
            self.W_stop_s_1(prev_state[0]) + self.W_stop_s(state[0])
        ))
        p_stop = F.softmax(p_score, dim=1)

        return state, topic, p_stop, visual_align, semantic_align, topic_var

    def init_hidden(self, batch_size):
        r"""Initialize hidden tensor

        Returns:
            Tuple[torch.Tensor]: Tuple of tensors with size
            [batch_size, hidden_size]
        """
        zeros = torch.zeros(batch_size, self.hidden_size)
        if torch.cuda.is_available():
            zeros = zeros.cuda()

        return (zeros, zeros)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
            "hidden_size": 512,
            "visual_dim": 1024,
        }

    @property
    def output_size(self):
        return self.lstm.output_size


class LstmWord(ModuleBase):
    """This is an implementation of 2nd in the hierarchy for 2 layered
    hierarchical LSTM implementation in Texar. In this particular
    application the first tier takes in the input, which here is from
    co-attention and outputs a hidden state vector. At each time step of the
    tier-1 LSTM, the hidden state vector is fed into the tier 2 LSTM
    as a state tuple to output a sentence.

    NOTE: We set the token embedder of the decoder as the Identity.
    Because we need to customize the input (cat[topic; start]). We also
    need to use a customized inference helper during inference phase.

    Args:
        hparams (dict or HParams, optional): LstmWord hyperparameters.
        Missing hyperparameters will be set to default values.
        See :meth:`default_hparams` for the hyperparameter
        structure and default values.
            * hidden_size (int): hidden_size the same as the num_units,
                and the input size of the word LSTMCell
            * vocab_size (int): Vocabulary size
            * max_decoding_length (int): maximum number of words per sentence
            * BOS (int): Index of the BOS
            * EOS (int): Index of the EOS
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        self.hidden_size = self.hparams.hidden_size
        self.vocab_size = self.hparams.vocab_size
        self.max_decoding_length = self.hparams.max_decoding_length

        self.BOS = self.hparams.BOS
        self.EOS = self.hparams.EOS

        # Embedding layer to embed the words
        self.embedding = WordEmbedder(
            vocab_size=self.vocab_size,
            hparams={'dim': self.hidden_size}
        )

        enc_hparams = {
            'rnn_cell': {
                'type': 'LSTMCell',
                'kwargs': {'num_units': self.hidden_size}
            }
        }

        default_hparams = BasicRNNDecoder.default_hparams()
        hparams_rnn = HParams(enc_hparams, default_hparams)

        # We set the token embedder as the Identity.
        self.decoder = BasicRNNDecoder(
            input_size=self.hidden_size,
            token_embedder=nn.Identity(),
            vocab_size=self.vocab_size,
            hparams=hparams_rnn.todict()
        )

    def forward(self, topic, train=True, inp=None, sentence_len=None):
        """
        Generate sentence given the topic from the Sentence LSTM
        Args:
            topic (torch.Tensor): topic state from Sentence LSTM
                Dimension [batch size, hidden_size]
            train (bool): If it is in the training mode
            inp (torch.Tensor): Groundtruth tokens in a sentence.
                Only used in training. Dimension [batch size, max_word_num].
                max_word_num denotes the maximum number of words in one
                sentence, w.r.t current batch
            sentence_len (torch.Tensor): Number of tokens for each
                sentence. Dimension [batch size, ]

        Returns:
            output (torch.Tensor): Generated output from the decoder
        """
        topic = topic.unsqueeze(1)
        batch_size = topic.shape[0]

        start_tokens = topic.new_ones(batch_size) * self.BOS
        start_tokens = start_tokens.long()
        end_tokens = self.EOS

        start_embeddings = self.embedding(start_tokens).unsqueeze(1)
        prefix_embeddings = torch.cat([topic, start_embeddings], dim=1)

        if train:
            assert inp is not None, "During training, inp cannot be None"

            # As per the paper, the first and second input to the Word
            # LSTM are topics and BOS, respectively
            embeddings = self.embedding(inp)
            embeddings = torch.cat([prefix_embeddings, embeddings], dim=1)
            output, _, _ = self.decoder(
                decoding_strategy='train_greedy',
                inputs=embeddings,
                sequence_length=sentence_len)
        else:
            # Create helper
            helper = InferenceHelper(
                start_tokens=start_tokens,
                end_token=end_tokens,
                token_embedder=self.embedding
            )
            # Inference sample
            # here sentence length is the max_decoding_length
            output, _, _ = self.decoder(
                helper=helper,
                inputs=prefix_embeddings,
                max_decoding_length=self.max_decoding_length)

        return output

    def init_hidden(self, batch_size):
        r"""Initialize hidden tensor

        Returns:
            Tuple[torch.Tensor]: Tuple of tensors with size
            [batch_size, hidden_size]
        """
        zeros = torch.zeros(batch_size, self.hidden_size)
        if torch.cuda.is_available():
            zeros = zeros.cuda()

        return (zeros, zeros)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
            "hidden_size": 512,
            "vocab_size": 1004,
            "max_decoding_length": 60,
            "BOS": 2,
            "EOS": 3,
        }
