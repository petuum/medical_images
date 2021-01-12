#  Copyright 2020 Petuum Inc. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pickle
import numpy as np

import torch
import torch.nn as nn

from transformers import BertTokenizer

from texar.torch.losses import sequence_sparse_softmax_cross_entropy


###################################################
# Phrase generation and Loss for phrase generation
###################################################
# here the phrase_decoder is an instance of the class LSTM_word
# This is a function that is executed at every time step S of sentence LSTM
# new_node_embedding is the hidden statevector of time step S from sentence LSTM

# index2phrase is written exclusively for the G2G task.
# This is a list containing the phrases/concepts of the nodes
# corresponding to that node's index in the strongly connected parent graph.
# We can remove this for any other task.
# For more details on index2phrase please look into the zip folder for complete 2 layer lstm
# implementation for the G2G task.

# embedding_keys.pkl contains the concepts of all the nodes in the parent graph
with open("embedding_keys.pkl", "rb") as f:
   embedding_keys = pickle.load(f)

# store them in the embedding_keys
for i,item in enumerate(embedding_keys):
    words = item.replace("_", " ")
    embedding_keys[i] = words

# use only those embedding_keys that are present in the strongly connected component of the
# parent graph
embedding_keys = np.array(embedding_keys)
# TODO: Confirm component
index2phrase = embedding_keys["component"]


def generate_phrase(
        phrase_decoder,
        new_node_embedding,
        index=None, batch_size = 1, train=True, device='cpu'):
    r"""This is a function that is executed at every time step S of sentence LSTM
    Args:
        phrase_decoder (): An instance of the class LSTM_word.
        new_node_embedding (): hidden statevector of time step S from sentence LSTM
        index ():
        train ():
        device ():

    Returns:

    """
    phrase_loss = 0
    tokenizer = BertTokenizer("embedding_keys.txt")
    device = torch.device(device)

    if train:
        phrase_decoder_input = torch.Tensor(tokenizer(index2phrase[index])['input_ids'])\
            .view(phrase_decoder.batch_size, -1).long().to(device)

        sentence_len = torch.Tensor([len(phrase_decoder_input[0, 1:])]).long()

    state = new_node_embedding

    (state_h_0, state_c_0) = state

    # 1 because the number of stacked lstm layers are 1 pytorch input specification
    # TODO: Confirm batch_size and hidden_size implementation
    state_h_0 = (state_h_0).view(phrase_decoder.batch_size, phrase_decoder.hidden_size)

    state_c_0 = (state_c_0).view(phrase_decoder.batch_size, phrase_decoder.hidden_size)

    phrase_decoder_hidden = (state_h_0, state_c_0)

    if train:
        output = phrase_decoder(phrase_decoder_hidden, train, inp=phrase_decoder_input,
                                sentence_len=sentence_len)

        # per time step S of sentence LSTM
        phrase_loss = sequence_sparse_softmax_cross_entropy(labels=phrase_decoder_input[:, 1:],
                                                            logits=output.logits,
                                                            sequence_length=sentence_len)
    else:

        output = phrase_decoder(phrase_decoder_hidden, train, sentence_len=5)

    predict = list(np.array(output.sample_id.view(-1)))

    if train:
        return phrase_loss, tokenizer.decode(predict), phrase_decoder_input
    else:
        return phrase_loss, tokenizer.decode(predict)


def tag_loss(p_pred, p_target):
    r"""Tag loss function. Apply the weighting lambdas in the main function this is just a loss.
    without lambda weights.
    Args:
        p_pred (): Size: [batch_size, hidden]. It is logits before applying softmax or sigmoid
        p_target (): Size: [batch_size, hidden]. Assuming p_target is a normalised vector of
        distribution l/||l||_1 of tags

    Returns:
        loss (torch.Tensor): Calculated tag loss
    """
    logsoftmax = nn.LogSoftmax(dim=1)

    return torch.mean(torch.sum(- p_target * logsoftmax(p_pred), 1))


def attn_loss(alpha, beta):
    r"""Attention loss function. Weigh it in the main loop as per lambda

    Args:
        alpha (): Size: [batch_size, N, S]. N is the number of visual features. S is the number
        of time steps in Sentence LSTM.
        beta (): Size: [batch_size, M, S]. M is the number of semantic features. S is the number
        of time steps in Sentence LSTM

    Returns:
        loss (torch.Tensor): Calculated attention loss
    """
    # alpha is [Batch_size, N, S]
    # beta is [Batch_size, M, S]
    # N is the number of visual features, M is the number of semantic features
    # S is the number of time steps in Sentence LSTM
    visual_attn_loss = torch.sum(((1 - torch.sum(alpha, -1)) ** 2), -1)

    semantic_attn_loss = torch.sum(((1 - torch.sum(beta, -1)) ** 2), -1)

    return torch.mean(visual_attn_loss, semantic_attn_loss)


def sentence_loss(p_pred, p_target):
    r"""Sentence loss function.
    Args:
        p_pred ():
        p_target ():

    Returns:

    """
    sentence_lossfn = nn.CrossEntropyLoss()

    return sentence_lossfn(p_pred, p_target)
