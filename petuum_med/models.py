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
Model classes for medical report generation.
"""

from typing import Optional

import torch

from texar.torch.data import Vocab
from texar.torch.modules import (
    WordEmbedder, BasicRNNDecoder, EncoderBase,
)
from texar.torch.utils.types import MaybeList, MaybeTuple


class WordLSTM(EncoderBase):
    r""" The Word LSTM decoding layer.

    Args:
        word_embedding_size (int): Dimension of the word embedding.
        vocab_size (int): Vocabulary size.

    The output of this decoder is the same as
    :class:`~texar.torch.modules.BasicRNNDecoder`, which is the output shape
    of each time step.
    """

    def __init__(self, word_embedding_size: int, vocab_size: int):
        super().__init__()

        # Set up the Word Embedding layer.
        self.embedding = WordEmbedder(
            vocab_size=self.output_size,
            hparams={'dim': self.hidden_size}
        )

        # Setup the LSTM layer -- batch_first = true.
        self.decoder = BasicRNNDecoder(
            input_size=word_embedding_size,
            token_embedder=self.embedding,
            vocab_size=vocab_size,
            hparams={
                'rnn_cell': {
                    'type': 'LSTMCell',
                    'kwargs': {'num_units': self.hidden_size}
                }}
        )

    def forward(self,  # type ignore
                initial_state: MaybeList[MaybeTuple[torch.Tensor]],
                train: bool, inputs: torch.Tensor,
                vocab: Vocab, max_sentence_len: Optional[int] = None):
        """ Performs decoding.

        At training time (train == True), it will use the `train_greedy`
        strategy as in :class:`~texar.torch.modules.BasicRNNDecoder`.

        At inference time (train == False), it will use the `infer_greedy`
        strategy as in :class:`~texar.torch.modules.BasicRNNDecoder`.

        Args:
            initial_state: Initial state of decoding.
            train (bool): Whether to use training model and inference mode.
            inputs (torch.Tensor): Input tensor for decoding.
            vocab : The Texar vocabulary class that stores the vocabulary of
                the data.
            max_sentence_len (optional): A int scalar Tensor indicating the
                maximum allowed number of tokens (decoding steps). If `None`,
                either `hparams["max_decoding_length_train"]` or
                `hparams["max_decoding_length_infer"]` will be used depending
                on the mode defined by `train`.

        Returns:
            Refer to :class:`~texar.torch.modules.BasicRNNDecoder`.

        """
        if train:
            (output, final_state, sequence_lengths) = self.decoder(
                decoding_strategy='train_greedy', inputs=inputs,
                sequence_length=max_sentence_len, initial_state=initial_state
            )
        else:
            # Create helper.
            helper = self.decoder.create_helper(
                decoding_strategy='infer_greedy',
                start_tokens=vocab.bos_token_id * self.batch_size,
                end_token=vocab.eos_token_id, embedding=self.embedding)

            # Inference sample.
            # Here sentence length is the max_decoding_length
            (output, final_state, sequence_lengths) = self.decoder(
                helper=helper, initial_state=initial_state,
                max_decoding_length=max_sentence_len)

        return output, final_state, sequence_lengths

    @property
    def output_size(self):
        """Output size of one step."""
        return self.decoder.output_size
