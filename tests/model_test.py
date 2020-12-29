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
Unit tests for the models.
"""
import tempfile
import unittest

import torch
from texar.torch.data import Vocab

from petuum_med.models import WordLSTM


class WordLSTMTest(unittest.TestCase):
    """Tests :class:`~petuum_med.models.WordLSTM`.
    """

    def setUp(self):
        self._vocab_size = 4
        self._max_time = 8
        self._batch_size = 16
        self._emb_dim = 20
        self._inputs = torch.randint(
            self._vocab_size, size=(self._batch_size, self._max_time))
        self._hidden_size = 5
        with tempfile.NamedTemporaryFile(mode='w') as t:
            self.vocab = Vocab(t.name)

    def test_decode(self):
        model = WordLSTM(
            hparams={
                "word_embedding_size": self._emb_dim,
                "vocab_size": self._vocab_size,
                "lstm_cell_hidden": self._hidden_size,
                "batch_size": self._batch_size
            }
        )
        model(
            None,
            train=False,
            inputs=self._inputs,
            vocab=self.vocab,
        )
