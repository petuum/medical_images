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

import unittest
import torch
from models.nlp_model import LSTMSentence, LSTMWord, CoAttention


class NLPModelTest(unittest.TestCase):
    """
    Unit test for LSTM_sentence and LSTM_word
    """

    def test_lstm_sentence(self):
        model = LSTMSentence(
            hidden_size=512,
            num_units=512,
            visual_units=512,
            semantic_units=512,
            N=1,
            M=1,
            seq_len=6,
            batch_size=1
        )
        _input = model.init_hidden()
        # _output = model(_input)

    def test_lstm_word(self):
        model = LSTMWord(
            hidden_size=512,
            output_size=10,
            seq_len=30,
            batch_size=1
        )
        _input = model.init_hidden()
        _ouput = model(_input, train=False)

    def test_attn(self):
        model = CoAttention(
            num_units=512,
            visual_units=512,
            semantic_units=512,
            hidden_size=512,
            N=1,
            M=1,
        )


if __name__ == '__main__':
    unittest.main()