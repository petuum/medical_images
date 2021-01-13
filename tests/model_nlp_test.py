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
from models.nlp_model import LstmSentence, LstmWord, CoAttention


class NLPModelTest(unittest.TestCase):
    """
    Unit test for LstmSentence, LstmWord and CoAttention
    """

    def test_lstm_sentence(self):
        r"""Test for LstmSentence module initialization and forward
        """
        model = LstmSentence({
            "input_size": 512,
            "hidden_size": 512,
            "num_units": 121,
            "visual_units": 2048,
            "semantic_units": 512,
            "seq_len": 1,
            "num_visual": 2048,
            "num_semantic": 512,
            "batch_size": 1
        })

        # v: [Batch_size, max_time_steps = N_v, hidden_state = visual_units]
        # a: [Batch_size, max_time_steps = N_a, hidden_state = semantic_units]
        input_v = torch.zeros(1, 121, 2048)
        input_a = torch.zeros(1, 121, 512)
        hidden = model.init_hidden()
        output, hidden, visual_alignments, semantic_alignments = model(input_v, input_a, hidden)
        assert output.size() == torch.Size([1, 1, 512])
        assert hidden[0].size() == torch.Size([1, 512])
        assert hidden[1].size() == torch.Size([1, 512])
        assert visual_alignments.size() == torch.Size([1, 1, 121])
        assert semantic_alignments.size() == torch.Size([1, 1, 121])

    def test_lstm_word(self):
        r"""Test for LstmWord module initialization and forward
        """
        model = LstmWord({
            "hidden_size": 512,
            "output_size": 512,
            "seq_len": 1,
            "batch_size": 1
        })
        _input = (torch.zeros(1, 512), torch.zeros(1, 512))
        _ouput = model(_input, train=False)

    def test_coattn(self):
        r"""Test for CoAttention module initialization and forward
        """
        model = CoAttention({
            'hidden_size': 512,
            'batch_size': 1,
            'num_units': 121,
            'visual_units': 2048,
            'semantic_units': 512,
            'num_visual': 2048,
            'num_semantic': 512,
        })
        input_v = torch.zeros(1, 121, 2048)
        input_a = torch.zeros(1, 121, 512)
        hidden = torch.zeros(1, 1, 512)
        ctx, visual_alignments, semantic_alignments = model(input_v, input_a, hidden)
        assert ctx.size() == torch.Size([1, 512])
        assert visual_alignments.size() == torch.Size([1, 1, 121])
        assert semantic_alignments.size() == torch.Size([1, 1, 121])

    def test_connection(self):
        r"""Test lstm module connection
        """
        pass


if __name__ == '__main__':
    unittest.main()
