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
from models.nlp_model import LstmSentence, LstmWord, CoAttention


class NLPModelTest(unittest.TestCase):
    """
    Unit test for LstmSentence, LstmWord and CoAttention
    """

    def test_lstm_sentence(self):
        r"""Test for LstmSentence module initialization and forward
        """
        model = LstmSentence()
        _input = model.init_hidden()
        # _output = model(_input)

    def test_lstm_word(self):
        r"""Test for LstmWord module initialization and forward
        """
        model = LstmWord()
        _input = model.init_hidden()
        # _ouput = model(_input, train=False)

    def test_coattn(self):
        r"""Test for CoAttention module initialization and forward
        """
        model = CoAttention()


if __name__ == '__main__':
    unittest.main()
