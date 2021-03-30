# Copyright 2021 The Petuum Authors. All Rights Reserved.
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

"""This module tests for NLP Model."""

import unittest
import torch

from config import HIDDEN_SIZE
from config import dataset as config
from models.nlp_model import SemanticTagGenerator, \
    LstmSentence, LstmWord


class TestNLPModel(unittest.TestCase):
    r"""
    Unit test for CoAttention, SemanticTagGenerator,
    LstmSentence, and LstmWord
    """

    def setUp(self):
        sentence_lstm = LstmSentence(config['model']['sentence_lstm'])
        self.co_attn = sentence_lstm.co_attn
        self.sentence_lstm = sentence_lstm

        self.tag_generator = SemanticTagGenerator(
            config['model']['tag_generator'])

        config['model']['vocab_path'] = \
            "tests/test_iu_xray_data/test_vocab.txt"
        self.word_lstm = LstmWord(
            config['model']['word_lstm'])

        self.visual_dim = config['model']['sentence_lstm']['visual_dim']
        self.num_tags = config['model']['tag_generator']['num_tags']
        self.top_k = config['model']['tag_generator']['top_k_for_semantic']

        self.batch_size = 4
        self.num_v_features = 49
        # 1004 = top 1000 frequentest words + BOS + EOS + UNK + PAD
        self.vocab_size = 1004

    def test_co_attention(self):
        r"""
        Unit test for CoAttention, mainly check the dimension of the output.
        Make sure the visual and semantic attention weights add up to one
        """
        batch_size = self.batch_size
        input_v = torch.rand(batch_size, self.num_v_features, self.visual_dim)
        input_a = torch.rand(batch_size, self.num_tags, HIDDEN_SIZE)
        zeros = torch.zeros(batch_size, HIDDEN_SIZE)
        hidden = (zeros, zeros)
        context, visual_align, semantic_align = self.co_attn(
            input_v, input_a, hidden)

        self.assertEqual(
            context.size(),
            torch.Size([batch_size, HIDDEN_SIZE]))

        self.assertEqual(
            visual_align.size(),
            torch.Size([batch_size, 49, 1]))

        self.assertEqual(
            semantic_align.size(),
            torch.Size([batch_size, self.num_tags, 1]))

        self.assertTrue(torch.allclose(
            visual_align.sum(1).squeeze(),
            torch.ones(batch_size)))

        self.assertTrue(torch.allclose(
            semantic_align.sum(1).squeeze(),
            torch.ones(batch_size)))

    def test_tag_generator(self):
        r"""
        Unit test for SemanticTagGenerator, check the dimension of the output.
        """
        batch_size = self.batch_size
        input_prob = torch.rand(batch_size, self.num_tags)
        semantic_feature = self.tag_generator(input_prob)

        self.assertEqual(
            semantic_feature.size(),
            torch.Size([batch_size, self.top_k, HIDDEN_SIZE]))

    def test_sentence_lstm(self):
        r"""
        Unit test for LstmSentence, check the dimension of the output.
        """
        batch_size = self.batch_size
        input_v = torch.rand(batch_size, self.num_v_features, self.visual_dim)
        input_a = torch.rand(batch_size, self.num_tags, HIDDEN_SIZE)
        zeros = torch.zeros(batch_size, HIDDEN_SIZE)
        hidden = (zeros, zeros)
        state, topic, pred_stop, _, _, _ = self.sentence_lstm(
            input_v, input_a, hidden)

        self.assertEqual(
            state[0].size(),
            torch.Size([batch_size, HIDDEN_SIZE]))

        self.assertEqual(
            state[1].size(),
            torch.Size([batch_size, HIDDEN_SIZE]))

        self.assertEqual(
            topic.size(),
            torch.Size([batch_size, HIDDEN_SIZE]))

        self.assertEqual(
            pred_stop.size(),
            torch.Size([batch_size, 2]))


    def test_word_lstm(self):
        r"""
        Unit test for LstmSentence, check the dimension of the output.
        """
        batch_size = self.batch_size
        max_sentence_len = self.word_lstm.max_decoding_length
        topic = torch.zeros(batch_size, HIDDEN_SIZE)
        teacher_words = 4 + torch.randint(
            high=1000, size=(batch_size, max_sentence_len))
        sentence_len = 1 + torch.randint(
            high=max_sentence_len - 1, size=(batch_size, ))

        for i in range(batch_size):
            end = sentence_len[i]
            teacher_words[i, end - 1] = self.word_lstm.EOS
            # PAD token id is 0
            teacher_words[(end - 1):] = 0

        # Test the train mode
        word_output = self.word_lstm(
            topic, train=True,
            inp=teacher_words, sentence_len=sentence_len + 1)

        max_len_batch = torch.max(sentence_len)

        self.assertEqual(
            word_output.sample_id.size(),
            torch.Size([batch_size, max_len_batch + 1]))

        self.assertEqual(
            word_output.logits.size(),
            torch.Size([
                batch_size, max_len_batch + 1, self.vocab_size]))

        # Test the inference mode
        word_output = self.word_lstm(topic, train=False)

        self.assertEqual(
            word_output.sample_id.size(),
            torch.Size([batch_size, max_sentence_len]))


if __name__ == "__main__":
    unittest.main()
