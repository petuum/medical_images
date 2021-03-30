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
# pylint: disable=attribute-defined-outside-init
"""This module tests for IU_XRay_Dataset."""

import unittest
import torch
from texar.torch.data.data.data_iterators import DataIterator

from config import transforms, pathologies
from iu_xray_data import IU_XRay_Dataset


class TestIUXRayDataset(unittest.TestCase):
    r"""
    Unit test for IU XRay Dataset
    """
    def setUp(self):
        self.batch_size = 4
        self.num_label = len(pathologies)
        hparams = {
            "datasource":{
                "img_root": "tests/test_iu_xray_data/iu_xray_images",
                "text_root": "tests/test_iu_xray_data/text_root",
                "vocab_path": "tests/test_iu_xray_data/test_vocab.txt",
                "transforms": transforms,
                "pathologies": pathologies,
            },
            "batch_size": self.batch_size,
            "shuffle": False,
        }
        dataset = IU_XRay_Dataset(hparams)
        dataset.to(torch.device('cpu'))
        self.vocab = dataset.source.vocab

        self.ground_truth_keys = [
            'img_tensor', 'label', 'token_tensor', 'stop_prob']
        self.ground_truth_findings = ['cardiac and mediastinal contours '
                                      'are within normal limits <EOS> the '
                                      'lungs are clear <EOS> bony structures '
                                      'are intact <EOS>']

        self.loader = DataIterator(dataset)

    def test_dataload(self):
        batch = next(iter(self.loader))

        for key in self.ground_truth_keys:
            self.assertIn(key, batch.keys())

        img_tensor = batch.img_tensor
        token_tensor = batch.token_tensor
        stop_prob = batch.stop_prob
        label = batch.label

        # Image shape
        self.assertEqual(
            img_tensor.size(),
            torch.Size([4, 3, 224, 224]))
        # Label shape
        self.assertEqual(label.size(), torch.Size([4, self.num_label]))

        # The second dimension of token_tensor and stop_prob
        # should be equal (max_sentence_num + 1)
        self.assertEqual(token_tensor.size(1), stop_prob.size(1))

        # token_tensor should be padded with zero at the
        # the end of the second dimension
        self.assertTrue(torch.equal(
            token_tensor[:, -1, :],
            torch.zeros_like(token_tensor[:, -1, :])
        ))

        # Examine the reconstructed paragraph
        # using the second sample in the batch
        paragrah = []
        for j in range(token_tensor.shape[1] - 1):
            sen = token_tensor[0, j].long()
            # Get the effective length of the sentence
            mask = sen != self.vocab.pad_token_id
            len_sen = mask.to(torch.long).sum()
            if len_sen > 0:
                sen = sen[:len_sen].cpu().tolist()
                sen_tokens = self.vocab.map_ids_to_tokens_py(sen)
                paragrah.append(' '.join(sen_tokens))

        paragrah = ' '.join(paragrah)
        self.assertEqual(paragrah, self.ground_truth_findings[0])


if __name__ == "__main__":
    unittest.main()
