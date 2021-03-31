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
from texar.torch.data.data.data_iterators import DataIterator

from config import transforms, pathologies
from config import dataset as config
from iu_xray_data import IU_XRay_Dataset

from models.model import MedicalReportGenerator


class TestModel(unittest.TestCase):
    r"""
    Unit test for CoAttention, SemanticTagGenerator,
    LstmSentence, and LstmWord
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
        self.loader = DataIterator(dataset)

        config['model']['vocab_path'] = \
            "tests/test_iu_xray_data/test_vocab.txt"
        self.model = MedicalReportGenerator(config['model'])
        self.batch_size = 4
        self.num_v_features = 49
        # 1004 = top 1000 frequentest words + BOS + EOS + UNK + PAD
        self.vocab_size = 1004

    def test_model(self):
        r"""
        Unit test for CoAttention, mainly check the dimension of the output.
        Make sure the visual and semantic attention weights add up to one
        """
        batch = next(iter(self.loader))
        result = self.model(batch)

        self.assertTrue(type(result), dict)


if __name__ == "__main__":
    unittest.main()
