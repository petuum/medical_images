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

"""This module tests for CV Model."""

import unittest
import torch
from texar.torch.data.data.data_iterators import DataIterator

from config import transforms, pathologies
from iu_xray_data import IU_XRay_Dataset
from models.cv_model import SimpleFusionEncoder, MLC, MLCTrainer


class TestVisualModel(unittest.TestCase):
    r"""
    Unit test for CV Model
    """
    def setUp(self):
        self.batch_size = 4
        self.num_label = len(pathologies)
        data_hparams = {
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
        dataset = IU_XRay_Dataset(data_hparams)
        dataset.to(torch.device('cpu'))
        self.loader = DataIterator(dataset)

        self.extractor = SimpleFusionEncoder()
        mlc_hparam = {
            'num_tags': len(pathologies),
        }
        self.mlc = MLC(mlc_hparam)
        self.mlc_trainer = MLCTrainer(mlc_hparam)

        self.loss = torch.nn.BCEWithLogitsLoss()

    def test_visual_extractor(self):
        batch = next(iter(self.loader))

        img_tensor = batch.img_tensor
        visual_feature = self.extractor(img_tensor)

        self.assertEqual(
            visual_feature.size(),
            torch.Size([4, 1024, 49]))

    def test_mlc(self):
        batch = next(iter(self.loader))

        img_tensor = batch.img_tensor
        visual_feature = self.extractor(img_tensor)
        pred_score = self.mlc(visual_feature)
        pred_prob = self.mlc.get_tag_probs(visual_feature)

        self.assertEqual(
            pred_score.size(),
            torch.Size([4, self.num_label])
        )
        self.assertTrue(
            torch.equal(
                torch.sigmoid(pred_score),
                pred_prob
            )
        )

    def test_mlc_trainer(self):
        batch = next(iter(self.loader))
        img_tensor = batch.img_tensor
        label = batch.label

        result = self.mlc_trainer(batch)

        visual_feature = self.mlc_trainer.extractor(img_tensor)
        pred_score = self.mlc_trainer.mlc(visual_feature)
        pred_probs = self.mlc_trainer.mlc.get_tag_probs(visual_feature)

        self.assertTrue(torch.equal(
            pred_probs, result['probs']))

        self.assertTrue(torch.equal(
            self.loss(pred_score, label),
            result['loss']))


if __name__ == "__main__":
    unittest.main()
