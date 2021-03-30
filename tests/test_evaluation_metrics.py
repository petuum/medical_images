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

"""This module tests for evaluation metrics."""

import unittest
import torch
import numpy as np
from texar.torch.data.data.data_iterators import DataIterator
from sklearn.metrics import roc_auc_score

from config import transforms, pathologies
from iu_xray_data import IU_XRay_Dataset
from models.cv_model import MLCTrainer
from evaluation_metrics import HammingLoss, MultiLabelConfusionMatrix, \
    MultiLabelF1, MultiLabelPrecision, MultiLabelRecall, RocAuc


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) \
        -> np.ndarray:
    # Credit: sklearn.metrics.classification._prf_divide
    if numerator.size == 1:
        if denominator == 0.0:
            return np.array(0.0)
        return numerator / denominator

    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1.0
    value = numerator / denominator
    return value


class TestEvaluationMetrics(unittest.TestCase):
    r"""
    Unit test for Evaluation Metrics that are used for
    multi-label classification
    """
    def setUp(self):
        self.batch_size = 2
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

        mlc_hparam = {
            'num_tags': len(pathologies),
        }
        self.mlc_trainer = MLCTrainer(mlc_hparam)

    def test_hamming_loss(self):
        r"""Test the HammingLoss implementations"""
        hm_loss = HammingLoss(
            num_label=self.num_label, pred_name='preds')

        simple_hm_loss = 0.
        num_sample = 0
        correct = np.zeros(self.num_label)
        for batch in self.loader:
            result = self.mlc_trainer(batch)
            predicted = np.array(result['preds'])
            label = np.array(batch.label)
            batch_size = batch.batch_size
            num_sample += batch_size

            hm_loss.add(predicted, label)
            correct += np.sum(predicted == label, axis=0)

        simple_hm_loss = (num_sample - correct).mean() / num_sample
        self.assertEqual(simple_hm_loss, hm_loss.value())

    def test_confusion_matrix_related(self):
        r"""Test the MultiLabelConfusionMatrix, MultiLabelPrecision
        MultiLabelRecall, and MultiLabelF1 implementation"""
        cf_matrix = MultiLabelConfusionMatrix(
            num_label=self.num_label, pred_name='preds')
        precision = MultiLabelPrecision(
            num_label=self.num_label, pred_name="preds")
        recall = MultiLabelRecall(
            num_label=self.num_label, pred_name="preds")
        f1 = MultiLabelF1(
            num_label=self.num_label, pred_name="preds")

        simple_cf_matrix = np.zeros([self.num_label, 2, 2])
        for batch in self.loader:
            result = self.mlc_trainer(batch)
            predicted = np.array(result['preds'])
            label = np.array(batch.label)
            batch_size = batch.batch_size

            true_and_pred = predicted * label
            tp_sum = np.sum(true_and_pred, axis=0)
            pred_sum = np.sum(predicted, axis=0)
            true_sum = np.sum(label, axis=0)

            fp = pred_sum - tp_sum
            fn = true_sum - tp_sum
            tp = tp_sum
            tn = batch_size - tp - fp - fn

            simple_cf_matrix += np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)

            cf_matrix.add(predicted, label)
            precision.add(predicted, label)
            recall.add(predicted, label)
            f1.add(predicted, label)

        tp = simple_cf_matrix[:, 1, 1]
        tp_fp = simple_cf_matrix[:, 1, 1] + simple_cf_matrix[:, 0, 1]
        tp_fn = simple_cf_matrix[:, 1, 1] + simple_cf_matrix[:, 1, 0]
        simple_cf_matrix = np.mean(simple_cf_matrix, axis=0)

        simple_precision = safe_divide(tp, tp_fp).mean()
        simple_recall = safe_divide(tp, tp_fn).mean()
        simple_f1 = safe_divide(
            2 * simple_precision * simple_recall,
            simple_precision + simple_recall
        )

        self.assertTrue(np.array_equal(
            simple_cf_matrix, cf_matrix.value()
        ))

        self.assertEqual(simple_precision, precision.value())

        self.assertEqual(simple_recall, recall.value())

        self.assertEqual(simple_f1, f1.value())

    def test_roc_auc(self):
        r"""Test the RocAuc implementation

        NOTE: To test the ROC AUC, we have to modify the label
        to make sure there is at list one sample for one label
        is active/negative. The goal here is to make sure the
        calculation is correct, i.e., the results from our
        implementation should match some straight forward calculation.
        """
        roc_auc = RocAuc(pred_name='preds')

        predicted_list = []
        label_list = []
        for batch in self.loader:
            result = self.mlc_trainer(batch)
            predicted = np.array(result['preds'])
            label = np.array(batch.label)
            label[0] = np.ones_like(label[0])

            roc_auc.add(predicted, label)
            predicted_list.append(predicted)
            label_list.append(label)

        label = np.concatenate(label_list, axis=0)
        predicted = np.concatenate(predicted_list, axis=0)

        simple_roc_auc = roc_auc_score(
            label.astype(np.int), predicted)

        self.assertEqual(simple_roc_auc, roc_auc.value())


if __name__ == "__main__":
    unittest.main()
