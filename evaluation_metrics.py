
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
from abc import ABC
from typing import Optional, Sequence, TypeVar
from sklearn.metrics import roc_auc_score

import numpy as np

from texar.torch.run.metric.base_metric import SimpleMetric, StreamingMetric


Input = TypeVar('Input')
Value = TypeVar('Value')


class MultiLabelStreamingMetric(StreamingMetric[Input, Value]):
    r"""Base class of multi-label streaming metrics
    that support incremental computation.

    Keyword Args:
        num_label (int): Number of labels in total
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """

    def __init__(self, num_label: int, *args, **kwargs) -> None:
        self.num_label = num_label
        super().__init__(*args, **kwargs)

    def value(self) -> Value:
        raise NotImplementedError


class _MultiLabelConfusionMatrix(MultiLabelStreamingMetric[Input, Value], ABC):
    r"""Please refer details to ``sklearn.metrics.multilabel_confusion_matrix``
    """
    tp_sum: np.array
    pred_sum: np.array
    true_sum: np.array
    matrix: np.array

    def reset(self) -> None:
        super().reset()
        self.matrix = None
        self.tp_sum = np.zeros(self.num_label)
        self.pred_sum = np.zeros(self.num_label)
        self.true_sum = np.zeros(self.num_label)

    def add(self, predicted: Sequence[Input], labels: Sequence[Input]) -> None:
        r"""Update the confusion matrix using the results calculated for
        the current batch. Specifically, update
        self.tp_sum (total number of TP for each label)
        self.pred_sum (total number of TP + FP for each label)
        self.true_sum (total number of TP + FN for each label)

        Keyword Args:
            predicted: One-hot representation of the predicted results.
                Dimension [batch size, num_label]
            label_name: One-hot representation of the target labels.
                Dimension [batch size, num_label]
        """
        super().add(predicted, labels)
        predicted = np.array(predicted)
        labels = np.array(labels)
        sum_axis = 0

        true_and_pred = predicted * labels
        self.tp_sum += np.sum(true_and_pred, axis=sum_axis)
        self.pred_sum += np.sum(predicted, axis=sum_axis)
        self.true_sum += np.sum(labels, axis=sum_axis)

        fp = self.pred_sum - self.tp_sum
        fn = self.true_sum - self.tp_sum
        tp = self.tp_sum
        tn = self.count - tp - fp - fn
        self.matrix = np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)

    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) \
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


class MultiLabelConfusionMatrix(
    _MultiLabelConfusionMatrix[Input, Optional[np.ndarray]]
):
    r"""The confusion matrix is an evaluation metric for
    multi-label classification tasks.

    The value are averaged across different labels, with matrix[0, 0] represents
    TN, matrix[0, 1] represents FP, matrix[1, 0] represents FN,
    and matrix[1, 1] represents TP.

    Keyword Args:
        num_label (int): Number of labels in total
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """

    def value(self) -> Optional[np.ndarray]:
        # Dimension of self.matrix: [num_label]
        return np.mean(self.matrix, axis=0)

    def better(self, cur: Value, prev: Value) -> Optional[bool]:
        # Always return `None` to indicate values are uncomparable.
        return None


class MultiLabelPrecision(
    _MultiLabelConfusionMatrix[Input, Optional[np.ndarray]]
):
    r"""The precision metric for multi-label classification tasks. Precision is
    defined as the ratio of ``tp / (tp + fp)``, where ``tp`` is the number of
    true positives and ``fp`` is the number of false positives.
    The value are averaged across different labels.

    MultiLabelPrecision values are :class:`float` numbers between 0 and 1,
    with higher values being better.

    Keyword Args:
        num_label (int): Number of labels in total
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """
    def value(self) -> float:
        if self.count == 0:
            return np.zeros(self.num_label).mean()
        numerator = self.matrix[:, 1, 1] # tp
        denominator = self.matrix[:, 1, 1] + self.matrix[:, 0, 1] # tp + fp

        value = self._safe_divide(numerator, denominator)
        return value.mean()


class MultiLabelRecall(_MultiLabelConfusionMatrix[Input, Optional[np.ndarray]]):
    r"""The recall metric for multi-label classification tasks. Recall is
    defined as the ratio of ``tp / (tp + fn)``, where ``tp`` is the number of
    true positives and ``fn`` is the number of false negatives. The value are
    averaged across different labels.

    MultiLabelRecall values are :class:`float` numbers between 0 and 1,
    with higher values being better.

    Keyword Args:
        num_label (int): Number of labels in total
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """
    def value(self) -> float:
        if self.count == 0:
            return np.zeros(self.num_label).mean()

        numerator = self.matrix[:, 1, 1] # tp
        denominator = self.matrix[:, 1, 1] + self.matrix[:, 1, 0] # tp + fn

        value = self._safe_divide(numerator, denominator)
        return value.mean()


class MultiLabelF1(
    MultiLabelPrecision[Input], MultiLabelRecall[Input]
):
    r"""The F1 metric for multi-label classification tasks. MultiLabelF1
    is defined as the harmonic mean of MultiLabelPrecision and MultiLabelRecall.

    MultiLabelF1 requires both predicted values and labels.
    MultiLabelF1 values are :class:`float` numbers between 0 and 1,
    with higher values being better.

    Keyword Args:
        num_label (int): Number of labels in total
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """

    def value(self) -> float:
        precision = MultiLabelPrecision.value(self)
        recall = MultiLabelRecall.value(self)
        f1 = self._safe_divide(
            2 * precision * recall, precision + recall) # type: ignore
        # pylint: enable=protected-access
        return f1


class HammingLoss(MultiLabelStreamingMetric[Input, float]):
    r"""The HammingLoss metric for label classification tasks. HammingLoss is
    defined as the fraction of labels that are incorrectly predicted

    HammingLoss are :class:`float`numbers between 0 and 1,
    with lower values being better.

    Keyword Args:
        num_label (int): Number of labels in total
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """
    correct: np.float

    def reset(self) -> None:
        super().reset()
        self.correct = np.zeros(self.num_label)

    def add(self, predicted: Sequence[Input], labels: Sequence[Input]) -> None:
        super().add(predicted, labels)
        predicted = np.array(predicted)
        labels = np.array(labels)

        self.correct += np.sum(predicted == labels, axis=0)

    def value(self):
        if self.count == 0:
            return np.zeros(self.num_label).mean()

        return np.mean(self.count - self.correct) / self.count


class RocAuc(SimpleMetric[Input, float]):
    r"""Compute Area Under the Receiver Operating
    Characteristic Curve (ROC AUC) from prediction scores.
    Please refer details to sklearn.metrics.roc_auc_score"""
    def _value(self) -> Value:
        labels = np.stack(self.labels, axis=0)
        probs = np.stack(self.predicted, axis=0)
        try:
            score = roc_auc_score(labels, probs)
        except AttributeError:
            score = 0.
        return score
