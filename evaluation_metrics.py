from abc import ABC
from typing import Optional, Sequence, TypeVar, List, Dict, Tuple
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.metrics import roc_auc_score

import numpy as np

from texar.torch.run.metric.base_metric import StreamingMetric, SimpleMetric


Input = TypeVar('Input')
Value = TypeVar('Value')


class MultiLabelStreamingMetric(StreamingMetric[Input, Value]):
    r"""Base class of multi-label streaming metrics. Streaming metrics are metrics that
    support incremental computation. The value of the metric may be queried
    before all data points have been added, and the computation should not be
    expensive.

    The default implementation of :meth:`add` only keeps track of the number of
    data points added. You should override this method.
    """

    def __init__(self, num_label: int, *args, **kwargs) -> None:
        self.num_label = num_label
        super().__init__(*args, **kwargs)
    
    def value(self) -> Value:
        raise NotImplementedError


class _MultiLabelConfusionMatrix(MultiLabelStreamingMetric[Input, Value], ABC):
    tp_sum: np.array
    pred_sum: np.array
    true_sum: np.array

    def reset(self) -> None:
        super().reset()
        self.matrix = None
        self.tp_sum = np.zeros(self.num_label)
        self.pred_sum = np.zeros(self.num_label)
        self.true_sum = np.zeros(self.num_label)

    def add(self, predicted: Sequence[Input], labels: Sequence[Input]) -> None:
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


class MultiLabelConfusionMatrix(_MultiLabelConfusionMatrix[Input, Optional[np.ndarray]]):
    r"""The confusion matrix is an evaluation metric for classification tasks.

    Confusion matrix is a :class:`~texar.torch.run.metric.StreamingMetric`,
    requires both predicted values and labels. Confusion matrix values are NumPy
    arrays, with no clear definition of "better". Comparison of two confusion
    matrices are not meaningful.

    The value indexed at ``(i, j)`` of the confusion matrix is the number of
    data points whose predicted label is `i` and whose ground truth label is
    `j`. Labels are internally mapped to indices.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """

    def value(self) -> Optional[np.ndarray]:
        return np.mean(self.matrix, axis=0)

    def better(self, cur: Value, prev: Value) -> Optional[bool]:
        # Always return `None` to indicate values are uncomparable.
        return None


class MultiLabelPrecision(_MultiLabelConfusionMatrix[Input, Optional[np.ndarray]]):
    r"""The precision metric for evaluation classification tasks. Precision is
    defined as the ratio of ``tp / (tp + fp)``, where ``tp`` is the number of
    true positives and ``fp`` is the number of false positives.

    Precision is a :class:`~texar.torch.run.metric.StreamingMetric`, requires
    both predicted values and labels. Precision values are :class:`float`
    numbers between 0 and 1, with higher values being better.

    Args:
        mode (str): The mode for computing averages across multiple labels.
            Defaults to ``"binary"``. Available options include:

            - ``"binary"``: Only report results for the class specified by
              :attr:`pos_label`. This is only meaningful for binary
              classification tasks.
            - ``"micro"``: Return the precision value computed using globally
              counted true positives and false positives.
            - ``"macro"``: Return the unweighted average of precision values for
              each label.
            - ``"weighted"``: Return the average of precision values for each
              label, weighted by the number of true instances for each label.
        pos_label (str, optional): The label for the positive class. Only used
            if :attr:`mode` is set to ``"binary"``.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """
    
    def value(self) -> float:
        if self.count == 0:
            return np.zeros(self.num_label)
        numerator = self.matrix[:, 1, 1] # tp
        denominator = self.matrix[:, 1, 1] + self.matrix[:, 0, 1] # tp + fp

        value = self._safe_divide(numerator, denominator)
        return value.mean()


class MultiLabelRecall(_MultiLabelConfusionMatrix[Input, Optional[np.ndarray]]):
    r"""The recall metric for evaluation classification tasks. Precision is
    defined as the ratio of ``tp / (tp + fn)``, where ``tp`` is the number of
    true positives and ``fn`` is the number of false negatives.

    Recall is a :class:`~texar.torch.run.metric.StreamingMetric`, requires both
    predicted values and labels. Recall values are :class:`float` numbers
    between 0 and 1, with higher values being better.

    Args:
        mode (str): The mode for computing averages across multiple labels.
            Defaults to ``"binary"``. Available options include:

            - ``"binary"``: Only report results for the class specified by
              :attr:`pos_label`. This is only meaningful for binary
              classification tasks.
            - ``"micro"``: Return the recall value computed using globally
              counted true positives and false negatives.
            - ``"macro"``: Return the unweighted average of recall values for
              each label.
            - ``"weighted"``: Return the average of recall values for each
              label, weighted by the number of true instances for each label.
        pos_label (str, optional): The label for the positive class. Only used
            if :attr:`mode` is set to ``"binary"``.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """
    def value(self) -> float:
        if self.count == 0:
            return np.zeros(self.num_label)

        numerator = self.matrix[:, 1, 1] # tp
        denominator = self.matrix[:, 1, 1] + self.matrix[:, 1, 0] # tp + fn

        value = self._safe_divide(numerator, denominator)
        return value.mean()


class MultiLabelF1(MultiLabelPrecision[Input], MultiLabelRecall[Input]):
    r"""The F1 metric for evaluation classification tasks. F1 is defined as the
    harmonic mean of precision and recall.

    F1 is a requires both predicted values and labels. 
    F1 values are :class:`float` numbers between 0 and 1, with higher values being better.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """

    def value(self) -> float:
        # pylint: disable=protected-access
        precision = MultiLabelPrecision.value(self)
        recall = MultiLabelRecall.value(self)

        f1 = self._safe_divide(2 * precision * recall, precision + recall)
        # pylint: enable=protected-access
        return f1


class HammingLoss(MultiLabelStreamingMetric[Input, float]):
    r"""The accuracy metric for evaluation classification tasks. Accuracy is
    defined as the ratio of correct (exactly matching) predictions out of all
    predictions.

    Accuracy is a :class:`~texar.torch.run.metric.StreamingMetric`, requires
    both predicted values and labels. Accuracy values are :class:`float`
    numbers between 0 and 1, with higher values being better.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset. Defaults to ``"label"``.
    """
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
            return np.zeros(self.num_label)

        return np.mean(self.count - self.correct) / self.count


class RocAuc(SimpleMetric[Input, float]):
    def _value(self) -> Value:
        r"""Compute the metric value. This function is called in
        :meth:`texar.torch.run.metric.SimpleMetric.value` and the output is
        cached. This prevents recalculation of metrics which may be time
        consuming.

        Returns:
            The metric value.
        """
        labels = np.concatenate(self.labels, axis=0)
        predicted = np.concatenate(self.predicted, axis=0)

        score = roc_auc_score(labels.astype(np.int), predicted)
        return score


if __name__ == "__main__":
    m = MultiLabelStreamingMetric(num_label=2, pred_name="preds", label_name="target")
    b = HammingLoss(num_label=2, pred_name="preds", label_name="target")
    c = MultiLabelConfusionMatrix(num_label=2, pred_name="preds", label_name="target")
    p = MultiLabelPrecision(num_label=2, pred_name="preds", label_name="target")
    r = MultiLabelRecall(num_label=2, pred_name="preds", label_name="target")
    f = MultiLabelF1(num_label=2, pred_name="preds", label_name="target")

    from sklearn.datasets import make_multilabel_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.metrics import roc_auc_score
    X, y = make_multilabel_classification(random_state=0)
    inner_clf = LogisticRegression(solver="liblinear", random_state=0)
    clf = MultiOutputClassifier(inner_clf).fit(X, y)

    y_score = np.transpose([y_pred[:, 1] for y_pred in clf.predict_proba(X)])
    print(y)
    print(y_score)
    print(roc_auc_score(y, y_score, average=None))
