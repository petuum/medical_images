# Note: Here we freeze the feature extractor, and only tune the MLC.
# We used the default learning rate of Adam to train the MLC.

# https://arxiv.org/pdf/2004.12274.pdf finetunes the feature extractor
# on the ChestX-ray 14 dataset.
from typing import Dict, Any
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

# Texar Library
from texar.torch import ModuleBase


class SimpleFusionEncoder(ModuleBase):
    r"""Visual feature extractor. Implementation is adapted
    from
    https://gitlab.int.petuum.com/shuxin.yao/image_report_generation/blob/master/implementation/Encoders/Encoders.py
    Base encoder is set to be DenseNet121. Pretrained weights
    are reused from https://github.com/berneylin/chexnet

    NOTE: The output features are not a vector. Instead, we
    treat the output from the feature layer of the densenet
    as the features, and reshape it [batch size, outfeatures, -1]
    """
    def __init__(self):
        super().__init__()

        self.cnn = tvm.densenet121(pretrained=True)
        self._load_from_ckpt()
        self.out_features = self.cnn.classifier.in_features

    def _load_from_ckpt(self, ckpt='./model.pth.tar'):
        # Only support the specific chekpoint
        assert ckpt == './model.pth.tar'
        if osp.exists(ckpt):
            pretrained_weight = torch.load(ckpt)['state_dict']
            new_state_dict = {}
            prefix = 'module.dense_net_121.'
            for k, v in pretrained_weight.items():
                if 'classifier' not in k:
                    new_k = k[len(prefix):]
                    new_state_dict[new_k] = v

            msg = self.cnn.load_state_dict(new_state_dict, strict=False)
            assert set(msg.missing_keys) == {
                "classifier.weight",
                "classifier.bias"
            }, set(msg.missing_keys)

    def forward(self, images):
        r"""
        Extract visual features from the input images

        Args:
            * images (torch.Tensor): dimension
            [batch size, channels, height, width]
        Returns:
            * res (torch.Tensor): dimension
            [batch size, out_features, 49 = 7 * 7]
        """
        batch_size = images.shape[0]
        res = self.cnn.features(images)
        res = res.view(batch_size, self.out_features, -1)

        return res


class MLC(ModuleBase):
    r"""Multilabel classifier
    Args:
        hparams (dict or HParams, optional): MLC hyperparameters.
            Missing hyperparameters will be set to default values.
            See :meth:`default_hparams` for the hyperparameter structure
            and default values.
                * fc_in_features (int): Dimension of input visual features
                * num_tags (int): Number of tags in total
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        self.classifier = nn.Linear(
            in_features=self.hparams.fc_in_features,
            out_features=self.hparams.num_tags)

        # As per the wingspan project
        nn.init.kaiming_normal_(
            self.classifier.weight, mode='fan_in')
        self.classifier.bias.data.fill_(0)

    def forward(self, visual_feature):
        r"""Generate logits (scores) for all tags given
        the input visual_feature
        Args:
            visual_feature (torch.Tensor): dimension
                [batch size, num_visual_features, visual_dim]
        Returns:
            tag_scores (torch.Tensor): scores for all tags.
                Dimension [batch size, num_tags]
        """
        flat_feature = F.avg_pool1d(
            visual_feature,
            visual_feature.size(-1)
        ).squeeze(-1)

        tag_scores = self.classifier(flat_feature)

        return tag_scores

    def get_tag_probs(self, visual_feature):
        r"""Generate probability distributions for all tags given
        the input visual_feature
        Args:
            visual_feature (torch.Tensor): dimension
                [batch size, num_visual_features, visual_dim]
        Returns:
            tag_probs (torch.Tensor): probability distributions
                for all tags. Dimension [batch size, num_tags]
        """
        tag_scores = self.forward(visual_feature)
        tag_probs = torch.sigmoid(tag_scores)
        return tag_probs

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
            'num_tags': 210,
            'fc_in_features': 1024,
        }


class MLCTrainer(ModuleBase):
    r""" Trainer for the Multilabel classifier
    Args:
        hparams (dict or HParams, optional): MLCTrainer hyperparameters.
            Missing hyperparameters will be set to default values.
            See :meth:`default_hparams` for the hyperparameter structure
            and default values.
                * num_tags (int): Number of tags in total
                * threshold (float): Threshold to determine if a tag is active
                    or not
                * train_encoder (bool): indicate whether keep training
                    the encoder or not
    """
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)
        self.extractor = SimpleFusionEncoder()
        hparams_mlc = {
            'num_tags': self.hparams.num_tags,
            'fc_in_features': self.extractor.out_features,
        }

        self.mlc = MLC(hparams_mlc)

        self.threshold = self.hparams.threshold
        self.train_encoder = self.hparams.train_encoder

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        r"""Generate logits (scores) for all tags given
        the input visual_feature
        Args:
            batch (tx.torch.data.Batch[str, Union[torch.Tensor, int]]):
                * batch_size: batch size
                * label: Dimension [batch size, num_tags]
                * img_tensor: Dimension [batch size, channels, height, width]
                * token_tensor: Dimension
                    [batch size, max_sentence_num + 1, max_word_num]
                * stop_prob: Dimension [batch size, max_sentence_num + 1]
        Returns:
            loss (torch.float): classification loss
            preds (torch.Tensor): indicators of whether a tag
                is active. Dimension [batch size, num_tags]
            probs (torch.Tensor): probability distributions
                for all tags. Dimension [batch size, num_tags]
        """
        if self.train_encoder:
            visual_feature = self.extractor(batch.img_tensor)
        else:
            with torch.no_grad():
                visual_feature = self.extractor(batch.img_tensor)

        tag_scores = self.mlc(visual_feature)

        loss = self.loss(tag_scores, batch.label)

        probs = torch.sigmoid(tag_scores)
        preds = (probs > self.threshold).to(torch.float)
        return {"loss": loss, "preds": preds, "probs": probs}

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
            'num_tags': 210,
            'threshold': 0.5,
            'train_encoder': False
        }


if __name__ == "__main__":
    m = MLCTrainer()
