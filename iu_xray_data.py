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
import os
import os.path as osp
import torch
import torch.utils.data

import texar as tx
from texar.torch.hyperparams import HParams
from texar.torch.data.data import DatasetBase
from texar.torch.data.data import DataSource
from texar.torch.data.data.data_iterators import DataIterator
from texar.torch.data import Vocab

from torchvision.datasets.folder import pil_loader
import torchvision.transforms as tfms

from forte.data.data_pack import DataPack
import config

from iu_xray.onto import Findings, FilePath, Tags


def collate_fn(data):
    r"""Collate the data

    NOTE: max_sentence_num denotes the maximum number
    of sentence in the current batch.
    max_word_num denotes the maximum number
    of words in one sentence in the current batch.

    the "+1" in "max_sentence_num + 1" is for the
    trainining of Stop control of the sentence LSTM.

    Returns:
        img_tensor (torch.Tensor): Dimension
            [batch size, channels, height, width]
        label (torch.Tensor): Dimension [batch size, num_tags]
        token_tensor (torch.Tensor): Dimension
            [batch size, max_sentence_num + 1, max_word_num]
        stop_prob (torch.Tensor): Dimension
            [batch size, max_sentence_num + 1]
    """
    img_tensor, label, caption_token, \
        max_word_num, sentence_num = zip(*data)

    img_tensor = torch.stack(img_tensor, 0)
    label = torch.stack(label, 0)

    max_sentence_num = max(sentence_num)
    max_word_num = max(max_word_num)
    batch_size = len(caption_token)

    # During training, we will iterate through the second dimension of
    # the token_tensor, the "+1" enables us to train the stop controller
    # to predict "Stop" when at the end of paragraph generation
    token_tensor = torch.zeros([batch_size, max_sentence_num + 1, max_word_num])
    stop_prob = torch.zeros([batch_size, max_sentence_num + 1])

    for i, token in enumerate(caption_token):
        for j, sentence in enumerate(token):
            token_tensor[i, j, :len(sentence)] = torch.Tensor(sentence)
            stop_prob[i][j] = len(sentence) > 0

    return img_tensor, label, token_tensor, stop_prob.to(torch.long)

class IU_XRay_DataSource(DataSource):
    r"""Dataset website here: https://openi.nlm.nih.gov/
    NOTE: For the image without findings, we set the default
    findings to be "the lungs are normal. "

    Args:
        hparams (dict or HParams, optional): IU_XRay_DataSource hyperparameters.
            Missing hyperparameters will be set to default values.
            See :meth:`default_hparams` for the hyperparameter
            structure and default values.
                * img_root (str): directory to the image root
                * label_path (str): directory to the txt file that contains
                    ground truth tags. Each line: [key, label]. Note that
                    {key}.png is the name of the corresponding parent image.
                * text_root (str): directory to the text root
                    (e.g. findings, impression)
                * vocab_path (str): directory to the txt file that contains all
                    the words for the vocabulary
                * transforms (str): data augmentation methods for input images
    """

    def __init__(self, hparams):
        self._hparams = HParams(hparams, self.default_hparams())
        self.img_root = self._hparams.img_root
        self.transforms = self.build_transform(self._hparams.transforms)
        self.text_root = self._hparams.text_root
        self.vocab = Vocab(self._hparams.vocab_path)
        self.pathologies = self._hparams.pathologies

    def __len__(self):
        r"""Returns the size of the data source"""
        return len(os.listdir(self.text_root))

    def __iter__(self):
        r"""Returns an iterator from data source"""
        for file_name in os.listdir(self.text_root):
            yield self.__getitem__(file_name)

    def __getitem__(self, file_name):
        r"""Fetch a data sample for a given key.

        Args:
            file_name (str): file name of the data sample

        Returns:
            img_tensor (torch.Tensor): Image tensor.
                Dimension [channels, height, width]
            label (torch.Tensor): Label for MLC task. Dimension [num_tags]
            caption_token: Tokenized texts.
                Dimension [sentence_num, max_word_num]
            max_word_num (int): maximum number of words in a sentence for
                this specific data sample
            sentence_num (int): number of sentences in this data sample
        """
        json_name = osp.join(self.text_root, file_name)
        with open(json_name, 'r') as f:
            datapack = DataPack.deserialize(f.read())

        # Get image tensor
        key = datapack.get_single(FilePath).img_study_path
        assert file_name.replace('.json', '') == key
        img_path = osp.join(self.img_root, key) + '.png'
        image_tensor = self.get_image(img_path, self.transforms)

        # Get the label for tag classification
        tags = datapack.get_single(Tags).content
        tag_index = [self.pathologies.index(tag) for tag in tags\
            if tag in self.pathologies]

        label = torch.zeros(len(self.pathologies))
        label[tag_index] = 1

        # Get the findings
        findings = datapack.get_single(Findings)

        if findings.content:
            caption = findings.content
        else:
            caption = 'the lungs are normal.'

        caption_token = list()
        max_word_num = 0

        for sentence in caption.split('. '):
            sentence = sentence.replace('.', '').split()
            if len(sentence) == 0 or len(sentence) == 1:
                continue
            tokens = self.vocab.map_tokens_to_ids_py(sentence).tolist()
            tokens.append(self.vocab.eos_token_id)

            max_word_num = max(max_word_num, len(tokens))
            caption_token.append(tokens)
        sentence_num = len(caption_token)

        return image_tensor, label, caption_token, max_word_num, sentence_num

    @staticmethod
    def build_transform(tsfm_list):
        r"""Build the data augmentation pipeline given the
        list of data augmentation strategies

        Args:
            tsfm_list (List[Tuple[str, Dict[str, Any]]]): A list of
                Tuples that specifies transformation functions. In each tuple,
                the first item specifies the name of the function. The
                second item specifies the configurations for the function.

        Return: (function) data augmentation pipeline to augment
            a given image.
        """
        t = []
        for func, args in tsfm_list:
            t.append(getattr(tfms, func)(**args))
        return tfms.Compose(t)

    def get_image(self, img_root, transforms):
        r"""Build the data augmentation pipeline given the
        list of data augmentation strategies

        Args:
            img_root (str): directory to the image file
            transforms (function): data augmentation functions
                to augment the given image.
        """
        # In this way, we can skip the ToPILImage in the data augmentations,
        # speeding up the data loading
        image = pil_loader(img_root)
        image_tensor = transforms(image)
        return image_tensor

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (Dict) default hyperparameters
        """
        hparams = {
            "img_root": None,
            "text_root": None,
            "vocab_path": None,
            "transforms": None,
            "pathologies": None,
        }
        return hparams


class IU_XRay_Dataset(DatasetBase):
    r"""Dataset for IU XRay

    Args:
        hparams (dict or HParams, optional): IU_XRay_Dataset hyperparameters.
            Missing hyperparameters will be set to default values.
            See :meth:`default_hparams` for the hyperparameter
            structure and default values.
                * datasource (Dict): hyperparameters for IU_XRay_DataSource
        device: device to transer the data to. Usage is the same as PyTorch.
            Please refer to `torch.device` for details.
    """
    def __init__(self, hparams=None, device="cuda:0"):
        self.source = IU_XRay_DataSource(hparams["datasource"])
        super().__init__(self.source, hparams, device)

    def collate(self, examples):
        r"""Collate the examples. Please refer details to the
        docstring of `collate_fn`

        Returns (tx.torch.data.Batch): A batch of data samples
        """
        img_tensor, label, token_tensor, stop_prob = collate_fn(examples)

        return tx.torch.data.Batch(
            len(examples),
            img_tensor=img_tensor,
            label=label,
            token_tensor=token_tensor,
            stop_prob=stop_prob,
        )

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (Dict) default hyperparameters
        """
        hparams = DatasetBase.default_hparams()
        hparams.update({
            "datasource": None,
        })
        return hparams


if __name__ == "__main__":
    dataset_hparams = config.dataset
    dataset = IU_XRay_Dataset(dataset_hparams["train"])
    # Dataloader
    dataset.to(torch.device('cpu'))
    train_loader = DataIterator(dataset)

    for batch in train_loader:
        print(batch)
        break
