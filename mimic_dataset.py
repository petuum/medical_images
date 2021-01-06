from texar.torch.data.data import DatasetBase
from mimic_cxr import MIMICCXR_DataSource
import texar as tx
import torch
import numpy as np
import imageio


class MIMICCXR_Dataset(DatasetBase):
    def __init__(self, hparams=None, device="cuda:0"):
        self.source = MIMICCXR_DataSource(hparams)
        super().__init__(self.source, hparams, device)

    def process(self, raw_example):
        return {
            "image": raw_example[0],
            "target": raw_example[1],
            "channels": raw_example[2]
        }

    def collate(self, examples):
        # `examples` is a list of objects returned from the
        # `process` method. These data examples should be collated
        # into a batch.

        # `images` is a `tensor` of input images, storing the transformed
        # images for each example in the batch.
        images = [ex["image"] for ex in examples]
        images = torch.stack([i for i in images])
        channels = [ex["channels"] for ex in examples]
        # `target` is the one hot encoding of the labels with the size of
        # number of classes, stack into the batch
        target = [ex["target"] for ex in examples]
        target = torch.stack([i for i in target])
        return tx.torch.data.Batch(
            len(examples),
            image=images,
            target=target,
            channels=channels)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters.
        See the specific subclasses for the details.
        """
        hparams = DatasetBase.default_hparams()
        hparams.update({
            "transforms": None,
            "processed_csv": None,
            "mode": None,
            "batch_size": 1,
            "shuffle": False,
            "shuffle_buffer_size": 32
        })
        return hparams
