import numpy as np
import os,sys,os.path
import pandas as pd
import torch
import torch.utils.data
from pathlib import Path
import torchvision.transforms as tfms
import imageio
from PIL import Image
from texar.torch.hyperparams import HParams
from texar.torch.data.data import DatasetBase
import argparse
import importlib
from typing import Any, Dict, Tuple


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str, default='config_kim',
    help='The config to use.')
args = parser.parse_args()
config: Any = importlib.import_module(args.config)

MIN = 256
MAX_CHS = 11
MEAN = 0.4
STDEV = 0.2

cxr_train_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.RandomAffine((-5, 5), translate=None, scale=None, shear=(0.9, 1.1)),
    tfms.RandomResizedCrop((MIN, MIN), scale=(0.5, 0.75), ratio=(0.95, 1.05), interpolation=Image.LANCZOS),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
])


cxr_test_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize(MIN, Image.LANCZOS),
    tfms.CenterCrop(MIN),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
])


def get_study(img_paths, transforms):
    image_tensor = torch.randn(MAX_CHS, MIN, MIN) * STDEV + MEAN
    rand = transforms == cxr_train_transforms
    rand_idx = torch.randperm(len(img_paths))
    for i, img_path in enumerate(img_paths):
        image = imageio.imread(img_path, as_gray=True)
        j = rand_idx[i] if rand else i
        image_tensor[j, :, :] = transforms(image)
    return image_tensor

def get_image(img_path, transforms):
    image = imageio.imread(img_path, as_gray=True)
    image_tensor = transforms(image)
    return image_tensor



class MIMICCXR_Dataset:
    """

    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(self, hparams):
        self._hparams = HParams(hparams, self.default_hparams())
        np.random.seed(hparams["seed"])  # Reset the seed so all runs are the same.
        self.pathologies = sorted(hparams["pathologies"])

        self.imgpath = hparams["imgpath"]
        self.csvpath = hparams["csvpath"]
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = hparams["metacsvpath"]
        self.metacsv = pd.read_csv(self.metacsvpath)

        self.csv = self.csv.set_index(['subject_id', 'study_id'])
        self.metacsv = self.metacsv.set_index(['subject_id', 'study_id'])

        self.csv = self.csv.join(self.metacsv).reset_index()

        self.mode = hparams["mode"]
        self.views = hparams["views"]
        self.prepare_csv_entries()
        self.transforms = cxr_train_transforms

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        index = int(index)
        def get_entries(index):
            df = self.csv.iloc[index]
            paths = [x for x in df[0].split(',')]
            label = df[1:].tolist()
            return paths, label

        if self.mode == "PER_IMAGE":
            img_paths, label = get_entries(index)
            image_tensor = get_image(img_paths[0], self.transforms)
            target_tensor = torch.FloatTensor(label)
            channels = 1
        else:  # PER_STUDY
            img_paths, label = get_entries(index)
            image_tensor = get_study(img_paths, self.transforms)
            target_tensor = torch.FloatTensor(label)
            channels = len(img_paths)

        return image_tensor, target_tensor, channels


    def prepare_csv_entries(self):
        healthy = self.csv["No Finding"] == 1
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[self.csv[pathology] == 1, pathology] = 1.0
                self.csv.loc[healthy, pathology] = 0.0
                self.csv.loc[self.csv[pathology] == -1, pathology] = 0.0
                self.csv.loc[pd.isna(self.csv[pathology]), pathology] = 0.0

        self.csv["path"] = self.csv.apply(lambda row: combine_path(self.imgpath, row), axis=1)

        if self.mode == 'PER_IMAGE':
            # Keep only the PA view.
            idx_pa = self.csv["ViewPosition"].isin(self.views)
            self.csv = self.csv[idx_pa]
            new_csv_column = ['path']
            new_csv_column.append(self.pathologies)
            self.csv = self.csv.filter(new_csv_column,axis=1)
        else: # MODE='PER_STUDY'
            # grouping by study id
            self.csv['study'] = self.csv.apply(lambda x: str(Path(x['path']).parent), axis=1)
            self.csv.set_index(['study'], inplace=True)
            path_column_idx = self.csv.columns.get_loc('path')
            aggs = {self.csv.columns[path_column_idx]: lambda x: ','.join(x.astype(str))}
            aggs.update({x: 'mean' for x in self.pathologies})
            self.csv = self.csv.groupby(['study']).agg(aggs).reset_index(0, drop=True)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters.
        See the specific subclasses for the details.
        """
        hparams = DatasetBase.default_hparams()
        hparams.update({
            "imgpath": None,
            "csvpath": None,
            "metacsvpath": None,
            "mode": None,
            "views": [],
            "seed": 0,
            "pathologies": []
        })
        return hparams



def combine_path(imgpath, row):
    subjectid = str(row["subject_id"])
    studyid = str(row["study_id"])
    dicom_id = str(row["dicom_id"])
    img_path = os.path.join(imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")

    return img_path


if __name__ == "__main__":
    hparams = config.dataset
    dataset = MIMICCXR_Dataset(hparams)
    # Dataloader
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    torch.manual_seed(0)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # disable data aug
    valid_dataset.data_aug = None

    train_dataset.csv = dataset.csv.iloc[train_dataset.indices]
    valid_dataset.csv = dataset.csv.iloc[valid_dataset.indices]
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=10,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True,
                                               drop_last=True)

    for batch in train_loader:
        print(batch)
