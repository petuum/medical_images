import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as tfms
import imageio
from texar.torch.hyperparams import HParams
from texar.torch.data.data import DatasetBase
from texar.torch.data.data import DataSource


class MIMICCXR_DataSource(DataSource):
    """
    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(self, hparams):
        self._hparams = HParams(hparams, self.default_hparams())
        self.mode = self._hparams["mode"]
        self.csvpath = self._hparams["processed_csv"]
        self.csv = pd.read_csv(self.csvpath)
        self.transforms = self.build_transform(self._hparams['transforms'])

    def __len__(self):
        return len(self.csv)

    def __iter__(self):
        for index, row in self.csv.iterrows():
            yield row

    def __getitem__(self, index):
        index = int(index)
        def get_entries(index):
            df = self.csv.iloc[index]
            paths = [x for x in df[0].split(',')]
            label = df[1:].tolist()
            return paths, label

        if self.mode == "PER_IMAGE":
            img_paths, label = get_entries(index)
            image_tensor = self.get_image(img_paths[0], self.transforms)
            target_tensor = torch.FloatTensor(label)
            channels = 3
        else:  # PER_STUDY
            img_paths, label = get_entries(index)
            image_tensor = self.get_study(img_paths, self.transforms)
            target_tensor = torch.FloatTensor(label)
            channels = len(img_paths)

        return image_tensor, target_tensor, channels

    @staticmethod
    def build_transform(tsfm_list):
        t = []
        for func, args in tsfm_list:
            t.append(getattr(tfms, func)(**args))
        return tfms.Compose(t)

    def get_study(self, img_paths, shuffle=False):
        if shuffle:
            img_paths = np.random.permutation(img_paths).tolist()
        ret = []
        for i, img_path in enumerate(img_paths):
            image = imageio.imread(img_path, as_gray=True)
            ret.append(self.transforms(image))
        return ret

    def get_image(self, img_path, transforms):
        if self._hparams["input_channel"] == "GRAY":
            image = imageio.imread(img_path, as_gray=True)
        else:
            image = imageio.imread(img_path, as_gray=False, pilmode="RGB")
        image_tensor = transforms(image)
        return image_tensor

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
            "input_channel": "RGB"
        })
        return hparams



if __name__ == "__main__":
    hparams = config.dataset
    dataset = MIMICCXR_DataSource(hparams)
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
