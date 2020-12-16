from skimage.io import imread, imsave
import numpy as np
import os,sys,os.path
import pandas as pd
import torch
import torch.utils.data


class MIMICCXR_Dataset:
    """

    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(self, imgpath, csvpath, metacsvpath, views=["PA"], transform=None, data_aug=None,
                 flat_dir=True, seed=0):

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.MAXVAL = 255

        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]

        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = metacsvpath
        self.metacsv = pd.read_csv(self.metacsvpath)

        self.csv = self.csv.set_index(['subject_id', 'study_id'])
        self.metacsv = self.metacsv.set_index(['subject_id', 'study_id'])

        self.csv = self.csv.join(self.metacsv).reset_index()

        # Keep only the PA view.
        if type(views) is not list:
            views = [views]
        self.views = views

        idx_pa = self.csv["ViewPosition"].isin(views)
        self.csv = self.csv[idx_pa]

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        idx = int(idx)
        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])

        img_path = os.path.join(self.imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
        img = imread(img_path)

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]

        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)

        return {"img": img, "img_shape": img.shape, "lab": self.labels[idx], "idx": idx}


if __name__ == "__main__":
    dataset = MIMICCXR_Dataset(
        imgpath="/media/files/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
        csvpath="/media/files/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv",
        metacsvpath="/media/files/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv",
        transform=None, data_aug=None)
    # Dataloader
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    torch.manual_seed(0)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # disable data aug
    valid_dataset.data_aug = None

    # fix labels
    train_dataset.labels = dataset.labels[train_dataset.indices]
    valid_dataset.labels = dataset.labels[valid_dataset.indices]
    train_dataset.csv = dataset.csv.iloc[train_dataset.indices]
    valid_dataset.csv = dataset.csv.iloc[valid_dataset.indices]
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True,
                                               drop_last=True)

    for batch in train_loader:
        print(batch)