import numpy as np
import os,sys,os.path
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from config_mimic import dataset as hparams_dataset



class preprocess_mimic():
    def __init__(self):
        self.pathologies = sorted(hparams_dataset["pathologies"])

        self.imgpath = hparams_dataset["imgpath"]
        self.csvpath = hparams_dataset["csvpath"]
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = hparams_dataset["metacsvpath"]
        self.metacsv = pd.read_csv(self.metacsvpath)
        self.csv = self.csv.set_index(['subject_id', 'study_id'])
        self.metacsv = self.metacsv.set_index(['subject_id', 'study_id'])

        self.csv = self.csv.join(self.metacsv).reset_index()

        self.mode = hparams_dataset["mode"]
        self.views = hparams_dataset["views"]
        self.prepare_csv_entries()
        self.save_csv()

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
            for col_i in self.pathologies:
                new_csv_column.append(col_i)
            print(new_csv_column)
            self.csv = self.csv.filter(new_csv_column, axis=1)
        else:
            # grouping by study id
            self.csv['study'] = self.csv.apply(lambda x: str(Path(x['path']).parent), axis=1)
            self.csv.set_index(['study'], inplace=True)
            path_column_idx = self.csv.columns.get_loc('path')
            aggs = {self.csv.columns[path_column_idx]: lambda x: ','.join(x.astype(str))}
            aggs.update({x: 'mean' for x in self.pathologies})
            self.csv = self.csv.groupby(['study']).agg(aggs).reset_index(0, drop=True)

    def save_csv(self):
        dataset_indices = list(range(len(self.csv)))
        np.random.shuffle(dataset_indices)
        self.train_indices, test_indices = train_test_split(dataset_indices, test_size=0.3)
        self.val_indices, self.test_indices = train_test_split(test_indices, test_size=0.66)
        train_csv = self.csv.iloc[self.train_indices]
        train_csv.to_csv('mimic_train.csv', header=False, index=False)
        val_csv = self.csv.iloc[self.val_indices]
        val_csv.to_csv('mimic_val.csv', header=False, index=False)
        test_csv = self.csv.iloc[self.test_indices]
        test_csv.to_csv('mimic_test.csv', header=False, index=False)

def combine_path(imgpath, row):
    subjectid = str(row["subject_id"])
    studyid = str(row["study_id"])
    dicom_id = str(row["dicom_id"])
    img_path = os.path.join(imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")

    return img_path

if __name__ == "__main__":
    preprocess_mimic()