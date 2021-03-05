import os
import os.path as osp
import shutil
import random
import argparse

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--data-dir", type=str,
                        default="/data/jiachen.li/iu_xray/ecgen-radiology/",
                        help="Data directory to read the xml files from")
    PARSER.add_argument("--target-folder", type=str,
                        default="/home/jiachen.li/data/ecgen-radiology-split/",
                        help="Data directory to save xml files that split into"
                        "tran, val and test")

    ARGS = PARSER.parse_args()

    # Set the seed for random so that running this file
    # multiple times will not unexpectedly generate new
    # files to each subset
    random.seed(1858)

    # Split the original report into train, val and test
    all_files = os.listdir(ARGS.data_dir)
    file_shuffle = random.sample(all_files, k=len(all_files))
    val_files = file_shuffle[:250]
    test_files = file_shuffle[250:500]
    train_files = file_shuffle[500:]

    # Generate the validation files
    target_folder = osp.join(ARGS.target_folder, 'val')
    if not osp.exists(target_folder):
        os.mkdir(target_folder)

    for i, f in enumerate(val_files):
        source = osp.join(ARGS.data_dir, f)
        target = osp.join(target_folder, f)
        shutil.copy(source, target)

    assert len(os.listdir(target_folder)) == len(val_files)

    # Generate the test files
    target_folder = osp.join(ARGS.target_folder, 'test')
    if not osp.exists(target_folder):
        os.mkdir(target_folder)

    for f in test_files:
        source = osp.join(ARGS.data_dir, f)
        target = osp.join(target_folder, f)
        shutil.copy(source, target)

    assert len(os.listdir(target_folder)) == len(test_files)

    # Generate the train files
    target_folder = osp.join(ARGS.target_folder, 'train')
    if not osp.exists(target_folder):
        os.mkdir(target_folder)

    for f in train_files:
        source = osp.join(ARGS.data_dir, f)
        target = osp.join(target_folder, f)
        shutil.copy(source, target)

    assert len(os.listdir(target_folder)) == len(train_files)
