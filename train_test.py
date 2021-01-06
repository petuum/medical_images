import torch
import argparse, sys
from network import ClassifierWrapper
from texar.torch.run import *

from mimic_dataset import MIMICCXR_Dataset
from config_mimic_test import dataset as hparams_dataset
from pathlib import Path


# args
parser = argparse.ArgumentParser(description="Train MIMIC model")
parser.add_argument(
    '--save_dir',
    type=str,
    help='Place to save training results',
    default='exp_default/'
)
parser.add_argument(
    '--output_dir',
    type=str,
    help='Place to save logs results',
    default='output_default/'
)
parser.add_argument(
    '--grad_clip',
    type=float,
    help='Gradient clip value',
    default=None
)
parser.add_argument(
    '--display_steps',
    type=int,
    help='log result every * steps',
    default=1
)
parser.add_argument(
    '--max_train_steps',
    type=int,
    help='Maximum number of steps to train',
    default=1000000
)
args = parser.parse_args()

# Dataloader
datasets = {split: MIMICCXR_Dataset(hparams=hparams_dataset[split])
            for split in ["train", "val", "test"]}
print("done with loading")
# model
model = ClassifierWrapper()
output_dir = Path(args.output_dir)
# Trainer
executor = Executor(
    model=model,
    train_data=datasets["train"],
    valid_data=datasets["val"],
    test_data=datasets["test"],
    checkpoint_dir=args.save_dir,
    save_every=cond.validation(better=True),
    train_metrics=[("loss", metric.RunningAverage(args.display_steps))],
    optimizer={"type": torch.optim.Adam},
    grad_clip=args.grad_clip,
    log_every=cond.iteration(args.display_steps),
    log_destination=[sys.stdout, output_dir / "log.txt"],
    validate_every=cond.epoch(1),
    valid_metrics=[metric.Accuracy[float](pred_name="preds", label_name="target"),
                    metric.Precision[float](pred_name="preds", label_name="target",mode='macro'),
                   metric.Recall[float](pred_name="preds", label_name="target",mode='macro'),
                   metric.F1[float](pred_name="preds", label_name="target",mode='macro'),
                   metric.ConfusionMatrix[float](pred_name="preds", label_name="target"),
                   ("loss", metric.Average())],
    plateau_condition=[
        cond.consecutive(cond.validation(better=False), 2)],
    action_on_plateau=[
        action.early_stop(patience=2),
        action.reset_params(),
        action.scale_lr(0.8)],
    stop_training_on=cond.iteration(args.max_train_steps),
    test_mode='eval',
    tbx_logging_dir='tbx_folder',
    test_metrics=[metric.Accuracy[float](pred_name="preds", label_name="target"),
                    metric.Precision[float](pred_name="preds", label_name="target",mode='macro'),
                    metric.Recall[float](pred_name="preds", label_name="target",mode='macro'),
                    metric.F1[float](pred_name="preds", label_name="target",mode='macro'),
                    metric.ConfusionMatrix[float](pred_name="preds", label_name="target"),
                   ("loss", metric.Average())]
)

executor.train()
executor.test()
