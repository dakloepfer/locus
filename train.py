import math
import argparse
import pprint
import torch
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
    DeviceStatsMonitor,
    TQDMProgressBar,
)
from lightning import Trainer


from src.default_config.default import get_cfg_defaults
from src.utils.misc import (
    get_rank_zero_only_logger,
    add_pl_argparse_args,
    pl_from_argparse_args,
)
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.module import PLModule

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("main_cfg_path", type=str, help="main config path")
    parser.add_argument("data_cfg_path", type=str, help="data config path")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data/locus_logs",
    )
    parser.add_argument("--exp_name", type=str, default="default_exp_name")
    parser.add_argument("--device_list", type=int, nargs="+", default=None)
    parser.add_argument(
        "--batch_size", type=int, default=16, help="total batch size (across all GPUs)"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--pin_memory",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        default=True,
        help="whether loading data to pinned memory or not",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="pretrained checkpoint path",
    )
    parser.add_argument(
        "--disable_ckpt",
        action="store_true",
        help="disable checkpoint saving (useful for debugging).",
    )
    parser.add_argument(
        "--profiler_name",
        type=str,
        default=None,
        help="options: [inference, pytorch], or leave it unset",
    )
    parser.add_argument(
        "--tqdm_refresh_rate",
        type=int,
        default=1,
        help="refresh rate for tqdm progress bar",
    )

    parser = add_pl_argparse_args(Trainer, parser)
    parser = add_pl_argparse_args(PLModule, parser)
    parser = add_pl_argparse_args(TensorBoardLogger, parser)

    args = parser.parse_args()

    return args


def main():
    loguru_logger.info(f"Number of available GPUs: {torch.cuda.device_count()}")
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    config.freeze()

    # lightning module
    profiler = build_profiler(args)
    model = PLModule(
        config, ckpt_path=args.ckpt_path, devices=args.device_list, profiler=profiler
    )
    loguru_logger.info(f"LightningModule initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"DataModule initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir=args.save_dir,
        name=args.exp_name,
        default_hp_metric=False,
    )
    ckpt_dir = Path(logger.log_dir) / "checkpoints"

    # Callbacks
    ckpt_callback = ModelCheckpoint(
        monitor="loss",
        verbose=True,
        save_top_k=10,
        mode="min",
        save_last=True,
        dirpath=str(ckpt_dir),
        filename="{epoch}-{avg_loss:.3f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=10)
    tqdm_progress_bar = TQDMProgressBar(refresh_rate=args.tqdm_refresh_rate)
    callbacks = [
        lr_monitor,
        model_summary,
        tqdm_progress_bar,
    ]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)
    if args.profiler_name is not None:
        device_stats_monitor = DeviceStatsMonitor(cpu_stats=True)
        callbacks.append(device_stats_monitor)

    # Lightning Trainer
    trainer = pl_from_argparse_args(
        Trainer,
        args,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        accumulate_grad_batches=config.TRAINER.ACCUMULATE_GRAD_BATCHES,
        callbacks=callbacks,
        logger=logger,
        reload_dataloaders_every_n_epochs=False,  # avoid repeated samples!
        profiler=profiler,
        detect_anomaly=args.detect_anomaly,
        devices=1,  # only use one GPU for the trainer
    )
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
