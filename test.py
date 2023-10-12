import os
import argparse
import pprint
from tqdm import tqdm
import torch
from distutils.util import strtobool
from loguru import logger as loguru_logger

from lightning import pytorch as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
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
from src.lightning.linear_probe import LinearProbeModule
from src.data.matterport_segmentation_dataset import (
    MatterportSegmentationDataset,
    N_OBJECTS_PER_SCENE,
)

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
        default="locus_logs/test",
    )
    parser.add_argument("--exp_name", type=str, default="default_exp_name")
    parser.add_argument("--task", type=str, default="patch_retrieval")
    parser.add_argument("--device_list", type=int, nargs="+", default=None)
    parser.add_argument(
        "--batch_size", type=int, default=32, help="total batch size (across all GPUs)"
    )
    parser.add_argument("--num_workers", type=int, default=16)
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
    parser.add_argument("--linear_probe_max_epochs", type=int, default=1000)

    parser = add_pl_argparse_args(Trainer, parser)
    parser = add_pl_argparse_args(PLModule, parser)
    parser = add_pl_argparse_args(TensorBoardLogger, parser)

    args = parser.parse_args()

    return args


def precalculate_features(args, feature_extractor, dataset):
    # calculate the features, return a TensorDataset with the features and ground truth segmentations as labels
    # Don't downsample the gt_segmentations to the feature resolution
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    feature_extractor.eval()

    features = []
    gt_segmentations = []

    for batch in tqdm(dataloader, "Pre-calculating Segmentation Features"):
        gt_segmentations.append(batch["segmentation"])
        with torch.no_grad():
            features.append(feature_extractor(batch["img"].to(args.device_list[0])))
    features = torch.cat(features, dim=0)
    gt_segmentations = torch.cat(gt_segmentations, dim=0)
    feature_dataset = torch.utils.data.TensorDataset(features, gt_segmentations)

    return feature_dataset


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

    profiler = build_profiler(args)

    if args.task in ["patch_retrieval"]:
        # lightning data
        data_module = MultiSceneDataModule(args, config)
        loguru_logger.info(f"DataModule initialized!")

        # lightning module
        assert args.ckpt_path is not None, "ckpt_path must be specified for testing!"
        model = PLModule(
            config,
            ckpt_path=args.ckpt_path,
            devices=args.device_list,
            profiler=profiler,
            test_task=args.task,
        )
        loguru_logger.info(f"LightningModule initialized!")

        # TensorBoard Logger
        logger = TensorBoardLogger(
            save_dir=os.path.join(args.save_dir, args.exp_name),
            name=args.task,
            default_hp_metric=False,
        )

        # Callbacks
        model_summary = ModelSummary(max_depth=10)
        tqdm_progress_bar = TQDMProgressBar(refresh_rate=args.tqdm_refresh_rate)
        callbacks = [
            model_summary,
            tqdm_progress_bar,
        ]
        if args.profiler_name is not None:
            device_stats_monitor = DeviceStatsMonitor(cpu_stats=True)
            callbacks.append(device_stats_monitor)

        # Lightning Trainer
        trainer = pl_from_argparse_args(
            Trainer,
            args,
            callbacks=callbacks,
            logger=logger,
            reload_dataloaders_every_n_epochs=False,  # avoid repeated samples!
            profiler=profiler,
            detect_anomaly=args.detect_anomaly,
            devices=1,  # only use one GPU for the trainer
        )
        loguru_logger.info(f"Trainer initialized!")
        loguru_logger.info(f"Start testing!")

        trainer.test(model, datamodule=data_module)

    elif args.task == "segmentation":
        feature_extractor = PLModule(
            config,
            ckpt_path=args.ckpt_path,
            devices=args.device_list,
            profiler=profiler,
            test_task=args.task,
        )

        with open(config.DATASET.TEST_SCENE_LIST, "r") as f:
            scene_names = [name.split()[0] for name in f.readlines()]

        data_root = config.DATASET.TEST_DATA_ROOT

        overall_results = {
            "mAP": [],
            "mIoU": [],
            "Jaccard": [],
            "Accuracy": [],
            "object_mAP": [],
            "object_mIoU": [],
            "object_Jaccard": [],
            "object_Accuracy": [],
            "stuff_mAP": [],
            "stuff_mIoU": [],
            "stuff_Jaccard": [],
            "stuff_Accuracy": [],
            "n_images": [],
        }
        output_file = os.path.join(
            args.save_dir, args.exp_name, "segmentation_results.txt"
        )
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        # write a header to the output_file
        with open(output_file, "w+") as f:
            f.write(
                "|\tScene Name\t|\tmAP\t\t|\tmIoU\t\t|\tJaccard\t\t|\tAccuracy\t|\tobject_mAP\t|\tobject_mIoU\t|\tobject_Jaccard\t|\tobject_Accuracy\t|\tstuff_mAP\t|\tstuff_mIoU\t|\tstuff_Jaccard\t|\tstuff_Accuracy\t|\tn_images\t|\n"
            )
            f.write(
                "_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________\n"
            )

        for scene_name in scene_names:
            loguru_logger.info('Testing scene "{}"'.format(scene_name))

            segmentation_dataset = MatterportSegmentationDataset(
                data_root,
                scene_name,
                horizontal_only=config.DATASET.MATTERPORT_HORIZONTAL_IMGS_ONLY,
                normalize=config.DATASET.MATTERPORT_NORMALIZE,
                img_height=config.DATASET.IMG_HEIGHT,
                img_width=config.DATASET.IMG_WIDTH,
            )
            feature_dataset = precalculate_features(
                args, feature_extractor, segmentation_dataset
            )

            linear_probe_train_loader = torch.utils.data.DataLoader(
                feature_dataset, batch_size=args.batch_size, shuffle=True
            )
            linear_probe = LinearProbeModule(config, scene_name)

            logger = TensorBoardLogger(
                save_dir=os.path.join(args.save_dir, args.exp_name),
                name=f"linear_probe_training_scene{scene_name}",
                default_hp_metric=False,
            )

            trainer = pl.Trainer(
                max_epochs=args.linear_probe_max_epochs,
                logger=logger,
                enable_progress_bar=True,
                devices=1,
            )
            loguru_logger.info(
                'Start training linear probe for "{}"'.format(scene_name)
            )
            trainer.fit(linear_probe, linear_probe_train_loader)
            loguru_logger.info(f"Finished training linear probe for {scene_name}!")

            linear_probe_test_loader = torch.utils.data.DataLoader(
                feature_dataset, batch_size=args.batch_size, shuffle=False
            )
            loguru_logger.info(
                f"Start calculating segmentation performance on scene {scene_name}!"
            )
            trainer.test(linear_probe, linear_probe_test_loader)

            scene_results = linear_probe.get_results()
            scene_results["n_images"] = len(feature_dataset)

            with open(output_file, "a") as f:
                f.write(
                    f"|\t{scene_name}\t|\t{scene_results['mAP']:.5f}\t\t|\t{scene_results['mIoU']:.5f}\t\t|\t{scene_results['Jaccard']:.5f}\t\t|\t{scene_results['Accuracy']:.5f}\t\t|\t{scene_results['object_mAP']:.5f}\t\t|\t{scene_results['object_mIoU']:.5f}\t\t|\t{scene_results['object_Jaccard']:.5f}\t\t|\t{scene_results['object_Accuracy']:.5f}\t\t|\t{scene_results['stuff_mAP']:.5f}\t\t|\t{scene_results['stuff_mIoU']:.5f}\t\t|\t{scene_results['stuff_Jaccard']:.5f}\t\t|\t{scene_results['stuff_Accuracy']:.5f}\t\t|\t{scene_results['n_images']}\t\t|\n"
                )

            for key in overall_results.keys():
                if key == "n_images":
                    overall_results[key].append(len(feature_dataset))
                    continue
                overall_results[key].append(scene_results[key])
            loguru_logger.info(
                'Finished segmentation testing on scene "{}"'.format(scene_name)
            )
        mean = lambda x: sum(x) / len(x)
        with open(output_file, "a") as f:
            f.write(
                "_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________\n"
            )
            f.write(
                f"|\tAverage\t\t|\t{mean(overall_results['mAP']):.5f}\t\t|\t{mean(overall_results['mIoU']):.5f}\t\t|\t{mean(overall_results['Jaccard']):.5f}\t\t|\t{mean(overall_results['Accuracy']):.5f}\t\t|\t{mean(overall_results['object_mAP']):.5f}\t\t|\t{mean(overall_results['object_mIoU']):.5f}\t\t|\t{mean(overall_results['object_Jaccard']):.5f}\t\t|\t{mean(overall_results['object_Accuracy']):.5f}\t\t|\t{mean(overall_results['stuff_mAP']):.5f}\t\t|\t{mean(overall_results['stuff_mIoU']):.5f}\t\t|\t{mean(overall_results['stuff_Jaccard']):.5f}\t\t|\t{mean(overall_results['stuff_Accuracy']):.5f}\t\t|\t{sum(overall_results['n_images'])}\t\t|\n"
            )


if __name__ == "__main__":
    main()
