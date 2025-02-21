from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import json
import os
import argparse
import random
import numpy as np

from dataset_loader import TrainingDataset, ValidationDataset
from solver import Solver
import loss_functions
from models import u_net, deeplabv3, fcb_former, fcn, fpn_net, hr_net, link_net
from conditionally_adaptive_loss_pytorch import AdaptiveDistributionLoss
from region_specific_adaptive_loss_pytorch import RegionSpecificAdaptiveLoss

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, barrier


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def load_train_data():
    # Initialize datasets
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    train_paths = []
    train_path = "/ediss_data/ediss1/datasets/train"
    train_paths.extend(
        os.path.join(train_path, path) for path in os.listdir(train_path)
    )

    train = TrainingDataset(train_paths, transform=transform)
    return train


def load_test_data():
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    test_paths = []
    test_path = "/ediss_data/ediss1/datasets/test"
    test_paths.extend(os.path.join(test_path, path) for path in os.listdir(test_path))

    test = ValidationDataset(test_paths, transform=transform)
    return test


def load_model(model_name):
    if model_name == "u-net":
        model = u_net.UNet(in_channels=1, out_channels=1)
    elif model_name == "deeplabv3":
        model = deeplabv3.Deeplabv3Plus()
    elif model_name == "fcb-former":
        model = fcb_former.FCBFormer(num_classes=1, in_channels=1)
    elif model_name == "fcn":
        model = fcn.FCN()
    elif model_name == "fpn-net":
        model = fpn_net.FPN(fpn_net.Bottleneck, [2, 2, 2, 2])
    elif model_name == "hr-net":
        model = hr_net.HRNet()
    elif model_name == "link-net":
        model = link_net.LinkNet(n_classes=1)
    else:
        raise ValueError("Invalid model name")
    return model


def load_loss_fn(loss_name):
    if loss_name == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif loss_name == "dice":
        loss_fn = loss_functions.DiceLoss()
    elif loss_name == "focal":
        loss_fn = loss_functions.FocalLoss()
    elif loss_name == "tversky":
        loss_fn = loss_functions.TverskyLoss()
    elif loss_name == "iou":
        loss_fn = loss_functions.IoULoss()
    elif loss_name == "combo":
        loss_fn = loss_functions.ComboLoss()
    elif loss_name == "adaptive":
        loss_fn = AdaptiveDistributionLoss()
    elif loss_name == "region":
        loss_fn = RegionSpecificAdaptiveLoss()
    else:
        raise ValueError("Invalid loss function name")
    return loss_fn


def execute(
    rank: int,
    world_size: int,
    model_name: str,
    loss_name: str,
    checkpoint: str,
    ratio: float,
):
    try:
        ddp_setup(rank, world_size)
        print(f"Running DDP process {rank}/{world_size}")

        # Set the seed BEFORE any randomness happens
        seed_value = 12345
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Hyperparameters
        num_epochs = 20
        learning_rate = 0.00001
        batch_size = 32
        min_delta = 1e-8  # minimum change in training loss
        patience = 10  # number of epochs to wait before early stopping

        model = load_model(model_name)

        if checkpoint:
            model.load_state_dict(torch.load(checkpoint))
            print(f"Model loaded from {checkpoint}")

        # Convert BatchNorm layers to SyncBatchNorm for multi-GPU training
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Loss function
        loss_fn = load_loss_fn(loss_name)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Data loader
        train = load_train_data()

        if ratio:
            train.set_segmentation_ratio(ratio)
            _, _, metadata = train[0]
            print(f"Metadata of the first example for sanity check: {metadata}")

        train_sampler = DistributedSampler(
            train, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            train,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True,
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
        )

        # Train the model
        time_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Starting training at {time_start}")

        solver = Solver(model, loss_fn, optimizer, rank)
        solver.train(
            train_loader,
            num_epochs,
            min_delta=min_delta,
            patience=patience,
            checkpoint_dir="/ediss_data/ediss1/checkpoints",
        )

        # Wait for all processes to finish training
        barrier()

        if rank == 0:
            time_end = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            time_taken = datetime.strptime(
                time_end, "%Y-%m-%d_%H-%M-%S"
            ) - datetime.strptime(time_start, "%Y-%m-%d_%H-%M-%S")
            print(f"Training finished at {time_end}")
            print(f"Time taken: {time_taken}")

            # Save final model
            path = "/ediss_data/ediss1/best-models/"
            os.makedirs(path, exist_ok=True)
            path += f"{solver.model_name}{solver.loss_fn.__class__.__name__}"
            timestamp = time_end
            path += timestamp
            path += ".pth"

            torch.save(
                solver.model.module.state_dict(), path
            )  # Because of DataParallel
            print(f"Model saved at {path}")

        # Evaluate the model
        test = load_test_data()
        test_sampler = DistributedSampler(
            test, num_replicas=world_size, rank=rank, shuffle=True
        )
        test_loader = DataLoader(
            test,
            batch_size=batch_size,
            sampler=test_sampler,
            pin_memory=True,
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
        )
        solver.validate(test_loader)

        # Wait for all processes to finish validation
        barrier()

        if rank == 0:
            print(f"Validation Accuracy: {solver.history['val_accuracy']:.4f}")
            print(f"Validation DSC: {solver.history['val_dsc']:.4f}")

            # Save the results
            time_taken_str = str(time_taken)

            results = {
                "epochs": num_epochs,
                "actual_epochs": len(solver.history["train_loss"]),
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "min_delta": min_delta,
                "patience": patience,
                "time_taken": time_taken_str,
                "loss_history": solver.history["train_loss"],
                "dsc_history": solver.history["train_dsc"],
                "val_accuracy": solver.history["val_accuracy"],
                "val_dice_score": solver.history["val_dsc"],
                "val_specificity": solver.history["val_specificity"],
                "val_sensitivity": solver.history["val_sensitivity"],
                "val_precision": solver.history["val_precision"],
                "val_mae": solver.history["val_mae"],
                "segmentation_ratio": train.ratio,
                "mode": train.mode,
            }

            path = "/ediss_data/ediss1/results/"
            os.makedirs(path, exist_ok=True)

            path += f"{solver.model_name}{loss_fn.__class__.__name__}"
            path += timestamp
            path += ".json"

            with open(path, "w") as f:
                json.dump(results, f)

            print("Results saved successfully")

    except Exception as e:
        print(e)
    finally:
        destroy_process_group()


def main(args):
    # Multi-GPU training
    world_size = torch.cuda.device_count()
    mp.spawn(
        execute,
        args=(world_size, args.model, args.loss, args.checkpoint, args.ratio),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the dataset")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to train (u-net, deeplabv3, fcb-former, fcn, fpn-net, hr-net, link-net)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        help="Name of the loss function to use (bce, dice, focal, tversky, iou, combo, adaptive, region)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to the checkpoint file to load the model from",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        required=False,
        help="Segmentation ratio for the dataset",
    )

    parser.print_help()
    args = parser.parse_args()
    main(args)
