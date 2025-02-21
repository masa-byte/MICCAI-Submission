import torch
from tqdm import tqdm
import os
import json
from torch.nn.parallel import DistributedDataParallel as DDP


# Dice similarity coefficient
def dice_similarity_coefficient(preds, targets):
    """
    Calculate the Dice Similarity Coefficient for binary segmentation.

    Args:
        preds (torch.Tensor): Predictions from the model (shape: [B, 1, H, W]).
        targets (torch.Tensor): Ground truth labels (shape: [B, 1, H, W]).
        threshold (float): Threshold for converting probabilities to binary predictions.


    """
    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item()

    if union == 0:  # Avoid division by zero
        return 1.0

    dice = (2.0 * intersection) / union
    return dice


# Specificity
def specificity(preds, targets):
    """
    Calculate the Specificity metric for binary segmentation.

    Args:
        preds (torch.Tensor): Predictions from the model (shape: [B, 1, H, W]).
        targets (torch.Tensor): Ground truth labels (shape: [B, 1, H, W]).

    """
    # True Negative (TN): (1 - preds) * (1 - targets)
    tn = ((1 - preds) * (1 - targets)).sum().item()

    # False Positive (FP): preds * (1 - targets)
    fp = (preds * (1 - targets)).sum().item()

    if (tn + fp) == 0:  # Avoid division by zero
        return 1.0

    specificity_score = tn / (tn + fp)
    return specificity_score


# Sensitivity
def sensitivity(preds, targets):
    """
    Calculate the Sensitivity (Recall) metric for binary segmentation.

    Args:
        preds (torch.Tensor): Predictions from the model (shape: [B, 1, H, W]).
        targets (torch.Tensor): Ground truth labels (shape: [B, 1, H, W]).

    """
    # True Positive (TP): preds * targets
    tp = (preds * targets).sum().item()

    # False Negative (FN): (1 - preds) * targets
    fn = ((1 - preds) * targets).sum().item()

    if (tp + fn) == 0:  # Avoid division by zero
        return 1.0

    sensitivity_score = tp / (tp + fn)
    return sensitivity_score


# Precision
def precision(preds, targets):
    """
    Calculate the Precision metric for binary segmentation.

    Args:
        preds (torch.Tensor): Predictions from the model (shape: [B, 1, H, W]).
        targets (torch.Tensor): Ground truth labels (shape: [B, 1, H, W]).

    Returns:
        float: Precision value between 0 and 1.
    """
    # True Positive (TP): preds * targets
    tp = (preds * targets).sum().item()

    # False Positive (FP): preds * (1 - targets)
    fp = (preds * (1 - targets)).sum().item()

    # Precision calculation
    if (tp + fp) == 0:  # Avoid division by zero
        return 1.0  # Assume perfect precision if no positive predictions

    precision_score = tp / (tp + fp)
    return precision_score


# MAE
def mean_absolute_error(preds, targets):
    """
    Calculate the Mean Absolute Error (MAE) for binary segmentation.

    Args:
        preds (torch.Tensor): Predictions from the model (shape: [B, 1, H, W]).
        targets (torch.Tensor): Ground truth labels (shape: [B, 1, H, W]).
    """

    mae = torch.abs(preds - targets).mean().item()  # LOWER IS BETTER!
    return mae


class Solver:
    def __init__(self, model, loss_fn, optimizer, rank):
        self.model_name = model.__class__.__name__
        self.model = model.to(rank)
        self.model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        self.loss_fn = loss_fn.to(rank)
        self.optimizer = optimizer
        self.device = rank
        self.history = {
            "train_loss": [],
            "train_dsc": [],
            "val_accuracy": None,
            "val_dsc": None,
            "val_specificity": None,
            "val_sensitivity": None,
            "val_precision": None,
            "val_mae": None,
        }

    def train(
        self,
        train_loader,
        num_epochs,
        min_delta,
        patience,
        checkpoint_interval=5,
        checkpoint_dir="checkpoints",
    ):
        """
        Train the given model using the specified loss function and optimizer, track accuracy, and save checkpoints every `checkpoint_interval` epochs.

        Args:
            train_loader (DataLoader): The data loader for training data.
            num_epochs (int): Number of training epochs.
            min_delta (float): Stop training if the training loss decreases by less than `min_delta`.
            patience (int): Minimum number of epochs to train before early stopping.
            checkpoint_interval (int): Save model checkpoints every `checkpoint_interval` epochs.
            checkpoint_dir (str): Directory to save checkpoints.

        In self.history["train_loss"] - store the training loss for each epoch.
        In self.history["train_dsc"] - store the training DSC for each epoch.
        """

        self.history["train_loss"] = []
        self.history["train_dsc"] = []

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            dsc_train = 0.0

            train_loader.sampler.set_epoch(epoch)  # Shuffle data for DistributedSampler

            for batch in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"
            ):
                images, masks, _ = batch  # Extract images and masks
                images, masks = images.to(self.device), masks.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                binary = (outputs > 0).float()  # For binary classification

                if self.loss_fn.__class__.__name__ == "AdaptiveDistributionLoss":
                    loss = self.loss_fn.calculate_areas(masks)

                loss = self.loss_fn(outputs, masks)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                dsc_train += dice_similarity_coefficient(binary, masks)

            train_loss = torch.tensor(
                running_loss / len(train_loader), device=self.device
            )
            train_dsc = torch.tensor(dsc_train / len(train_loader), device=self.device)

            # Synchronize across all processes
            torch.distributed.all_reduce(train_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(train_dsc, op=torch.distributed.ReduceOp.SUM)

            # Compute the global average
            world_size = torch.distributed.get_world_size()
            train_loss /= world_size
            train_dsc /= world_size

            self.history["train_loss"].append(train_loss.item())
            self.history["train_dsc"].append(train_dsc.item())

            # Save checkpoint
            if self.device == 0 and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"{self.model_name}{self.loss_fn.__class__.__name__}_epoch_{epoch + 1}.pth",
                )
                torch.save(self.model.module.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

                # Save self.history to a JSON file
                history_path = os.path.join(
                    checkpoint_dir,
                    f"{self.model_name}{self.loss_fn.__class__.__name__}_history.json",
                )
                with open(history_path, "w") as f:
                    json.dump(self.history, f)

            # # Early stopping
            # if len(self.history["train_loss"]) > patience:
            #     if (
            #         self.history["train_loss"][-2] - self.history["train_loss"][-1]
            #         < min_delta
            #     ):
            #         print(
            #             f"Training stopped early at epoch {epoch + 1} due to small change in loss."
            #         )
            #         break

        if self.device == 0:
            # Logging
            print(f"Epoch {epoch + 1}/{num_epochs}: " f"Train Loss: {train_loss:.4f}, ")

    def validate(self, val_loader):
        """
        Validate the model using the specified validation data loader.

        Args:
            val_loader (DataLoader): The data loader for validation data.

        In self.history["val_accuracy"] - store the validation accuracy.
        In self.history["val_dsc"] - store the validation DSC.
        In self.history["val_specificity"] - store the validation specificity.
        In self.history["val_sensitivity"] - store the validation sensitivity.
        In self.history["val_presicion"] - store the validation presicion.
        In self.history["val_mae"] - store the validation mae.
        """

        self.history["val_accuracy"] = None
        self.history["val_dsc"] = None
        self.history["val_specificity"] = None
        self.history["val_sensitivity"] = None
        self.history["val_presicion"] = None
        self.history["val_mae"] = None

        self.model.eval()
        correct_val = 0
        total_val = 0
        dsc_val = 0.0
        specificity_val = 0.0
        sensitivity_val = 0.0
        precision_val = 0.0
        mae_val = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images, masks, _ = batch  # Extract image and mask
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model.module(images)
                binary = (outputs > 0).float()  # For binary classification

                # Calculate validation accuracy
                correct_val += (binary == masks).sum().item()
                total_val += masks.numel()

                # Calculate Dice Similarity Coefficient
                dsc_val += dice_similarity_coefficient(binary, masks)

                # Calculate Specificity
                specificity_val += specificity(binary, masks)

                # Calculate Sensitivity
                sensitivity_val += sensitivity(binary, masks)

                # Calculate Precision
                precision_val += precision(binary, masks)

                # Calculate MAE
                mae_val += mean_absolute_error(binary, masks)

            val_accuracy = torch.tensor(correct_val / total_val, device=self.device)
            val_dsc = torch.tensor(dsc_val / len(val_loader), device=self.device)
            val_specificity = torch.tensor(
                specificity_val / len(val_loader), device=self.device
            )
            val_sensitivity = torch.tensor(
                sensitivity_val / len(val_loader), device=self.device
            )
            val_precision = torch.tensor(
                precision_val / len(val_loader), device=self.device
            )
            val_mae = torch.tensor(mae_val / len(val_loader), device=self.device)

            # Synchronize across all processes
            torch.distributed.all_reduce(
                val_accuracy, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(val_dsc, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                val_specificity, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                val_sensitivity, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                val_precision, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(val_mae, op=torch.distributed.ReduceOp.SUM)

            # Compute the global average
            world_size = torch.distributed.get_world_size()
            val_accuracy /= world_size
            val_dsc /= world_size
            val_specificity /= world_size
            val_sensitivity /= world_size
            val_precision /= world_size
            val_mae /= world_size

            self.history["val_accuracy"] = val_accuracy.item()
            self.history["val_dsc"] = val_dsc.item()
            self.history["val_specificity"] = val_specificity.item()
            self.history["val_sensitivity"] = val_sensitivity.item()
            self.history["val_precision"] = val_precision.item()
            self.history["val_mae"] = val_mae.item()

    @staticmethod
    def test_model(model, image, mask, device):
        """
        Test the model using a single image and mask.

        Args:
            model (nn.Module): The trained model.
            image (torch.Tensor): The input image.
            mask (torch.Tensor): The ground truth mask.
            device (torch.device): The device to run the model on.

        Returns:
            torch.Tensor: The predicted mask.
        """
        dsc = 0.0
        spec = 0.0
        sens = 0.0

        image = image.to(device)

        # Assume model is already in evaluation mode
        with torch.no_grad():
            mask_pred = model(image.unsqueeze(0))
            mask_pred = mask_pred.cpu().squeeze(0)
            # Ensure predicted mask is binary
            binary = (mask_pred > 0).float()

            dsc = dice_similarity_coefficient(binary, mask)
            spec = specificity(binary, mask)
            sens = sensitivity(binary, mask)

        return binary, dsc, spec, sens

    @staticmethod
    def analyze_model(model, validation_loader, device):
        scan_type_metrics = {
            "T1": {"dsc": 0, "specificity": 0, "sensitivity": 0, "count": 0},
            "T2": {"dsc": 0, "specificity": 0, "sensitivity": 0, "count": 0},
            "FLAIR": {"dsc": 0, "specificity": 0, "sensitivity": 0, "count": 0},
        }

        image_metrics = {
            "minimal_dsc": (1, None, None, None),
            "maximal_dsc": (0, None, None, None),
            "minimal_specificity": (1, None, None, None),
            "maximal_specificity": (0, None, None, None),
            "minimal_sensitivity": (1, None, None, None),
            "maximal_sensitivity": (0, None, None, None),
        }

        # Assumes model is already in evaluation mode
        # Assumes batch size of 1
        for batch in tqdm(validation_loader, desc="Analyzing model"):
            image, mask, metadata = batch

            image = image.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                mask_pred = model(image)
                # Ensure predicted mask is binary
                binary = (mask_pred > 0).float()

                dsc = dice_similarity_coefficient(binary, mask)
                spec = specificity(binary, mask)
                sens = sensitivity(binary, mask)

                # Update scan type metrics
                scan_type = metadata["scan_type"][0]

                scan_type_metrics[scan_type]["dsc"] += dsc
                scan_type_metrics[scan_type]["specificity"] += spec
                scan_type_metrics[scan_type]["sensitivity"] += sens
                scan_type_metrics[scan_type]["count"] += 1

                # Update image metrics
                if dsc < image_metrics["minimal_dsc"][0] and dsc > 0:
                    image_metrics["minimal_dsc"] = (
                        dsc,
                        image.cpu(),
                        mask.cpu(),
                        metadata,
                    )
                elif dsc > image_metrics["maximal_dsc"][0] and dsc < 1:
                    image_metrics["maximal_dsc"] = (
                        dsc,
                        image.cpu(),
                        mask.cpu(),
                        metadata,
                    )

                if spec < image_metrics["minimal_specificity"][0] and spec > 0:
                    image_metrics["minimal_specificity"] = (
                        spec,
                        image.cpu(),
                        mask.cpu(),
                        metadata,
                    )
                elif spec > image_metrics["maximal_specificity"][0] and spec < 1:
                    image_metrics["maximal_specificity"] = (
                        spec,
                        image.cpu(),
                        mask.cpu(),
                        metadata,
                    )

                if sens < image_metrics["minimal_sensitivity"][0]:
                    image_metrics["minimal_sensitivity"] = (
                        sens,
                        image.cpu(),
                        mask.cpu(),
                        metadata,
                    )
                elif sens > image_metrics["maximal_sensitivity"][0] and sens < 1:
                    image_metrics["maximal_sensitivity"] = (
                        sens,
                        image.cpu(),
                        mask.cpu(),
                        metadata,
                    )

        # Calculate average metrics for each scan type
        for scan_type in scan_type_metrics:
            count = scan_type_metrics[scan_type]["count"]
            if count > 0:
                scan_type_metrics[scan_type]["dsc"] /= count
                scan_type_metrics[scan_type]["specificity"] /= count
                scan_type_metrics[scan_type]["sensitivity"] /= count

        return scan_type_metrics, image_metrics
