import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import scipy.ndimage


# DISTRIBUTION-BASED LOSS FUNCTIONS


##################################### Loss Function No: 1 ####################################
# Binary Cross-Entropy Loss
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets)


##################################### Loss Function No: 2 ####################################
# Weighted Binary Cross-Entropy Loss
class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


##################################### Loss Function No: 3 ####################################
# Top-K Loss
class TopKLoss(nn.Module):
    def __init__(self, k):
        super(TopKLoss, self).__init__()
        self.k = k

    def forward(self, logits, targets):
        # Compute per-sample cross-entropy loss
        losses = F.cross_entropy(logits, targets, reduction="none")

        # Ensure k does not exceed batch size
        k = min(self.k, losses.size(0))

        # Select top-k loss values
        topk_losses, _ = torch.topk(losses, k)

        # Return the mean of top-k losses
        return topk_losses.mean()


##################################### Loss Function No: 4 ####################################
# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-6, max=1 - 1e-6)  # Prevent log(0)

        # Compute BCE loss for both positive and negative classes
        bce_loss = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))

        # Compute the focal weight
        focal_weight = (targets * (1 - probs) + (1 - targets) * probs) ** self.gamma

        # Compute focal loss
        loss = focal_weight * bce_loss

        # Apply alpha balancing
        loss = self.alpha * targets * loss + (1 - self.alpha) * (1 - targets) * loss

        return loss.mean()


##################################### Loss Function No: 5 ####################################
# Distance Map Penalizing Loss
class DistanceMapPenalizingLoss(nn.Module):
    def __init__(self, lambda_=1):
        super(DistanceMapPenalizingLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # Compute the distance transform of the ground truth
        targets_np = targets.cpu().numpy()
        distance_maps = np.zeros_like(targets_np, dtype=np.float32)
        for b in range(targets_np.shape[0]):
            distance_maps[b] = scipy.ndimage.distance_transform_edt(1 - targets_np[b])
        distance_maps = torch.tensor(
            distance_maps, device=targets.device, dtype=torch.float32
        )

        return (probs * distance_maps).mean() + self.lambda_ * (1 - probs).mean()


# REGION-BASED LOSS FUNCTIONS


##################################### Loss Function No: 6 ####################################
# DICE
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets, smooth=1e-5):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = 2.0 * intersection / (union + smooth)
        return 1 - dice.mean()


##################################### Loss Function No: 7 ####################################
# IoU (Jaccard) Loss
class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, logits, targets, smooth=1e-5):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        total = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        union = total - intersection
        iou = intersection / (union + smooth)
        return 1 - iou.mean()


##################################### Loss Function No: 8 ####################################
# Lovász Loss (requires specific implementation for logits)
class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()

    def forward(self, logits, targets):
        return self.lovasz_hinge(logits, targets)

    @staticmethod
    def lovasz_hinge(logits, labels):
        signs = 2.0 * labels - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        grad = LovaszLoss.lovasz_grad(labels[perm])
        return torch.dot(F.relu(errors_sorted), grad)

    @staticmethod
    def lovasz_grad(gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        grads = torch.zeros(p, device=gt_sorted.device)
        for i in range(p):
            grads[i] = gts - gt_sorted[: i + 1].sum()
        grads[1:] -= grads[:-1]
        return grads


##################################### Loss Function No: 9 ####################################
# Tversky Loss
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets, smooth=1e-5):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Ensure target shape matches logits shape
        if targets.shape[1] != probs.shape[1]:
            targets = targets.expand_as(probs)

        # Compute TP, FP, FN
        tp = (probs * targets).sum(dim=(1, 2, 3))  # Fix summation dimensions
        fp = ((1 - targets) * probs).sum(dim=(1, 2, 3))
        fn = (targets * (1 - probs)).sum(dim=(1, 2, 3))

        # Compute Tversky index
        tversky = tp / (tp + self.alpha * fp + self.beta * fn + smooth)

        # Return Tversky loss
        return 1 - tversky.mean()


##################################### Loss Function No: 10 ####################################
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2, smooth=1e-5):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten predictions and targets for per-pixel calculation
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Compute true positives, false positives, and false negatives
        tp = (probs * targets).sum(dim=1)
        fp = ((1 - targets) * probs).sum(dim=1)
        fn = (targets * (1 - probs)).sum(dim=1)

        # Compute Tversky Index
        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        # Compute Focal Tversky Loss
        focal_tversky_loss = (1 - tversky_index) ** self.gamma
        return focal_tversky_loss.mean()


##################################### Loss Function No: 11 ####################################
# Robust T Loss - TO CHECK
class RobustTLoss(nn.Module):
    def __init__(self, alpha=0.25, beta=0.25, gamma=2):
        super(RobustTLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, targets, smooth=1e-5):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum(dim=(1, 2, 3))
        fp = ((1 - targets) * probs).sum(dim=(1, 2, 3))
        fn = (targets * (1 - probs)).sum(dim=(1, 2, 3))
        tversky = tp / (tp + self.alpha * fp + self.beta * fn + smooth)
        return 1 - tversky.mean() ** self.gamma


##################################### Loss Function No: 12 ####################################
# Sensitivity Specificity loss
class SensitivitySpecificityLoss(nn.Module):

    def __init__(self, alpha=0.25, beta=0.25, smooth=1e-5):
        super(SensitivitySpecificityLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to logits to compute probabilities
        probs = torch.sigmoid(logits)

        # Calculate True Positives, False Negatives, True Negatives, False Positives
        tp = (probs * targets).sum(dim=(1, 2))
        fn = ((1 - probs) * targets).sum(dim=(1, 2))
        tn = ((1 - probs) * (1 - targets)).sum(dim=(1, 2))
        fp = (probs * (1 - targets)).sum(dim=(1, 2))

        # Calculate Sensitivity and Specificity
        sensitivity = tp / (tp + fn + self.smooth)
        specificity = tn / (tn + fp + self.smooth)

        # Loss components: we minimize the deviation from ideal sensitivity and specificity
        sensitivity_loss = 1 - sensitivity
        specificity_loss = 1 - specificity

        # Combine losses using weighted sum
        loss = self.alpha * sensitivity_loss + self.beta * specificity_loss
        return loss.mean()


##################################### Loss Function No: 13 ####################################
# Asymmetric Similarity Loss
class AsymmetricSimilarityLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(AsymmetricSimilarityLoss, self).__init__()
        self.alpha = alpha

    def forward(self, logits, targets, smooth=1e-5):
        # Apply sigmoid to logits to convert them into probabilities
        probs = torch.sigmoid(logits)

        # Flatten probabilities and targets for per-pixel computation over 2D images
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Calculate intersection, false negatives, and false positives
        intersection = (probs * targets).sum(dim=1)
        false_neg = (targets * (1 - probs)).sum(dim=1)
        false_pos = ((1 - targets) * probs).sum(dim=1)

        # Calculate the asymmetric similarity loss
        loss = 1 - (intersection + smooth) / (
            intersection
            + self.alpha * false_neg
            + (1 - self.alpha) * false_pos
            + smooth
        )

        # Return mean loss over the batch
        return loss.mean()


##################################### Loss Function No: 14 ####################################
# Generalized Dice Loss
class GeneralizedDiceLoss(nn.Module):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, logits, targets, smooth=1e-5):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        weights = 1 / (targets.sum(dim=(1, 2, 3)) ** 2 + smooth)
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        generalized_dice = 2 * (weights * intersection).sum() / (weights * union).sum()
        return 1 - generalized_dice


##################################### Loss Function No: 15 ####################################
# Generalized Wasserstein Dice Loss
class GeneralizedWassersteinDiceLoss(nn.Module):
    def __init__(self):
        super(GeneralizedWassersteinDiceLoss, self).__init__()

    def forward(self, logits, targets, smooth=1e-5):
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)  # Apply sigmoid to get probabilistic predictions

        # Flatten predictions and targets across spatial dimensions for computation
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Compute intersection and sums
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2 * intersection + smooth) / (union + smooth)

        # Loss is computed as the complement of similarity
        loss = 1 - dice_score.mean()

        return loss


##################################### Loss Function No: 16 ####################################
# Penalty Loss
class PenaltyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(PenaltyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        # Apply sigmoid to logits to convert to probabilities
        probs = torch.sigmoid(logits)
        # Flatten predictions and targets over spatial dimensions
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Calculate Intersection and Union
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        # Calculate Penalty Loss
        penalty_loss = 1 - (intersection + self.beta) / (union + self.beta)

        # Return mean over the entire batch
        return penalty_loss.mean()


# BOUNDARY-BASED LOSS FUNCTIONS


##################################### Loss Function No: 17 ####################################
# Boundary Loss
class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, logits, targets):
        # Apply sigmoid to logits
        probs = torch.sigmoid(logits)

        # Compute distance maps for the ground truth mask
        with torch.no_grad():
            # Convert ground truth to numpy
            targets_np = targets.cpu().numpy()
            distance_maps = np.zeros_like(targets_np, dtype=np.float32)
            for b in range(targets_np.shape[0]):
                distance_maps[b] = scipy.ndimage.distance_transform_edt(
                    1 - targets_np[b]
                )

            # Convert distance maps back to PyTorch tensor
            distance_maps = torch.tensor(
                distance_maps, device=targets.device, dtype=torch.float32
            )

        # Compute the boundary loss by weighting probs with these distance maps
        boundary_loss = (probs * distance_maps).mean()
        return boundary_loss


##################################### Loss Function No: 18 ####################################
# Bounadry-Aware Loss
class BoundaryAwareLoss(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super(BoundaryAwareLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets, distance_maps):
        probs = torch.sigmoid(logits)

        # Compute the boundary loss term (weighted by distance maps)
        boundary_loss = (probs * distance_maps).mean()

        # Compute the complement misclassification term
        misclassification_loss = self.alpha * (1 - probs).mean()

        # Combine both losses
        total_loss = boundary_loss + misclassification_loss

        return total_loss


##################################### Loss Function No: 19 ####################################
# Inverse Form Loss
class InverseFormLoss(nn.Module):
    def __init__(self, alpha=1):
        super(InverseFormLoss, self).__init__()
        self.alpha = alpha

    def forward(self, logits, targets, distance_maps):
        probs = torch.sigmoid(logits)

        # Compute the inverse form loss term
        inverse_loss = (probs * distance_maps).mean()

        # Add a complement penalty term scaled by alpha
        complement_loss = self.alpha * (1 - probs).mean()

        # Combine the two terms
        total_loss = inverse_loss + complement_loss
        return total_loss


##################################### Loss Function No: 20 ####################################
# Hausdorff Distance Loss
class HausdorffDistanceLoss(nn.Module):
    def __init__(self, percentile=95):
        super(HausdorffDistanceLoss, self).__init__()
        self.percentile = percentile

    def compute_hausdorff_distance(self, preds, targets):
        # Find coordinates of nonzero regions in prediction and targets
        preds_coords = torch.nonzero(preds > 0.5, as_tuple=False).float()
        targets_coords = torch.nonzero(targets > 0.5, as_tuple=False).float()

        if preds_coords.numel() == 0 or targets_coords.numel() == 0:
            # Avoid computation if there are no predicted points or targets
            return torch.tensor(float("inf"), device=preds.device)

        # Compute pairwise distances between predicted points and target points
        distances_pred_to_target = torch.cdist(
            preds_coords.unsqueeze(0), targets_coords.unsqueeze(0)
        )
        distances_target_to_pred = torch.cdist(
            targets_coords.unsqueeze(0), preds_coords.unsqueeze(0)
        )

        # Directed Hausdorff distances
        h_dist_pred_to_target = torch.max(distances_pred_to_target)
        h_dist_target_to_pred = torch.max(distances_target_to_pred)

        # Symmetric Hausdorff distance
        hausdorff_distance = max(h_dist_pred_to_target, h_dist_target_to_pred)
        return hausdorff_distance

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # Ensure binary masks are cast to float for comparison
        targets = targets.float()

        # Compute directed Hausdorff distances over each batch element
        distances = torch.stack(
            [
                self.compute_hausdorff_distance(probs[b], targets[b])
                for b in range(logits.shape[0])
            ]
        )

        # Return the average over the batch
        return distances.mean()


# COMPOUND LOSS FUNCTIONS


##################################### Loss Function No: 21 ####################################
# Combo Loss (Binary Cross-Entropy - Dice Loss)
class ComboLoss(nn.Module):
    def __init__(self):
        """
        Combination of Binary Cross-Entropy (BCE) Loss and Dice Loss.
        """
        super(ComboLoss, self).__init__()
        self.bce_loss = BinaryCrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        """
        Compute Combo Loss (BCE + Dice).

        :param logits: Predicted logits (before sigmoid activation).
        :param targets: Ground truth labels (same shape as logits).
        :return: Combo loss value.
        """
        bce = self.bce_loss(logits, targets)  # Compute BCE Loss
        dice = self.dice_loss(logits, targets)  # Compute Dice Loss
        return bce + dice  # Combine both losses


##################################### Loss Function No: 22 ####################################
# Dice-Focal Loss
class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, smooth=1e-5):
        # Dice Loss
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

        # Focal Loss
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        return dice_loss + focal_loss


##################################### Loss Function No: 23 ####################################
# Dice-TopK Loss
class DiceTopKLoss(nn.Module):
    def __init__(self, k):
        super(DiceTopKLoss, self).__init__()
        self.k = k

    def forward(self, logits, targets, smooth=1e-5):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

        # Top-K Cross-Entropy Loss
        losses = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ).view(-1)
        topk_losses, _ = torch.topk(losses, self.k)
        topk_loss = topk_losses.mean()

        return dice_loss + topk_loss


##################################### Loss Function No: 24 ####################################
# Unified Focal Loss - TO CHECK
class UnifiedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, beta=1):
        super(UnifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, logits, targets, smooth=1e-5):
        # Focal Loss
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        # Boundary Loss
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        boundary_loss = 1 - (intersection + self.beta) / (union + self.beta)

        return focal_loss + boundary_loss


##################################### Loss Function No: 25 ####################################
# Exponential Logarithmic Loss
class ExponentialLogarithmicLoss(nn.Module):
    def __init__(self, alpha=1, lambda_=1, gamma=1):
        super(ExponentialLogarithmicLoss, self).__init__()
        self.alpha = alpha
        self.lambda_ = lambda_
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # Logarithmic loss component
        log_loss = -self.alpha * targets * torch.log(probs) - (1 - targets) * torch.log(
            1 - probs
        )

        # Exponential penalty component
        exp_penalty = self.lambda_ * torch.exp(self.gamma * targets * (1 - probs))

        # Final loss
        loss = log_loss + exp_penalty
        return loss.mean()
