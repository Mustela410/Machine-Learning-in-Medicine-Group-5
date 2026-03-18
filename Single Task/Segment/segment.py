import json
import os
import random
import shutil
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

DEFAULT_DATA_ROOT = "/content/drive/MyDrive/BUSI"
DEFAULT_SPLIT_FILE = os.path.join(DEFAULT_DATA_ROOT, "data_split_85_15.json")
DEFAULT_SAVE_ROOT = "/content/drive/MyDrive/BUSI1"
MODEL_PREFIX = "segment1"

CONFIG = {
    "input_size": 256,
    "batch_size": 64,
    "num_epochs": 100,
    "num_workers": 8,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "lr_patience": 4,
    "lr_factor": 0.5,
    "early_stopping_patience": 15,
    "n_folds": 5,
    "encoder": "resnet34",
    "architecture": "unet",
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device() -> torch.device:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fold_model_path(save_root: str, fold: int) -> str:
    return os.path.join(save_root, f"{MODEL_PREFIX}_fold_{fold + 1}.pth")


def main_model_path(save_root: str) -> str:
    return os.path.join(save_root, f"{MODEL_PREFIX}.pth")


def load_data_split(split_file: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    with open(split_file, "r", encoding="utf-8") as f:
        data_split = json.load(f)

    dev_images = data_split["development"]
    test_images = data_split["test"]

    dev_paths = [img["path"] for img in dev_images]
    dev_labels = [img["label"] for img in dev_images]
    test_paths = [img["path"] for img in test_images]
    test_labels = [img["label"] for img in test_images]

    return dev_paths, dev_labels, test_paths, test_labels


def load_all_masks(image_path: str) -> np.ndarray | None:
    base = image_path.replace(".png", "")
    merged = None
    i = 0
    while True:
        suffix = "_mask" if i == 0 else f"_mask_{i}"
        mask_path = base + suffix + ".png"
        if not os.path.exists(mask_path):
            break
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        merged = mask if merged is None else cv2.bitwise_or(merged, mask)
        i += 1
    return merged


class BUSIDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], labels: Sequence[str], transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = load_all_masks(self.image_paths[idx])
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        mask = mask / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0).float()


def build_transforms(config: Dict) -> Tuple[A.Compose, A.Compose]:
    train_transform = A.Compose(
        [
            A.Resize(config["input_size"], config["input_size"]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15, p=0.4),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.15),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(config["input_size"], config["input_size"]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    return train_transform, val_transform


def build_model(config: Dict) -> nn.Module:
    return smp.Unet(
        encoder_name=config["encoder"],
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )


class SegmentationLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.bce(pred, target) + 0.5 * self.dice_loss(pred, target)


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    pred = (torch.sigmoid(pred) > 0.5).float().view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    pred = (torch.sigmoid(pred) > 0.5).float().view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def auc_score(all_probs: List[np.ndarray], all_targets: List[np.ndarray]) -> float:
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    if targets.sum() == 0 or (1 - targets).sum() == 0:
        return float("nan")
    return float(roc_auc_score(targets.astype(int), probs))


def format_auc(value: float) -> str:
    return f"{value:.4f}" if not np.isnan(value) else "n/a"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    pbar = tqdm(loader, desc="train", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_dice = dice_score(outputs, masks)
        total_loss += loss.item()
        total_dice += batch_dice
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_dice:.4f}")

    return total_loss / len(loader), total_dice / len(loader)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, List[np.ndarray], List[np.ndarray]]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        pbar = tqdm(loader, desc="val", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, masks)

            probs = torch.sigmoid(outputs).float().cpu().numpy().ravel()
            targets = masks.float().cpu().numpy().ravel()
            all_probs.append(probs)
            all_targets.append(targets)

            batch_dice = dice_score(outputs, masks)
            total_loss += loss.item()
            total_dice += batch_dice
            total_iou += iou_score(outputs, masks)
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_dice:.4f}")

    return (
        total_loss / len(loader),
        total_dice / len(loader),
        total_iou / len(loader),
        auc_score(all_probs, all_targets),
        all_probs,
        all_targets,
    )


def train_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dev_paths: List[str],
    dev_labels: List[str],
    train_transform: A.Compose,
    val_transform: A.Compose,
    config: Dict,
    save_root: str,
    device: torch.device,
):
    train_paths = [dev_paths[i] for i in train_idx]
    train_labels = [dev_labels[i] for i in train_idx]
    val_paths = [dev_paths[i] for i in val_idx]
    val_labels = [dev_labels[i] for i in val_idx]

    train_dataset = BUSIDataset(train_paths, train_labels, train_transform)
    val_dataset = BUSIDataset(val_paths, val_labels, val_transform)

    label_to_idx = {"benign": 0, "malignant": 1, "normal": 2}
    class_counts = {0: 0, 1: 0, 2: 0}
    for label in train_labels:
        class_counts[label_to_idx[label]] += 1
    total = sum(class_counts.values())
    class_weights = {k: total / v if v > 0 else 0 for k, v in class_counts.items()}
    sample_weights = [class_weights[label_to_idx[label]] for label in train_labels]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
    )

    model = build_model(config).to(device)
    criterion = SegmentationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config["lr_patience"],
        factor=config["lr_factor"],
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_dice = 0.0
    best_epoch = -1
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": [], "val_auc": [], "lr": []}

    for epoch in range(config["num_epochs"]):
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_dice, val_iou, val_auc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_auc"].append(val_auc)
        history["lr"].append(lr_now)

        print(
            f"[fold {fold + 1} | ep {epoch + 1:03d}] "
            f"lr={lr_now:.2e} loss {train_loss:.4f}/{val_loss:.4f} "
            f"dice {train_dice:.4f}/{val_dice:.4f} iou {val_iou:.4f} auc {format_auc(val_auc)}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "val_auc": val_auc,
                    "config": config,
                    "fold": fold + 1,
                },
                fold_model_path(save_root, fold),
            )
        else:
            patience_counter += 1

        if patience_counter >= config["early_stopping_patience"]:
            print(f"early stop at epoch {epoch + 1}")
            break

    return history, best_dice, best_epoch


def run_cross_validation(
    dev_paths: List[str],
    dev_labels: List[str],
    train_transform: A.Compose,
    val_transform: A.Compose,
    config: Dict,
    save_root: str,
    device: torch.device,
):
    kfold = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=42)

    all_histories = []
    fold_scores = []
    fold_best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dev_paths, dev_labels)):
        print(f"\nFold {fold + 1}/{config['n_folds']}")
        history, best_dice, best_epoch = train_fold(
            fold,
            train_idx,
            val_idx,
            dev_paths,
            dev_labels,
            train_transform,
            val_transform,
            config,
            save_root,
            device,
        )
        all_histories.append(history)
        fold_scores.append(best_dice)
        fold_best_epochs.append(best_epoch)
        print(f"best val dice: {best_dice:.4f} at epoch {best_epoch}")

    best_fold = int(np.argmax(fold_scores))
    shutil.copyfile(fold_model_path(save_root, best_fold), main_model_path(save_root))

    print(f"\nmean val dice: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    print(f"best overall fold: {best_fold + 1}")
    print(f"saved main checkpoint: {main_model_path(save_root)}")

    return all_histories, fold_scores, fold_best_epochs, best_fold


def plot_training_curves(all_histories: List[Dict], fold_scores: List[float], config: Dict, save_root: str) -> None:
    n = config["n_folds"]
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 11))

    for fold, h in enumerate(all_histories):
        ep = range(1, len(h["train_loss"]) + 1)

        axes[0, fold].plot(ep, h["train_loss"], label="train", color="royalblue")
        axes[0, fold].plot(ep, h["val_loss"], label="val", color="tomato")
        axes[0, fold].set_title(f"Fold {fold + 1} - Loss", fontsize=10)
        axes[0, fold].set_xlabel("epoch")
        axes[0, fold].legend()
        axes[0, fold].grid(alpha=0.4)

        axes[1, fold].plot(ep, h["val_dice"], label="dice", color="mediumseagreen")
        axes[1, fold].plot(ep, h["val_iou"], label="iou", color="mediumpurple")
        axes[1, fold].set_ylim(0, 1)
        axes[1, fold].set_title(f"Fold {fold + 1} - Dice / IoU", fontsize=10)
        axes[1, fold].set_xlabel("epoch")
        axes[1, fold].legend()
        axes[1, fold].grid(alpha=0.4)

        aucs = [a for a in h["val_auc"] if not np.isnan(a)]
        if aucs:
            axes[2, fold].plot(range(1, len(aucs) + 1), aucs, label="AUC-ROC", color="darkorange")
        axes[2, fold].set_ylim(0, 1)
        axes[2, fold].set_title(f"Fold {fold + 1} - AUC-ROC", fontsize=10)
        axes[2, fold].set_xlabel("epoch")
        axes[2, fold].legend()
        axes[2, fold].grid(alpha=0.4)

    plt.suptitle("segment_1 - Training Curves (ResNet34 + U-Net)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()

    colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple", "sandybrown"]
    x = np.arange(n)
    mean_val = np.mean(fold_scores)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bars = axes[0].bar(x, fold_scores, color=colors, edgecolor="black", width=0.5)
    axes[0].axhline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"mean = {mean_val:.4f}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Fold {i + 1}" for i in range(n)])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Best Dice")
    axes[0].set_title("Per-Fold Best Val Dice")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.4)
    for bar, score in zip(bars, fold_scores):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    mean_aucs = [np.nanmean(h["val_auc"]) for h in all_histories]
    bars2 = axes[1].bar(x, mean_aucs, color=colors, edgecolor="black", width=0.5)
    axes[1].axhline(np.nanmean(mean_aucs), color="red", linestyle="--", linewidth=1.5, label=f"mean = {np.nanmean(mean_aucs):.4f}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Fold {i + 1}" for i in range(n)])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean AUC-ROC")
    axes[1].set_title("Per-Fold Mean AUC-ROC")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.4)
    for bar, score in zip(bars2, mean_aucs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.suptitle("segment_1 - Fold Score Summary", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "fold_scores.png"), dpi=150)
    plt.show()


def evaluate_test_set(
    test_paths: List[str],
    test_labels: List[str],
    val_transform: A.Compose,
    config: Dict,
    save_root: str,
    device: torch.device,
):
    test_dataset = BUSIDataset(test_paths, test_labels, val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    criterion = SegmentationLoss()

    fold_test_dices: List[float] = []
    fold_test_ious: List[float] = []
    fold_test_aucs: List[float] = []
    fold_roc_data: List[Tuple[int, np.ndarray, np.ndarray, float]] = []
    fold_ids: List[int] = []

    for fold in range(config["n_folds"]):
        ckpt_path = fold_model_path(save_root, fold)
        if not os.path.exists(ckpt_path):
            print(f"fold {fold + 1}: checkpoint not found, skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        _, test_dice, test_iou, test_auc, all_probs, all_targets = validate(model, test_loader, criterion, device)
        fold_test_dices.append(test_dice)
        fold_test_ious.append(test_iou)
        fold_test_aucs.append(test_auc)
        fold_ids.append(fold + 1)

        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets).astype(int)
        if targets.sum() > 0 and (1 - targets).sum() > 0:
            fpr, tpr, _ = roc_curve(targets, probs)
            fold_roc_data.append((fold + 1, fpr, tpr, test_auc))

        print(f"fold {fold + 1}  dice {test_dice:.4f}  iou {test_iou:.4f}  auc {format_auc(test_auc)}")

    print(f"\nmean dice: {np.mean(fold_test_dices):.4f} +/- {np.std(fold_test_dices):.4f}")
    print(f"mean iou:  {np.mean(fold_test_ious):.4f} +/- {np.std(fold_test_ious):.4f}")
    print(f"mean auc:  {np.nanmean(fold_test_aucs):.4f} +/- {np.nanstd(fold_test_aucs):.4f}")

    return fold_test_dices, fold_test_ious, fold_test_aucs, fold_roc_data, fold_ids


def plot_test_roc(fold_roc_data: List[Tuple[int, np.ndarray, np.ndarray, float]], save_root: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    palette = ["steelblue", "coral", "mediumseagreen", "mediumpurple", "sandybrown"]
    for i, (fold_id, fpr, tpr, auc) in enumerate(fold_roc_data):
        ax.plot(fpr, tpr, color=palette[i % len(palette)], lw=1.5, label=f"Fold {fold_id}  AUC={format_auc(auc)}")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Test Set (pixel-level)")
    if fold_roc_data:
        ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "roc_curve.png"), dpi=150)
    plt.show()


def print_summary_table(
    fold_ids: List[int],
    fold_scores: List[float],
    fold_best_epochs: List[int],
    fold_test_dices: List[float],
    fold_test_ious: List[float],
    fold_test_aucs: List[float],
    save_root: str,
) -> None:
    rows = []
    for idx, fold_id in enumerate(fold_ids):
        rows.append(
            {
                "Fold": fold_id,
                "Val Dice": round(fold_scores[fold_id - 1], 4),
                "Best Epoch": fold_best_epochs[fold_id - 1],
                "Test Dice": round(fold_test_dices[idx], 4),
                "Test IoU": round(fold_test_ious[idx], 4),
                "Test AUC": round(fold_test_aucs[idx], 4) if not np.isnan(fold_test_aucs[idx]) else float("nan"),
                "Checkpoint": os.path.basename(fold_model_path(save_root, fold_id - 1)),
            }
        )

    rows.append(
        {
            "Fold": "mean +/- std",
            "Val Dice": f"{np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}",
            "Best Epoch": "-",
            "Test Dice": f"{np.mean(fold_test_dices):.4f} +/- {np.std(fold_test_dices):.4f}",
            "Test IoU": f"{np.mean(fold_test_ious):.4f} +/- {np.std(fold_test_ious):.4f}",
            "Test AUC": f"{np.nanmean(fold_test_aucs):.4f} +/- {np.nanstd(fold_test_aucs):.4f}",
            "Checkpoint": os.path.basename(main_model_path(save_root)),
        }
    )

    df = pd.DataFrame(rows).set_index("Fold")
    print("segment_1 - ResNet34 | U-Net | BCE+Dice | ReduceLROnPlateau | 256px\n")
    print(df.to_string())


def qualitative_visualization(
    fold_scores: List[float],
    test_paths: List[str],
    test_labels: List[str],
    config: Dict,
    save_root: str,
    device: torch.device,
) -> None:
    best_fold = int(np.argmax(fold_scores))
    ckpt = torch.load(fold_model_path(save_root, best_fold), map_location=device, weights_only=False)
    vis_model = build_model(config).to(device)
    vis_model.load_state_dict(ckpt["model_state_dict"])
    vis_model.eval()

    sample_indices = random.sample(range(len(test_paths)), min(10, len(test_paths)))
    mean_np = np.array([0.485, 0.456, 0.406])
    std_np = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(len(sample_indices), 3, figsize=(12, 4 * len(sample_indices)))
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(sample_indices):
        img_path = test_paths[idx]
        label = test_labels[idx]

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rs = cv2.resize(img_rgb, (config["input_size"], config["input_size"]))

        mask_raw = load_all_masks(img_path)
        if mask_raw is not None:
            gt_mask = cv2.resize(mask_raw, (config["input_size"], config["input_size"])) / 255.0
        else:
            gt_mask = np.zeros((config["input_size"], config["input_size"]), dtype=np.float32)

        norm_img = ((img_rs / 255.0) - mean_np) / std_np
        tensor_img = torch.tensor(norm_img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_logit = vis_model(tensor_img)
            pred_prob = torch.sigmoid(pred_logit).squeeze().cpu().numpy()

        pred_binary = (pred_prob > 0.5).astype(np.float32)
        intersection = (pred_binary * gt_mask).sum()
        dsc = (2.0 * intersection + 1e-6) / (pred_binary.sum() + gt_mask.sum() + 1e-6)

        axes[row, 0].imshow(img_rs)
        axes[row, 0].set_title(label, fontsize=9)
        axes[row, 1].imshow(gt_mask, cmap="gray")
        axes[row, 1].set_title("Ground Truth", fontsize=9)
        axes[row, 2].imshow(pred_binary, cmap="gray")
        axes[row, 2].set_title(f"Prediction  dice={dsc:.3f}", fontsize=9)

        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle(f"Qualitative Results - best fold {best_fold + 1}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "qualitative_viz.png"), dpi=150)
    plt.show()


def main(data_root: str = DEFAULT_DATA_ROOT, split_file: str = DEFAULT_SPLIT_FILE, save_root: str = DEFAULT_SAVE_ROOT) -> None:
    os.makedirs(save_root, exist_ok=True)
    set_seed(42)
    device = setup_device()
    print(f"device: {device}")

    dev_paths, dev_labels, test_paths, test_labels = load_data_split(split_file)
    print(f"dev: {len(dev_paths)} | test: {len(test_paths)}")

    train_transform, val_transform = build_transforms(CONFIG)

    all_histories, fold_scores, fold_best_epochs, _ = run_cross_validation(
        dev_paths,
        dev_labels,
        train_transform,
        val_transform,
        CONFIG,
        save_root,
        device,
    )

    plot_training_curves(all_histories, fold_scores, CONFIG, save_root)

    fold_test_dices, fold_test_ious, fold_test_aucs, fold_roc_data, fold_ids = evaluate_test_set(
        test_paths,
        test_labels,
        val_transform,
        CONFIG,
        save_root,
        device,
    )

    plot_test_roc(fold_roc_data, save_root)

    print_summary_table(
        fold_ids,
        fold_scores,
        fold_best_epochs,
        fold_test_dices,
        fold_test_ious,
        fold_test_aucs,
        save_root,
    )

    qualitative_visualization(fold_scores, test_paths, test_labels, CONFIG, save_root, device)


if __name__ == "__main__":
    main()
