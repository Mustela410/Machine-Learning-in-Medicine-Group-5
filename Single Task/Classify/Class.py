import json
import os
import shutil
import random
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

DEFAULT_DATA_ROOT = "/content/drive/MyDrive/BUSI"
DEFAULT_SPLIT_FILE = os.path.join(DEFAULT_DATA_ROOT, "data_split_85_15.json")
DEFAULT_SAVE_ROOT = "/content/drive/MyDrive/BUSI1"
MODEL_PREFIX = "class_3class"

CONFIG = {
    "input_size": 256,
    "batch_size": 16,
    "num_epochs": 100,
    "num_workers": 8,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "lr_patience": 4,
    "lr_factor": 0.5,
    "early_stopping_patience": 15,
    "n_folds": 5,
}


def fold_model_path(save_root: str, fold: int) -> str:
    return os.path.join(save_root, f"{MODEL_PREFIX}_fold_{fold + 1}.pth")


def main_model_path(save_root: str) -> str:
    return os.path.join(save_root, f"{MODEL_PREFIX}.pth")


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


def load_data_split(split_file: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    with open(split_file, "r", encoding="utf-8") as f:
        data_split = json.load(f)

    dev_images = data_split["development"]
    test_images = data_split["test"]

    dev_paths = [img["path"] for img in dev_images]
    dev_labels = [img["label"] for img in dev_images]
    test_paths = [img["path"] for img in test_images]
    test_labels = [img["label"] for img in test_images]

    dev_pairs = sorted(zip(dev_paths, dev_labels))
    test_pairs = sorted(zip(test_paths, test_labels))
    dev_paths_sorted = [p for p, _ in dev_pairs]
    dev_labels_sorted = [l for _, l in dev_pairs]
    test_paths_sorted = [p for p, _ in test_pairs]
    test_labels_sorted = [l for _, l in test_pairs]

    return dev_paths_sorted, dev_labels_sorted, test_paths_sorted, test_labels_sorted


class CropDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], labels: Sequence[str], transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform
        self.label_map = {"normal": 0, "benign": 1, "malignant": 2}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.label_map[self.labels[idx]]
        return image, torch.tensor(label, dtype=torch.long)


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


def build_model() -> nn.Module:
    model = models.resnet34(weights="DEFAULT")
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 3),
    )
    return model


def safe_auc_multiclass(y_true, y_prob, n_classes: int = 3) -> float:
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))


def format_auc(value: float) -> str:
    return f"{value:.4f}" if not np.isnan(value) else "n/a"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_probs = []
    all_labels = []
    all_preds = []
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        probs = torch.softmax(logits, dim=1).float().detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        running_loss += loss.item() * images.size(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = safe_auc_multiclass(all_labels, all_probs, n_classes=3)
    epoch_acc = float((np.array(all_preds) == np.array(all_labels)).mean())
    return epoch_loss, epoch_auc, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []
    all_preds = []
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1).float().cpu().numpy()
            preds = np.argmax(probs, axis=1)
            running_loss += loss.item() * images.size(0)
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())

    all_labels_np = np.array(all_labels).astype(int)
    all_probs_np = np.array(all_probs)
    all_preds_np = np.array(all_preds)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = safe_auc_multiclass(all_labels_np, all_probs_np, n_classes=3)
    epoch_acc = float((all_preds_np == all_labels_np).mean())

    return epoch_loss, epoch_auc, epoch_acc, all_probs_np, all_labels_np, all_preds_np


def train_fold(
    fold: int,
    train_idx,
    val_idx,
    dev_paths_sorted,
    dev_labels_sorted,
    config,
    device,
    train_transform,
    val_transform,
    save_root,
):
    train_paths = [dev_paths_sorted[i] for i in train_idx]
    train_labels = [dev_labels_sorted[i] for i in train_idx]
    val_paths = [dev_paths_sorted[i] for i in val_idx]
    val_labels = [dev_labels_sorted[i] for i in val_idx]

    train_dataset = CropDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = CropDataset(val_paths, val_labels, transform=val_transform)

    train_label_ids = [train_dataset.label_map[l] for l in train_labels]
    class_counts = [max(train_label_ids.count(i), 1) for i in range(3)]
    class_weights = [sum(class_counts) / c for c in class_counts]
    sample_weights = [class_weights[label_id] for label_id in train_label_ids]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config["lr_patience"],
        factor=config["lr_factor"],
    )

    best_val_loss = float("inf")
    best_val_auc = float("nan")
    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_auc": [],
        "train_acc": [],
        "val_loss": [],
        "val_auc": [],
        "val_acc": [],
        "lr": [],
    }

    for epoch in range(config["num_epochs"]):
        train_loss, train_auc, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_acc, _, _, _ = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"[fold {fold + 1} | ep {epoch + 1:03d}] "
            f"loss {train_loss:.4f}/{val_loss:.4f} "
            f"auc {format_auc(train_auc)}/{format_auc(val_auc)} "
            f"acc {train_acc:.4f}/{val_acc:.4f} lr {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_auc": best_val_auc,
                    "best_val_acc": best_val_acc,
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

    return {
        "fold": fold,
        "best_val_loss": best_val_loss,
        "best_val_auc": best_val_auc,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "history": history,
    }


def run_cross_validation(dev_paths_sorted, dev_labels_sorted, config, device, train_transform, val_transform, save_root):
    skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(dev_paths_sorted, dev_labels_sorted)):
        print(f"\nFold {fold + 1}/{config['n_folds']}")
        result = train_fold(
            fold,
            train_idx,
            val_idx,
            dev_paths_sorted,
            dev_labels_sorted,
            config,
            device,
            train_transform,
            val_transform,
            save_root,
        )
        fold_results.append(result)
        print(
            f"best val loss: {result['best_val_loss']:.4f} | "
            f"best val auc: {format_auc(result['best_val_auc'])} | "
            f"best val acc: {result['best_val_acc']:.4f} | epoch: {result['best_epoch']}"
        )

    best_fold = int(np.argmin([r["best_val_loss"] for r in fold_results]))
    shutil.copyfile(fold_model_path(save_root, best_fold), main_model_path(save_root))

    fold_losses = [r["best_val_loss"] for r in fold_results]
    fold_aucs = [r["best_val_auc"] for r in fold_results]
    fold_accs = [r["best_val_acc"] for r in fold_results]

    print(f"\nmean best val loss: {np.mean(fold_losses):.4f} +/- {np.std(fold_losses):.4f}")
    print(f"mean best val auc: {np.nanmean(fold_aucs):.4f} +/- {np.nanstd(fold_aucs):.4f}")
    print(f"mean best val acc: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print(f"saved main checkpoint: {main_model_path(save_root)}")

    return fold_results


def plot_cv_scores(fold_results, save_root):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    fold_numbers = [r["fold"] + 1 for r in fold_results]
    fold_losses = [r["best_val_loss"] for r in fold_results]
    fold_aucs = [r["best_val_auc"] for r in fold_results]
    fold_accs = [r["best_val_acc"] for r in fold_results]

    axes[0].bar(fold_numbers, fold_losses, color="steelblue", alpha=0.8)
    axes[0].axhline(np.mean(fold_losses), color="red", linestyle="--", label=f"mean={np.mean(fold_losses):.4f}")
    axes[0].set_xlabel("fold")
    axes[0].set_ylabel("best val loss")
    axes[0].set_title("fold best val loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(fold_numbers, fold_aucs, color="seagreen", alpha=0.8)
    axes[1].axhline(np.nanmean(fold_aucs), color="red", linestyle="--", label=f"mean={np.nanmean(fold_aucs):.4f}")
    axes[1].set_xlabel("fold")
    axes[1].set_ylabel("best val macro AUC")
    axes[1].set_title("fold best val AUC")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].bar(fold_numbers, fold_accs, color="mediumpurple", alpha=0.8)
    axes[2].axhline(np.mean(fold_accs), color="red", linestyle="--", label=f"mean={np.mean(fold_accs):.4f}")
    axes[2].set_xlabel("fold")
    axes[2].set_ylabel("best val accuracy")
    axes[2].set_title("fold best val accuracy")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "class_3class_cv_scores.png"), dpi=150, bbox_inches="tight")
    plt.show()


def evaluate_test_ensemble(test_paths_sorted, test_labels_sorted, config, device, val_transform, save_root):
    test_dataset = CropDataset(test_paths_sorted, test_labels_sorted, transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    all_fold_probs = []
    y_true = None
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    for fold in range(config["n_folds"]):
        ckpt_path = fold_model_path(save_root, fold)
        if not os.path.exists(ckpt_path):
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        fold_probs = []
        fold_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                    logits = model(images)
                probs = torch.softmax(logits, dim=1).float().cpu().numpy()
                fold_probs.extend(probs.tolist())
                fold_labels.extend(labels.numpy().astype(int).tolist())

        all_fold_probs.append(np.array(fold_probs))
        if y_true is None:
            y_true = np.array(fold_labels).astype(int)

    if len(all_fold_probs) == 0:
        raise RuntimeError("No fold checkpoint found. Run cross-validation first.")

    y_prob = np.mean(np.stack(all_fold_probs, axis=0), axis=0)
    y_pred = np.argmax(y_prob, axis=1)

    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    test_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")

    report_text = classification_report(y_true, y_pred, target_names=["normal", "benign", "malignant"], digits=4)
    print(report_text)
    print(f"macro auc: {test_auc:.4f}")

    return y_true, y_pred, y_prob


def plot_test_eval(y_true, y_pred, y_prob, save_root):
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    class_names = ["normal", "benign", "malignant"]
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        auc_i = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
        axes[0].plot(fpr, tpr, lw=2, label=f"{class_name} (AUC={auc_i:.4f})")

    axes[0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve One-vs-Rest (Test)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1], xticklabels=class_names, yticklabels=class_names)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix (Test 3-Class)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "class_3class_eval.png"), dpi=150, bbox_inches="tight")
    plt.show()


def main(data_root: str = DEFAULT_DATA_ROOT, split_file: str = DEFAULT_SPLIT_FILE, save_root: str = DEFAULT_SAVE_ROOT):
    os.makedirs(save_root, exist_ok=True)

    set_seed(42)
    device = setup_device()
    print(f"device: {device}")

    dev_paths_sorted, dev_labels_sorted, test_paths_sorted, test_labels_sorted = load_data_split(split_file)
    print(f"dev: {len(dev_paths_sorted)} | test: {len(test_paths_sorted)}")

    train_transform, val_transform = build_transforms(CONFIG)

    fold_results = run_cross_validation(
        dev_paths_sorted,
        dev_labels_sorted,
        CONFIG,
        device,
        train_transform,
        val_transform,
        save_root,
    )

    plot_cv_scores(fold_results, save_root)

    y_true, y_pred, y_prob = evaluate_test_ensemble(
        test_paths_sorted,
        test_labels_sorted,
        CONFIG,
        device,
        val_transform,
        save_root,
    )

    plot_test_eval(y_true, y_pred, y_prob, save_root)


if __name__ == "__main__":
    main()
