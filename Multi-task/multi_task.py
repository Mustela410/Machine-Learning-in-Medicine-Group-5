import os
import re
import math
import time
import random
import warnings
import subprocess
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm


optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def install_dependencies() -> None:
    run_cmd(["python", "-m", "pip", "uninstall", "-y", "albumentations", "albucore"])
    run_cmd(["python", "-m", "pip", "install", "albumentations==1.4.7", "opencv-python-headless==4.10.0.84"])
    run_cmd(["python", "-m", "pip", "install", "segmentation-models-pytorch", "optuna", "plotly"])


def download_data_with_kaggle(dataset: str = "aryashah2k/breast-ultrasound-images-dataset") -> None:
    run_cmd(["python", "-m", "pip", "install", "-q", "kaggle"])
    run_cmd(["kaggle", "datasets", "download", "-d", dataset])
    run_cmd(["python", "-m", "zipfile", "-e", "breast-ultrasound-images-dataset.zip", "bus_dataset/"])


CLASS_TO_ID = {"normal": 0, "benign": 1, "malignant": 2}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMG_SIZE = 256

W_BCE_SEG = 0.5
W_DICE_SEG = 0.5

OPTUNA_N_TRIALS = 20
OPTUNA_N_FOLDS = 3
OPTUNA_N_EPOCHS = 15
OPTUNA_BATCH = 16
OPTUNA_LR = 3e-4
ALPHA_LOW = 0.1
ALPHA_HIGH = 0.9
ALPHA_STEP = 0.05

FINAL_N_FOLDS = 5
FINAL_N_EPOCHS = 80
FINAL_BATCH = 16
FINAL_LR = 3e-4
PATIENCE = 15


# If you want to override manually like the notebook's final section, set here.
MANUAL_BEST_ALPHA = 0.2


def is_mask_file(filename: str) -> bool:
    return "_mask" in filename.lower()


def get_base_key(filename: str) -> str:
    name, _ = os.path.splitext(filename)
    return re.sub(r"_mask(_\d+)?$", "", name, flags=re.IGNORECASE)


def build_dataframe(root: str) -> pd.DataFrame:
    records = []

    for cls_name in ["normal", "benign", "malignant"]:
        cls_dir = os.path.join(root, cls_name)
        files_list = sorted(os.listdir(cls_dir))
        image_map, mask_map = {}, {}

        for f in files_list:
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMG_EXTS:
                continue
            full_path = os.path.join(cls_dir, f)
            key = get_base_key(f)
            if is_mask_file(f):
                mask_map.setdefault(key, []).append(full_path)
            else:
                image_map[key] = full_path

        for key, img_path in image_map.items():
            records.append(
                {
                    "image_path": img_path,
                    "mask_paths": sorted(mask_map.get(key, [])),
                    "class_name": cls_name,
                    "class_id": CLASS_TO_ID[cls_name],
                }
            )

    return pd.DataFrame(records)


def make_transforms():
    train_tfms = A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.4,
            ),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.15),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    valid_tfms = A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return train_tfms, valid_tfms


class BUSIMultiTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def _read_image(self, path: str):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_and_merge_masks(self, mask_paths: List[str], image_shape):
        h, w = image_shape[:2]
        merged = np.zeros((h, w), dtype=np.uint8)
        for mp in mask_paths:
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            merged = np.maximum(merged, (m > 0).astype(np.uint8))
        return merged

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = self._read_image(row["image_path"])
        class_id = int(row["class_id"])

        mask_paths = row["mask_paths"]
        mask = (
            self._read_and_merge_masks(mask_paths, image.shape)
            if len(mask_paths) > 0
            else np.zeros(image.shape[:2], dtype=np.uint8)
        )

        if self.transforms:
            aug = self.transforms(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        mask = mask.float().unsqueeze(0)
        label = torch.tensor(class_id, dtype=torch.long)
        return image, mask, label


class MultiTaskUNet(nn.Module):
    def __init__(self, encoder_name: str = "resnet34", encoder_weights: str = "imagenet", num_classes: int = 3):
        super().__init__()
        self.seg_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
        encoder_channels = self.seg_model.encoder.out_channels[-1]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(encoder_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.seg_model.encoder(x)
        decoder_output = self.seg_model.decoder(features)
        seg_logits = self.seg_model.segmentation_head(decoder_output)
        cls_logits = self.cls_head(features[-1])
        return seg_logits, cls_logits


def dice_loss(logits, targets, smooth: float = 1e-6):
    probs = torch.sigmoid(logits).view(-1)
    targets = targets.view(-1)
    inter = (probs * targets).sum()
    return 1 - (2.0 * inter + smooth) / (probs.sum() + targets.sum() + smooth)


def seg_loss_fixed(seg_logits, masks, bce_fn):
    return W_BCE_SEG * bce_fn(seg_logits, masks) + W_DICE_SEG * dice_loss(seg_logits, masks)


def harmonic_mean(a, b, eps: float = 1e-8):
    return (2 * a * b) / (a + b + eps)


def dice_score_batch(logits, targets, threshold: float = 0.5, smooth: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    dice = (2.0 * inter + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)
    return dice.mean().item()


def iou_score_batch(logits, targets, threshold: float = 0.5, smooth: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


def run_epoch(model, loader, device, optimizer=None, train: bool = True, alpha: float = 0.5, bce_fn=None, ce_fn=None):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Train" if train else "Val", leave=False)
    for images, masks, labels in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            seg_logits, cls_logits = model(images)
            s_loss = seg_loss_fixed(seg_logits, masks, bce_fn)
            c_loss = ce_fn(cls_logits, labels)
            loss = alpha * c_loss + (1.0 - alpha) * s_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        dice = dice_score_batch(seg_logits.detach(), masks)
        iou = iou_score_batch(seg_logits.detach(), masks)
        preds = torch.argmax(cls_logits, dim=1)

        total_loss += loss.item()
        total_seg_loss += s_loss.item()
        total_cls_loss += c_loss.item()
        total_dice += dice
        total_iou += iou
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}"})

    n = len(loader)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    avg_dice = total_dice / n
    hm_score = harmonic_mean(macro_f1, avg_dice)

    return {
        "loss": total_loss / n,
        "seg_loss": total_seg_loss / n,
        "cls_loss": total_cls_loss / n,
        "dice": avg_dice,
        "iou": total_iou / n,
        "acc": acc,
        "macro_f1": macro_f1,
        "hm_score": hm_score,
    }


def objective(trial, train_val_df, device, train_tfms, valid_tfms):
    alpha = trial.suggest_float("alpha", ALPHA_LOW, ALPHA_HIGH, step=ALPHA_STEP)

    skf = StratifiedKFold(n_splits=OPTUNA_N_FOLDS, shuffle=True, random_state=42)
    fold_scores = []
    bce_fn = nn.BCEWithLogitsLoss()
    ce_fn = nn.CrossEntropyLoss()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df["class_id"])):
        train_fold = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_fold = train_val_df.iloc[val_idx].reset_index(drop=True)

        train_loader = DataLoader(BUSIMultiTaskDataset(train_fold, train_tfms), batch_size=OPTUNA_BATCH, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(BUSIMultiTaskDataset(val_fold, valid_tfms), batch_size=OPTUNA_BATCH, shuffle=False, num_workers=2, pin_memory=True)

        model = MultiTaskUNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=OPTUNA_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=OPTUNA_N_EPOCHS, eta_min=1e-6)

        best_hm = -np.inf
        for epoch in range(OPTUNA_N_EPOCHS):
            run_epoch(model, train_loader, device, optimizer=optimizer, train=True, alpha=alpha, bce_fn=bce_fn, ce_fn=ce_fn)
            scheduler.step()
            val_m = run_epoch(model, val_loader, device, optimizer=None, train=False, alpha=alpha, bce_fn=bce_fn, ce_fn=ce_fn)
            hm = val_m["hm_score"]

            if hm > best_hm:
                best_hm = hm

            global_step = fold_idx * OPTUNA_N_EPOCHS + epoch
            trial.report(hm, step=global_step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        fold_scores.append(best_hm)
        del model
        torch.cuda.empty_cache()

    return float(np.mean(fold_scores))


def run_optuna(train_val_df, device, train_tfms, valid_tfms):
    print("Start Optuna search")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name="alpha_harmonic_search",
    )

    def print_callback(study_obj, trial):
        alpha_val = trial.params.get("alpha", None)
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"trial {trial.number:03d} alpha={alpha_val:.2f} hm={trial.value:.4f} best={study_obj.best_value:.4f}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"trial {trial.number:03d} alpha={alpha_val:.2f} pruned")

    study.optimize(
        lambda t: objective(t, train_val_df, device, train_tfms, valid_tfms),
        n_trials=OPTUNA_N_TRIALS,
        callbacks=[print_callback],
        show_progress_bar=False,
    )

    best_alpha = study.best_params["alpha"]
    print(f"best alpha: {best_alpha:.2f}")
    print(f"best hm: {study.best_value:.4f}")

    return study, best_alpha


def train_final(train_val_df, device, train_tfms, valid_tfms, best_alpha, save_root):
    skf_final = StratifiedKFold(n_splits=FINAL_N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    best_val_hm = -np.inf
    best_model_wts = None

    bce_fn = nn.BCEWithLogitsLoss()
    ce_fn = nn.CrossEntropyLoss()

    for fold, (train_idx, val_idx) in enumerate(skf_final.split(train_val_df, train_val_df["class_id"])):
        print(f"fold {fold + 1}/{FINAL_N_FOLDS} alpha={best_alpha:.2f}")

        train_fold = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_fold = train_val_df.iloc[val_idx].reset_index(drop=True)

        train_loader = DataLoader(BUSIMultiTaskDataset(train_fold, train_tfms), batch_size=FINAL_BATCH, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(BUSIMultiTaskDataset(val_fold, valid_tfms), batch_size=FINAL_BATCH, shuffle=False, num_workers=2, pin_memory=True)

        model = MultiTaskUNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=FINAL_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_N_EPOCHS, eta_min=1e-6)

        fold_best_hm = -np.inf
        fold_best_wts = None
        no_improve = 0

        history = {k: [] for k in ["t_loss", "v_loss", "t_dice", "v_dice", "t_f1", "v_f1", "t_hm", "v_hm"]}

        for epoch in range(FINAL_N_EPOCHS):
            t = run_epoch(model, train_loader, device, optimizer=optimizer, train=True, alpha=best_alpha, bce_fn=bce_fn, ce_fn=ce_fn)
            v = run_epoch(model, val_loader, device, optimizer=None, train=False, alpha=best_alpha, bce_fn=bce_fn, ce_fn=ce_fn)
            scheduler.step()

            for key, val in zip(
                ["t_loss", "v_loss", "t_dice", "v_dice", "t_f1", "v_f1", "t_hm", "v_hm"],
                [t["loss"], v["loss"], t["dice"], v["dice"], t["macro_f1"], v["macro_f1"], t["hm_score"], v["hm_score"]],
            ):
                history[key].append(val)

            print(
                f"ep {epoch + 1:03d} "
                f"loss {t['loss']:.4f}/{v['loss']:.4f} "
                f"dice {t['dice']:.4f}/{v['dice']:.4f} "
                f"f1 {t['macro_f1']:.4f}/{v['macro_f1']:.4f} "
                f"hm {t['hm_score']:.4f}/{v['hm_score']:.4f}"
            )

            if v["hm_score"] > fold_best_hm:
                fold_best_hm = v["hm_score"]
                fold_best_wts = {k: v2.clone() for k, v2 in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"early stopping at epoch {epoch + 1}")
                    break

        if fold_best_hm > best_val_hm:
            best_val_hm = fold_best_hm
            best_model_wts = fold_best_wts

        model.load_state_dict(fold_best_wts)
        with torch.no_grad():
            fv = run_epoch(model, val_loader, device, optimizer=None, train=False, alpha=best_alpha, bce_fn=bce_fn, ce_fn=ce_fn)

        fold_results.append(
            {
                "fold": fold + 1,
                "val_loss": fv["loss"],
                "val_dice": fv["dice"],
                "val_iou": fv["iou"],
                "val_acc": fv["acc"],
                "val_f1": fv["macro_f1"],
                "val_hm": fv["hm_score"],
            }
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        pairs = [
            ("t_loss", "v_loss", "Loss"),
            ("t_dice", "v_dice", "Dice Score"),
            ("t_f1", "v_f1", "Macro F1"),
            ("t_hm", "v_hm", "Harmonic Mean(F1, Dice)"),
        ]
        for ax, (tk, vk, title) in zip(axes.flat, pairs):
            ax.plot(history[tk], label="Train", linewidth=1.5)
            ax.plot(history[vk], label="Val", linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_root, f"fold{fold + 1}_curves.png"), dpi=120, bbox_inches="tight")
        plt.show()

        del model
        torch.cuda.empty_cache()

    return fold_results, best_model_wts


def evaluate_test(best_model_wts, test_df, device, valid_tfms, best_alpha):
    best_model = MultiTaskUNet().to(device)
    best_model.load_state_dict(best_model_wts)
    best_model.eval()

    test_ds = BUSIMultiTaskDataset(test_df, transforms=valid_tfms)
    test_loader = DataLoader(test_ds, batch_size=FINAL_BATCH, shuffle=False, num_workers=2, pin_memory=True)

    bce_fn = nn.BCEWithLogitsLoss()
    ce_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        test_m = run_epoch(best_model, test_loader, device, optimizer=None, train=False, alpha=best_alpha, bce_fn=bce_fn, ce_fn=ce_fn)

    return best_model, test_ds, test_loader, test_m


def plot_class_report(best_model, test_loader, device, best_alpha, save_root):
    all_preds, all_labels = [], []

    best_model.eval()
    with torch.no_grad():
        for images, masks, labels in tqdm(test_loader, desc="Collecting preds"):
            images = images.to(device)
            _, cls_logits = best_model(images)
            preds = torch.argmax(cls_logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    id_to_class = {0: "normal", 1: "benign", 2: "malignant"}
    print(classification_report(all_labels, all_preds, target_names=[id_to_class[i] for i in range(3)], digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=[id_to_class[i] for i in range(3)], columns=[id_to_class[i] for i in range(3)])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=0.5, linecolor="gray")
    ax.set_title(f"Confusion Matrix - Test Set (alpha={best_alpha:.2f})")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc(best_model, test_loader, device, best_alpha, save_root):
    all_probs, all_labels_roc = [], []

    best_model.eval()
    with torch.no_grad():
        for images, masks, labels in tqdm(test_loader, desc="ROC probs"):
            images = images.to(device)
            _, cls_logits = best_model(images)
            probs = torch.softmax(cls_logits, dim=1)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels_roc.extend(labels.numpy().tolist())

    y_test_bin = label_binarize(all_labels_roc, classes=[0, 1, 2])
    y_score = np.array(all_probs)
    n_classes = 3

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = sum(np.interp(all_fpr, fpr[i], tpr[i]) for i in range(n_classes)) / n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    id_to_class = {0: "Normal", 1: "Benign", 2: "Malignant"}
    colors = ["aqua", "darkorange", "cornflowerblue"]

    plt.figure(figsize=(10, 8))
    plt.plot(fpr["macro"], tpr["macro"], label=f"Macro-avg ROC (AUC={roc_auc['macro']:.4f})", color="navy", linestyle=":", linewidth=4)
    for i, c in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=c, lw=2, label=f"{id_to_class[i]} (AUC={roc_auc[i]:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC AUC - Test Set (alpha={best_alpha:.2f})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_root, "roc_auc.png"), dpi=150, bbox_inches="tight")
    plt.show()


def denorm_image(x):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.clip(x * std + mean, 0, 1)


def show_predictions(dataset, model, device, best_alpha, save_root, num_samples: int = 5):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    id_to_class = {0: "normal", 1: "benign", 2: "malignant"}

    plt.figure(figsize=(14, 4 * num_samples))
    for i, idx in enumerate(indices):
        image, mask, label = dataset[idx]
        with torch.no_grad():
            seg_logits, cls_logits = model(image.unsqueeze(0).to(device))
            pred_mask = (torch.sigmoid(seg_logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
            pred_cls = torch.argmax(cls_logits, dim=1).item()

        img_np = denorm_image(image.permute(1, 2, 0).cpu().numpy())
        true_mask = mask[0].cpu().numpy()
        match = "Yes" if pred_cls == label.item() else "No"

        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.imshow(img_np)
        plt.title(f"Image {match}\nTrue: {id_to_class[label.item()]}\nPred: {id_to_class[pred_cls]}")
        plt.axis("off")

        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.imshow(true_mask, cmap="gray")
        plt.title("True Mask")
        plt.axis("off")

        plt.subplot(num_samples, 3, 3 * i + 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "sample_predictions.png"), dpi=120, bbox_inches="tight")
    plt.show()


def print_summary(fold_results, test_m, best_alpha):
    fold_only = pd.DataFrame(fold_results)
    metric_cols = ["val_loss", "val_dice", "val_iou", "val_acc", "val_f1", "val_hm"]

    print("summary")
    print("architecture: MultiTaskUNet (ResNet34 encoder)")
    print(f"seg loss: {W_BCE_SEG}*BCE + {W_DICE_SEG}*Dice")
    print(f"best alpha: {best_alpha:.2f}")
    print(f"total loss: {best_alpha:.2f}*cls + {1 - best_alpha:.2f}*seg")

    print("5-fold validation mean +/- std")
    for col in metric_cols:
        m, s = fold_only[col].mean(), fold_only[col].std()
        print(f"{col}: {m:.4f} +/- {s:.4f}")

    print("test set")
    for k, v in test_m.items():
        print(f"{k}: {v:.4f}")


def save_model(best_model_wts, best_alpha, test_m, fold_results, save_root):
    out_path = os.path.join(save_root, "best_model_harmonic.pth")
    torch.save(
        {
            "model_state_dict": best_model_wts,
            "best_alpha": best_alpha,
            "test_metrics": test_m,
            "fold_results": fold_results,
        },
        out_path,
    )
    print(out_path)


def main(root: str = "/content/bus_dataset/Dataset_BUSI_with_GT", save_root: str = "/content/drive/MyDrive/medicine_real3"):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    os.makedirs(save_root, exist_ok=True)

    df = build_dataframe(root)
    print(f"total samples: {len(df)}")
    print(df["class_name"].value_counts())

    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["class_id"])
    train_val_df = train_val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"train+val: {len(train_val_df)}")
    print(f"test: {len(test_df)}")

    train_tfms, valid_tfms = make_transforms()

    study, best_alpha_optuna = run_optuna(train_val_df, device, train_tfms, valid_tfms)

    # Keep compatibility with the notebook's final manual override.
    best_alpha = MANUAL_BEST_ALPHA if MANUAL_BEST_ALPHA is not None else best_alpha_optuna
    print(f"using alpha for final training: {best_alpha:.2f}")

    fold_results, best_model_wts = train_final(train_val_df, device, train_tfms, valid_tfms, best_alpha, save_root)

    best_model, test_ds, test_loader, test_m = evaluate_test(best_model_wts, test_df, device, valid_tfms, best_alpha)
    plot_class_report(best_model, test_loader, device, best_alpha, save_root)
    plot_roc(best_model, test_loader, device, best_alpha, save_root)
    show_predictions(test_ds, best_model, device, best_alpha, save_root, num_samples=5)

    print_summary(fold_results, test_m, best_alpha)
    save_model(best_model_wts, best_alpha, test_m, fold_results, save_root)


if __name__ == "__main__":
    main()
