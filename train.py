import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
import argparse
import json
from datetime import datetime
from tqdm import tqdm

from model import FCT
from dataloader import create_dataloaders, normalize_images, load_camus_dataset


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.reshape(pred.size(0), pred.size(1), -1)
        target = target.reshape(target.size(0), target.size(1), -1)

        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        if torch.isnan(dice_loss) or torch.isinf(dice_loss):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


class DeepSupervisionLoss(nn.Module):
    def __init__(self, loss_fn, weights=[0.3, 0.3, 0.4]):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights

    def forward(self, outputs, target):
        out7, out8, out9 = outputs

        target7 = F.interpolate(target, size=out7.shape[2:], mode="nearest")
        target8 = F.interpolate(target, size=out8.shape[2:], mode="nearest")

        loss7 = self.loss_fn(out7, target7)
        loss8 = self.loss_fn(out8, target8)
        loss9 = self.loss_fn(out9, target)

        # from FCT
        total = self.weights[0] * loss7 + self.weights[1] * loss8 + self.weights[2] * loss9
        return total


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    def __init__(self, patience=20, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_dice):
        if self.best_score is None:
            self.best_score = val_dice
        elif val_dice < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_dice
            self.counter = 0
        return self.early_stop


def calculate_dice(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    pred_flat = pred_binary.reshape(pred_binary.size(0), pred_binary.size(1), -1)
    target_flat = target.reshape(target.size(0), target.size(1), -1)

    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return dice.mean().item()


def calculate_dice_per_class(pred, target, threshold=0.5):
    """
    Calculate Dice score for each class separately.

    Returns:
        dict with keys: 'mean', 'background', 'lv' (left ventricle),
                       'myo' (myocardium), 'la' (left atrium)
    """
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    pred_flat = pred_binary.reshape(pred_binary.size(0), pred_binary.size(1), -1)
    target_flat = target.reshape(target.size(0), target.size(1), -1)

    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

    dice_per_class = (2.0 * intersection + 1e-7) / (union + 1e-7)  # (B, C)
    dice_per_class = dice_per_class.mean(dim=0)   

    class_names = ['background', 'lv', 'myo', 'la']
    result = {'mean': dice_per_class.mean().item()}
    for i, name in enumerate(class_names):
        if i < dice_per_class.size(0):
            result[name] = dice_per_class[i].item()

    return result


def apply_tta(model, images):
    """
    Apply Augmentation (TTA) using flips.
    Averages predictions over: original, hflip, vflip, hflip+vflip
    """
    # Original prediction
    _, _, out9 = model(images)
    pred = torch.sigmoid(out9)

    # Horizontal flip
    images_hflip = torch.flip(images, dims=[3])
    _, _, out9_hflip = model(images_hflip)
    pred_hflip = torch.sigmoid(torch.flip(out9_hflip, dims=[3]))

    # Vertical flip
    images_vflip = torch.flip(images, dims=[2])
    _, _, out9_vflip = model(images_vflip)
    pred_vflip = torch.sigmoid(torch.flip(out9_vflip, dims=[2]))

    # Both flips
    images_hvflip = torch.flip(images, dims=[2, 3])
    _, _, out9_hvflip = model(images_hvflip)
    pred_hvflip = torch.sigmoid(torch.flip(out9_hvflip, dims=[2, 3]))

    # Average predictions
    pred_avg = (pred + pred_hflip + pred_vflip + pred_hvflip) / 4.0

    return pred_avg


def train_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=1, scheduler=None):
    model.train()
    total_loss = 0
    total_dice = 0
    class_dice_sums = {'background': 0, 'lv': 0, 'myo': 0, 'la': 0}

    optimizer.zero_grad()

    for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Step optimizer every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            dice = calculate_dice(outputs[2], masks)
            dice_per_class = calculate_dice_per_class(outputs[2], masks)

        total_loss += loss.item() * accumulation_steps  
        total_dice += dice
        for key in class_dice_sums:
            class_dice_sums[key] += dice_per_class.get(key, 0)

    # Handle remaining gradients if batch count not divisible by accumulation_steps
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    n = len(dataloader)
    class_dice_avg = {k: v / n for k, v in class_dice_sums.items()}
    return total_loss / n, total_dice / n, class_dice_avg


def validate_epoch(model, dataloader, criterion, device, use_tta=False):
    model.eval()
    total_loss = 0
    total_dice = 0
    class_dice_sums = {'background': 0, 'lv': 0, 'myo': 0, 'la': 0}

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            if use_tta:
                # Use TTA for dice calculation
                pred_avg = apply_tta(model, images)
                pred_binary = (pred_avg > 0.5).float()

                # Calculate dice from averaged predictions
                pred_flat = pred_binary.reshape(pred_binary.size(0), pred_binary.size(1), -1)
                target_flat = masks.reshape(masks.size(0), masks.size(1), -1)
                intersection = (pred_flat * target_flat).sum(dim=2)
                union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
                dice_scores = (2.0 * intersection + 1e-7) / (union + 1e-7)
                dice = dice_scores.mean().item()

                # Per-class dice
                dice_per_class_vals = dice_scores.mean(dim=0)
                class_names = ['background', 'lv', 'myo', 'la']
                dice_per_class = {'mean': dice_per_class_vals.mean().item()}
                for i, name in enumerate(class_names):
                    if i < dice_per_class_vals.size(0):
                        dice_per_class[name] = dice_per_class_vals[i].item()
            else:
                dice = calculate_dice(outputs[2], masks)
                dice_per_class = calculate_dice_per_class(outputs[2], masks)

            total_loss += loss.item()
            total_dice += dice
            for key in class_dice_sums:
                class_dice_sums[key] += dice_per_class.get(key, 0)

    n = len(dataloader)
    class_dice_avg = {k: v / n for k, v in class_dice_sums.items()}
    return total_loss / n, total_dice / n, class_dice_avg


def save_checkpoint(model, optimizer, scheduler, epoch, val_dice, save_path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_dice": val_dice,
    }
    torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../DATA")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (default: 2, as per winning model)")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps for effective larger batch")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max epochs (default: 200, early stopping will stop earlier)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--split_type", type=str, default="random", choices=["random", "official"],
                        help="'random' for 70/15/15 split, 'official' for CAMUS official split")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau"],
                        help="LR scheduler: 'cosine' or 'plateau' ")
    parser.add_argument("--use_tta", action="store_true",
                        help="Use Test-Time Augmentation (TTA) final test evaluation only")
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=False,
                        help="")
    parser.add_argument("--patience", type=int, default=20,
                        help="")
    parser.add_argument("--t0", type=int, default=10,
                        help="T_0 for Cosine (restart period in epochs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="path for resume-training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Loading dataset...")
    images, masks, metadata = load_camus_dataset(
        data_root=args.data_path,
        target_size=(224, 224),
    )
    images = normalize_images(images, method="minmax")

    print(f"Using {args.split_type} split...")
    dataloaders = create_dataloaders(
        images, masks, metadata,
        batch_size=args.batch_size,
        split_type=args.split_type,
        data_root=args.data_path
    )

    print(f"Train: {len(dataloaders['train'].dataset)} samples")
    print(f"Val: {len(dataloaders['val'].dataset)} samples")
    print(f"Test: {len(dataloaders['test'].dataset)} samples")

    model = FCT(num_classes=args.num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    base_loss = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    criterion = DeepSupervisionLoss(base_loss)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "cosine":
        # T_0 = restart period, T_mult = 1 means fixed period
        # Steps per epoch calculation for per-iteration stepping
        steps_per_epoch = len(dataloaders['train']) // args.accumulation_steps
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.t0 * steps_per_epoch,  # Convert epochs to steps
            T_mult=1,
            eta_min=1e-6
        )
        scheduler_step_per_iter = True
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        scheduler_step_per_iter = False

    early_stopping = EarlyStopping(patience=args.patience) if args.early_stopping else None

    start_epoch = 1
    best_dice = 0.0
    history = {
        "train_loss": [], "train_dice": [], "val_loss": [], "val_dice": [],
        "train_dice_background": [], "train_dice_lv": [], "train_dice_myo": [], "train_dice_la": [],
        "val_dice_background": [], "val_dice_lv": [], "val_dice_myo": [], "val_dice_la": [],
        "lr": []
    }

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("val_dice", 0.0)
        print(f"Resumed from epoch {checkpoint['epoch']}, best dice: {best_dice:.4f}")

        resume_dir = Path(args.resume).parent
        history_path = resume_dir / "history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                history = json.load(f)
            print(f"Loaded training history ({len(history['train_loss'])} epochs)")

    print(f"  Scheduler: {args.scheduler}")
    print(f"  TTA: {args.use_tta}")
    print(f"  Early stopping: {args.early_stopping} (patience={args.patience})")
    print(f"  Gradient accumulation: {args.accumulation_steps}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_dice, train_dice_per_class = train_epoch(
            model, dataloaders["train"], criterion, optimizer, device,
            accumulation_steps=args.accumulation_steps,
            scheduler=scheduler if scheduler_step_per_iter else None
        )

        val_loss, val_dice, val_dice_per_class = validate_epoch(
            model, dataloaders["val"], criterion, device, use_tta=False
        )

        if not scheduler_step_per_iter:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["lr"].append(current_lr)

        for key in ['background', 'lv', 'myo', 'la']:
            history[f"train_dice_{key}"].append(train_dice_per_class[key])
            history[f"val_dice_{key}"].append(val_dice_per_class[key])

        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"  Per-class: BG={train_dice_per_class['background']:.4f} LV={train_dice_per_class['lv']:.4f} "
              f"MYO={train_dice_per_class['myo']:.4f} LA={train_dice_per_class['la']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}" + (" (TTA)" if args.use_tta else ""))
        print(f"  Per-class: BG={val_dice_per_class['background']:.4f} LV={val_dice_per_class['lv']:.4f} "
              f"MYO={val_dice_per_class['myo']:.4f} LA={val_dice_per_class['la']:.4f}")
        print(f"LR: {current_lr:.6f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, scheduler, epoch, val_dice, save_dir / "best_model.pth")
            print(f"Saved best model (Dice: {val_dice:.4f})")

        # Save every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_dice, save_dir / f"epoch_{epoch}.pth")
            print(f"Saved checkpoint at epoch {epoch}")

        if early_stopping is not None:
            if early_stopping(val_dice):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"No improvement for {args.patience} epochs")
                break

    save_checkpoint(model, optimizer, scheduler, epoch, val_dice, save_dir / "final_model.pth")

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best validation Dice: {best_dice:.4f}")

    print("Final Test Evaluation")

    # Load best model for test evaluation
    best_checkpoint = torch.load(save_dir / "best_model.pth", weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # Test without TTA
    _, test_dice, test_dice_per_class = validate_epoch(
        model, dataloaders["test"], criterion, device, use_tta=False
    )
    print(f"\nTest (no TTA): Dice={test_dice:.4f}")
    print(f"  Per-class: BG={test_dice_per_class['background']:.4f} LV={test_dice_per_class['lv']:.4f} "
          f"MYO={test_dice_per_class['myo']:.4f} LA={test_dice_per_class['la']:.4f}")

    # Test with TTA as used in the CAMUS leaderboard
    if args.use_tta:
        _, test_dice_tta, test_dice_per_class_tta = validate_epoch(
            model, dataloaders["test"], criterion, device, use_tta=True
        )
        print(f"\nTest (with TTA): Dice={test_dice_tta:.4f}")
        print(f"  Per-class: BG={test_dice_per_class_tta['background']:.4f} LV={test_dice_per_class_tta['lv']:.4f} "
              f"MYO={test_dice_per_class_tta['myo']:.4f} LA={test_dice_per_class_tta['la']:.4f}")

        test_results = {
            "test_dice": test_dice,
            "test_dice_per_class": test_dice_per_class,
            "test_dice_tta": test_dice_tta,
            "test_dice_per_class_tta": test_dice_per_class_tta
        }
    else:
        test_results = {
            "test_dice": test_dice,
            "test_dice_per_class": test_dice_per_class
        }

    with open(save_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
