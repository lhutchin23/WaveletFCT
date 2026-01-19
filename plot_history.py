import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


# path to most recent checkpoint
checkpoints_dir = Path("./checkpoints")
if not checkpoints_dir.exists():
    print("No checkpoints directory found")
    exit(1)

latest_checkpoint = max(checkpoints_dir.iterdir(), key=lambda p: p.stat().st_mtime)
history_path = latest_checkpoint / "history.json"

with open(history_path, 'r') as f:
    history = json.load(f)

epochs = list(range(1, len(history['train_loss']) + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# loss plot
axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# dice plot
axes[1].plot(epochs, history['train_dice'], label='Train Dice', marker='o', markersize=3)
axes[1].plot(epochs, history['val_dice'], label='Val Dice', marker='s', markersize=3)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Dice Score')
axes[1].set_title('Training and Validation Dice Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

checkpoint_dir = Path(history_path).parent
output_path = checkpoint_dir / 'training_history.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

best_val_dice_idx = history['val_dice'].index(max(history['val_dice']))
print(f"Best validation Dice: {max(history['val_dice']):.4f} at epoch {best_val_dice_idx + 1}")
print(f"Final train loss: {history['train_loss'][-1]:.4f}")
print(f"Final val loss: {history['val_loss'][-1]:.4f}")
print(f"Final train Dice: {history['train_dice'][-1]:.4f}")
print(f"Final val Dice: {history['val_dice'][-1]:.4f}")

plt.show()

