"""
Compares the dice value to the leaderboard's dice value on the CAMUS dataset (2019)

CAMUS Leaderboard evaluates:
- LV-endocardium (inner wall of left ventricle)
- LV-epicardium (outer wall) 
- Left Atrium 

Results are split by cardiac phase:
- ED (End-Diastole)
- ES (End-Systole)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import load_camus_dataset, normalize_images
from model import FCT


# CAMUS Leaderboard reference (top 5)
LEADERBOARD = {
    "LV_endocardium": {
        "ED": [
            ("NN-Unet", 0.952),
            ("CLAS", 0.947),
            ("GUDU", 0.946),
            ("U-Net", 0.936),
            ("ACNN", 0.936),
        ],
        "ES": [
            ("NN-Unet", 0.935),
            ("CLAS", 0.929),
            ("GUDU", 0.929),
            ("ACNN", 0.913),
            ("U-Net", 0.912),
        ],
        "intra_observer": {"ED": 0.945, "ES": 0.930},
    },
    "LV_epicardium": {
        "ED": [
            ("NN-Unet", 0.963),
            ("CLAS", 0.961),
            ("GUDU", 0.960),
            ("U-Net", 0.956),
            ("ACNN", 0.953),
        ],
        "ES": [
            ("NN-Unet", 0.959),
            ("CLAS", 0.955),
            ("GUDU", 0.955),
            ("U-Net", 0.946),
            ("ACNN", 0.945),
        ],
        "intra_observer": {"ED": 0.957, "ES": 0.951},
    },
    "LA": {
        "ED": [
            ("NN-Unet", 0.902),
            ("CLAS", 0.902),
            ("GUDU", 0.894),
            ("U-Net", 0.889),
            ("ACNN", 0.881),
        ],
        "ES": [
            ("NN-Unet", 0.935),
            ("CLAS", 0.927),
            ("GUDU", 0.926),
            ("U-Net", 0.918),
            ("ACNN", 0.911),
        ],
    },
}


def calculate_dice(pred_binary, target_binary):
    """Calculate Dice score between binary masks."""
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection / (union + 1e-7)).item()



def evaluate_camus_metrics(model, images, masks, metadata, device):
    """
    Evaluate model using CAMUS leaderboard metrics.

    Returns Dice scores for:
    LV-endocardium (class 1) split by ED/ES
    LV-epicardium (class 1 + class 2) split by ED/ES
    LA (class 3) split by ED/ES
    """
    model.eval()

    results = {
        "LV_endocardium": {"ED": [], "ES": []},
        "LV_epicardium": {"ED": [], "ES": []},
        "LA": {"ED": [], "ES": []},
    }

    with torch.no_grad():
        for i in tqdm(range(len(images)), desc="Evaluating"):
            image = torch.FloatTensor(images[i:i+1]).to(device)
            mask = masks[i]  
            phase = metadata[i]["phase"]  # "ED" or "ES"

            # Get prediction
            _, _, out9 = model(image)
            pred = torch.sigmoid(out9)

            pred_binary = (pred > 0.5).float().cpu().numpy()[0]  

            # Ground truth classes
            gt_lv = mask[1]        # Class 1: LV (endocardium)
            gt_myo = mask[2]       # Class 2: Myocardium
            gt_la = mask[3]        # Class 3: LA

            pred_lv = pred_binary[1]
            pred_myo = pred_binary[2]
            pred_la = pred_binary[3]

            # LV-endocardium: just the LV cavity (class 1)
            dice_endo = calculate_dice(pred_lv, gt_lv)
            results["LV_endocardium"][phase].append(dice_endo)

            # LV-epicardium: LV + Myocardium combined (outer boundary)
            gt_epi = np.clip(gt_lv + gt_myo, 0, 1)
            pred_epi = np.clip(pred_lv + pred_myo, 0, 1)
            dice_epi = calculate_dice(pred_epi, gt_epi)
            results["LV_epicardium"][phase].append(dice_epi)

            # Left Atrium (class 3)
            dice_la = calculate_dice(pred_la, gt_la)
            results["LA"][phase].append(dice_la)

    summary = {}
    for structure in results:
        summary[structure] = {}
        for phase in ["ED", "ES"]:
            scores = results[structure][phase]
            if scores:
                summary[structure][phase] = np.mean(scores)
            else:
                summary[structure][phase] = 0.0

    return summary


def get_rank(score, leaderboard_entries):
    """Determine rank based on score compared to leaderboard."""
    for i, (name, lb_score) in enumerate(leaderboard_entries):
        if score >= lb_score:
            return i + 1
    return len(leaderboard_entries) + 1


def print_comparison(results):
    """print comparisons"""
    print(f"\n  {'Structure':<20} {'ED Dice':<10} {'Rank':<8} {'ES Dice':<10} {'Rank':<8}")
    print(f"  {'-'*56}")

    overall_scores = []
    for structure in ["LV_endocardium", "LV_epicardium", "LA"]:
        ed = results[structure]["ED"]
        es = results[structure]["ES"]
        ed_rank = get_rank(ed, LEADERBOARD[structure]["ED"])
        es_rank = get_rank(es, LEADERBOARD[structure]["ES"])
        overall_scores.extend([ed, es])
        print(f"  {structure:<20} {ed:.4f}    #{ed_rank:<6} {es:.4f}    #{es_rank:<6}")

    print(f"  {'-'*56}")
    print(f"  {'OVERALL MEAN':<20} {np.mean(overall_scores):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare model to CAMUS leaderboard")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. If not provided, uses latest best_model.pth")
    parser.add_argument("--data_path", type=str, default="./DATA",
                        help="Path to CAMUS dataset")
    parser.add_argument("--save_results", action="store_true",
                        help="Save results to JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoints_dir = Path("./checkpoints")
        if not checkpoints_dir.exists():
            print("No checkpoints directory found")
            return
        # Find most recent checkpoint directory
        latest_dir = max(checkpoints_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        checkpoint_path = latest_dir / "best_model.pth"

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load model
    model = FCT(num_classes=4).to(device)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test data using official split
    print(f"Loading CAMUS dataset from {args.data_path}...")
    images, masks, metadata = load_camus_dataset(
        data_root=args.data_path,
        target_size=(224, 224),
    )
    images = normalize_images(images, method="minmax")

    # Load official test split
    data_root = Path(args.data_path)
    test_file = data_root / "subgroup_testing.txt"
    if test_file.exists():
        with open(test_file, 'r') as f:
            test_patients = set(line.strip() for line in f if line.strip())

        # Filter to test set only
        test_indices = [i for i, m in enumerate(metadata) if m["patient_id"] in test_patients]
        images = images[test_indices]
        masks = masks[test_indices]
        metadata = [metadata[i] for i in test_indices]
        print(f"Using official test split: {len(test_indices)} samples from {len(test_patients)} patients")
    else:
        print("no split found")


    results = evaluate_camus_metrics(model, images, masks, metadata, device)

    print_comparison(results)

    # Save results
    if args.save_results:
        output_path = checkpoint_path.parent / "leaderboard_comparison.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
