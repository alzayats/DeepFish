#!/usr/bin/env python
"""
Simple prediction script for running inference on arbitrary images.

This script allows you to run trained DeepFish models on any folder of images
without requiring the full DeepFish dataset structure (CSV files, masks, etc.).

Usage:
    python scripts/predict_simple.py -i /path/to/images -m model.pth -t loc -o output/

Arguments:
    -i, --image_dir: Directory containing images to process
    -m, --model_path: Path to trained model checkpoint (.pth file)
    -t, --task: Task type (loc, seg, clf, reg)
    -o, --output_dir: Directory for output visualizations and JSON results

Output:
    output/
        predictions.json     # All predictions in JSON format
        visualizations/      # Visualization images (for loc and seg tasks)
            image1.png
            image2.png
"""

import sys
import os
import json
import argparse

# Add project root to path
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

import torch
from torch.backends import cudnn
from tqdm.auto import tqdm
import numpy as np

from src import datasets, wrappers
from haven import haven_utils as hu

cudnn.benchmark = True

# Task configuration mapping
TASK_CONFIG = {
    "loc": {
        "wrapper": "loc_wrapper",
        "transform": "rgb_normalize",
    },
    "seg": {
        "wrapper": "seg_wrapper",
        "transform": "rgb_normalize",
    },
    "clf": {
        "wrapper": "clf_wrapper",
        "transform": "resize_normalize",
    },
    "reg": {
        "wrapper": "reg_wrapper",
        "transform": "resize_normalize",
    },
}


def load_model(model_path, device):
    """Load a trained model from a checkpoint file."""
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model
    model = torch.load(model_path, map_location=device)

    # Handle case where model is wrapped in a wrapper already
    if hasattr(model, 'model'):
        # Model is already a wrapper, extract the inner model
        inner_model = model.model
        return inner_model, model
    else:
        # Model is a raw model
        return model, None


def run_inference(args):
    """Run inference on images in the specified directory."""
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Validate task
    if args.task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {args.task}. Must be one of: {list(TASK_CONFIG.keys())}")

    task_config = TASK_CONFIG[args.task]

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Load dataset
    print(f"Loading images from: {args.image_dir}")
    dataset = datasets.get_dataset(
        dataset_name="image_folder",
        split="inference",
        transform=task_config["transform"],
        datadir=args.image_dir
    )
    print(f"Found {len(dataset)} images")

    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0
    )

    # Load model
    model_raw, existing_wrapper = load_model(args.model_path, device)

    if existing_wrapper is not None:
        # Model was saved as a wrapper
        model = existing_wrapper.to(device)
    else:
        # Create wrapper for raw model
        model_raw = model_raw.to(device)
        model = wrappers.get_wrapper(task_config["wrapper"], model=model_raw, opt=None)
        model = model.to(device)

    model.eval()

    # Run inference
    predictions = {}
    print("Running inference...")

    for i, batch in enumerate(tqdm(data_loader)):
        filename = batch["meta"]["filename"][0]  # Get filename from batch
        image_path = batch["meta"]["image_path"][0]

        # Run task-specific inference
        if args.task == "loc":
            pred = model.predict_on_batch(batch)
            predictions[filename] = {
                "image_path": image_path,
                "count": pred["count"],
                "points": pred["points"]  # List of (y, x) coordinates
            }

            # Save visualization
            vis_path = os.path.join(vis_dir, f"{os.path.splitext(filename)[0]}.png")
            model.vis_on_batch_inference(batch, vis_path)

        elif args.task == "seg":
            pred_mask = model.predict_on_batch(batch)
            # Count number of fish (connected components)
            from skimage.measure import label
            labeled = label(pred_mask.squeeze())
            fish_count = labeled.max()

            predictions[filename] = {
                "image_path": image_path,
                "fish_count": int(fish_count),
                "has_fish": bool(fish_count > 0)
            }

            # Save visualization
            vis_path = os.path.join(vis_dir, f"{os.path.splitext(filename)[0]}.png")
            model.vis_on_batch_inference(batch, vis_path)

        elif args.task == "clf":
            result = model.vis_on_batch_inference(batch, None)
            predictions[filename] = {
                "image_path": image_path,
                "has_fish": bool(result["prediction"] > 0.5),
                "confidence": float(result["prediction"])
            }

        elif args.task == "reg":
            result = model.vis_on_batch_inference(batch, None)
            predictions[filename] = {
                "image_path": image_path,
                "count": int(result["prediction"])
            }

        if args.verbose:
            print(f"  {filename}: {predictions[filename]}")

    # Save predictions to JSON
    output_json = os.path.join(args.output_dir, "predictions.json")
    with open(output_json, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Predictions: {output_json}")
    if args.task in ["loc", "seg"]:
        print(f"  - Visualizations: {vis_dir}/")

    # Print summary
    print(f"\nSummary:")
    print(f"  - Total images processed: {len(predictions)}")

    if args.task == "loc":
        total_fish = sum(p["count"] for p in predictions.values())
        print(f"  - Total fish detected: {total_fish}")
        avg_fish = total_fish / len(predictions) if predictions else 0
        print(f"  - Average fish per image: {avg_fish:.2f}")

    elif args.task == "clf":
        fish_images = sum(1 for p in predictions.values() if p["has_fish"])
        print(f"  - Images with fish: {fish_images}")
        print(f"  - Images without fish: {len(predictions) - fish_images}")

    elif args.task in ["reg", "seg"]:
        if args.task == "reg":
            total_fish = sum(p["count"] for p in predictions.values())
        else:
            total_fish = sum(p["fish_count"] for p in predictions.values())
        print(f"  - Total fish detected: {total_fish}")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on arbitrary images using trained DeepFish models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '-i', '--image_dir',
        type=str,
        required=True,
        help='Directory containing images to process'
    )

    parser.add_argument(
        '-m', '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth file)'
    )

    parser.add_argument(
        '-t', '--task',
        type=str,
        required=True,
        choices=['loc', 'seg', 'clf', 'reg'],
        help='Task type: loc (localization), seg (segmentation), clf (classification), reg (regression/counting)'
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default='./output',
        help='Directory for output visualizations and JSON results (default: ./output)'
    )

    parser.add_argument(
        '--use_cuda',
        type=int,
        default=1,
        help='Use CUDA if available (default: 1)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print prediction for each image'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.image_dir):
        parser.error(f"Image directory does not exist: {args.image_dir}")

    if not os.path.isfile(args.model_path):
        parser.error(f"Model file does not exist: {args.model_path}")

    run_inference(args)


if __name__ == "__main__":
    main()
