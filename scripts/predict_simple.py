#!/usr/bin/env python
"""
Simple prediction script for running inference on arbitrary images or video.

This script allows you to run trained DeepFish models on any folder of images
or on a single video file, without requiring the full DeepFish dataset structure
(CSV files, masks, etc.). It runs on GPU or CPU automatically.

Usage:
    # On a folder of images
    python scripts/predict_simple.py -i /path/to/images -m model.pth -t loc -o output/

    # On a video (frames are extracted automatically)
    python scripts/predict_simple.py --video /path/to/fish.mp4 -m model.pth -t loc -o output/ --frame_stride 15

Arguments:
    -i, --image_dir:   Directory containing images to process
    --video:           Path to a video file (frames are extracted to output/frames/)
    --frame_stride:    Keep every Nth video frame (default: 1)
    -m, --model_path:  Path to trained model checkpoint (.pth file)
    -t, --task:        Task type (loc, seg, clf, reg)
    -o, --output_dir:  Directory for output visualizations and JSON results
    --use_cuda:        Use CUDA if available (default: 1)

Output:
    output/
        predictions.json     # All predictions in JSON format
        frames/              # Extracted video frames (only when --video is used)
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

from DeepFish import datasets, models, wrappers
from haven import haven_utils as hu

cudnn.benchmark = True

# Task configuration mapping. "dataset" and "model" are only used to rebuild the
# architecture when the checkpoint is a state_dict; "model" is the default
# architecture for that task (override with --model_name) and matches exp_configs.
TASK_CONFIG = {
    "loc": {
        "wrapper": "loc_wrapper",
        "transform": "rgb_normalize",
        "dataset": "fish_loc",
        "model": "fcn8",
    },
    "seg": {
        "wrapper": "seg_wrapper",
        "transform": "rgb_normalize",
        "dataset": "fish_seg",
        "model": "fcn8",
    },
    "clf": {
        "wrapper": "clf_wrapper",
        "transform": "resize_normalize",
        "dataset": "fish_clf",
        "model": "inception",
    },
    "reg": {
        "wrapper": "reg_wrapper",
        "transform": "resize_normalize",
        "dataset": "fish_reg",
        "model": "inception",
    },
}


def extract_frames(video_path, frames_dir, stride=1):
    """Extract frames from a video file into frames_dir.

    Args:
        video_path: Path to the input video file.
        frames_dir: Directory where extracted frames are written as .jpg.
        stride: Keep every Nth frame (1 = keep all frames).

    Returns:
        The number of frames written.
    """
    import cv2

    if stride < 1:
        raise ValueError("--frame_stride must be >= 1")

    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_digits = max(6, len(str(total)))

    print(f"Extracting frames from {video_path} (every {stride} frame(s))...")
    frame_idx = 0
    saved = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                out_path = os.path.join(frames_dir, f"frame_{frame_idx:0{n_digits}d}.jpg")
                cv2.imwrite(out_path, frame)
                saved += 1
            frame_idx += 1
    finally:
        cap.release()

    print(f"Extracted {saved} frame(s) to {frames_dir}")
    return saved


def load_model(model_path, task_config, task, device, model_name=None):
    """Load a trained model and return a task wrapper ready for inference.

    Handles the three checkpoint formats this codebase can produce:
      1. A state_dict (what trainval.py saves via ``model.state_dict()``).
         The architecture is rebuilt and the weights are loaded into a wrapper.
      2. A full wrapper object (e.g. saved via ``torch.save(model)``).
      3. A raw model object, which is then wrapped.
    """
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        # Case 1: state_dict of the wrapper -> rebuild architecture and load weights.
        arch = model_name or task_config["model"]
        print(f"Checkpoint is a state_dict; rebuilding '{arch}' architecture for task '{task}'")
        exp_dict = {"dataset": task_config["dataset"], "model": arch}
        base_model = models.get_model(arch, exp_dict=exp_dict)
        model = wrappers.get_wrapper(task_config["wrapper"], model=base_model, opt=None)
        model.load_state_dict(checkpoint)
    elif hasattr(checkpoint, 'model'):
        # Case 2: already a wrapper.
        model = checkpoint
    else:
        # Case 3: a raw model -> wrap it.
        model = wrappers.get_wrapper(task_config["wrapper"], model=checkpoint, opt=None)

    return model.to(device)


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

    # Load model (handles state_dict, full wrapper, or raw model checkpoints)
    model = load_model(
        args.model_path,
        task_config,
        args.task,
        device,
        model_name=getattr(args, "model_name", None),
    )

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

            # Save visualization (reuse the prediction to avoid a second forward pass)
            vis_path = os.path.join(vis_dir, f"{os.path.splitext(filename)[0]}.png")
            model.vis_on_batch_inference(batch, vis_path, pred=pred)

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

            # Save visualization (reuse the prediction to avoid a second forward pass)
            vis_path = os.path.join(vis_dir, f"{os.path.splitext(filename)[0]}.png")
            model.vis_on_batch_inference(batch, vis_path, pred_mask=pred_mask)

        elif args.task == "clf":
            result = model.vis_on_batch_inference(batch, None)
            # "confidence" is the raw sigmoid probability when available,
            # otherwise fall back to the binary prediction.
            confidence = float(result.get("confidence", result["prediction"]))
            predictions[filename] = {
                "image_path": image_path,
                "has_fish": bool(result["prediction"] > 0.5),
                "confidence": confidence
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

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '-i', '--image_dir',
        type=str,
        default=None,
        help='Directory containing images to process'
    )
    source.add_argument(
        '--video',
        type=str,
        default=None,
        help='Path to a video file; frames are extracted automatically'
    )

    parser.add_argument(
        '--frame_stride',
        type=int,
        default=1,
        help='When using --video, keep every Nth frame (default: 1)'
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
        '--model_name',
        type=str,
        default=None,
        choices=['fcn8', 'fcn8_vgg16', 'unet', 'resnet', 'inception'],
        help='Model architecture, only needed when the checkpoint is a state_dict '
             'and the architecture differs from the task default '
             '(loc/seg: fcn8, clf/reg: inception)'
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

    # Validate model
    if not os.path.isfile(args.model_path):
        parser.error(f"Model file does not exist: {args.model_path}")

    # Resolve the image source. If a video is given, extract its frames first
    # and point the pipeline at the extracted-frames directory.
    if args.video is not None:
        if not os.path.isfile(args.video):
            parser.error(f"Video file does not exist: {args.video}")
        os.makedirs(args.output_dir, exist_ok=True)
        frames_dir = os.path.join(args.output_dir, "frames")
        n_frames = extract_frames(args.video, frames_dir, stride=args.frame_stride)
        if n_frames == 0:
            parser.error(f"No frames could be extracted from video: {args.video}")
        args.image_dir = frames_dir
    else:
        if not os.path.isdir(args.image_dir):
            parser.error(f"Image directory does not exist: {args.image_dir}")

    run_inference(args)


if __name__ == "__main__":
    main()
