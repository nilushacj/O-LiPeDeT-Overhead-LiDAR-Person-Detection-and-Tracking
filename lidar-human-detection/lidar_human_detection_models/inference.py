import argparse, json, yaml, sys, os
from pathlib import Path
import numpy as np
from easydict import EasyDict
import time
# ensure OpenPCDet is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor/OpenPCDet"))

import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from lidar_human_detection_models.utils.dataset_openpcd import MyDataset
from statistics import median

def wrap_deg_0_360(deg: float) -> float:
    # map to [0, 360)
    return float(deg % 360.0)

def save_dets_txt(
    txt_path: Path,
    boxes_xyz_dxdydz_yaw: np.ndarray,
    scores: np.ndarray | None = None,
    class_name: str = "box",
):
    """
    Save detections to a text file in labelCloud centroid_abs-like format:
      box x y z dx dy dz yaw_deg score
    """
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    if scores is None:
        scores = np.ones((boxes_xyz_dxdydz_yaw.shape[0],), dtype=np.float32)
    else:
        scores = np.asarray(scores).reshape(-1)

    assert boxes_xyz_dxdydz_yaw.shape[0] == scores.shape[0], \
        f"Mismatch: {boxes_xyz_dxdydz_yaw.shape[0]} boxes vs {scores.shape[0]} scores"

    with open(txt_path, "w") as f:
        for b, s in zip(boxes_xyz_dxdydz_yaw, scores):
            x, y, z, dx, dy, dz, yaw = [float(v) for v in b[:7]]
            yaw_deg = wrap_deg_0_360(np.degrees(yaw))
            f.write(
                f"{class_name} "
                f"{x:.8f} {y:.8f} {z:.8f} "
                f"{dx:.4f} {dy:.4f} {dz:.4f} "
                f"{yaw_deg:.0f} "
                f"{float(s):.6f}\n"
            )

def parse_config():
    #NOTE: The script will always read "test/pointclouds"
    p = argparse.ArgumentParser(description="Export OpenPCDet predictions with lightweight test-time augmentation (TTA)")
    p.add_argument('--cfg_file', type=str, required=True, help='Model config .yaml for OpenPCDet')
    p.add_argument('--dataset_cfg_file', type=str, required=True, help='Dataset config .yaml')
    p.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    p.add_argument('--out_dir', type=str, default='outputs/preds_voxelnext', help='Root output dir (runs will be subfolders)')
    p.add_argument('--conf_thresh', type=json.loads, default='{"1": 0.45}', help='Confidence thresholds per class (JSON)')
    p.add_argument('--topk', type=int, default=5, help='Top-K boxes per frame after thresholding')
    p.add_argument('--num_runs', type=int, default=1, help='Number of TTA runs') # Keep as 1 for no augmentation

    # ---- Simple, robust TTA knobs (recommended defaults) ----
    p.add_argument('--rot_deg', type=float, default=10.0, help='Global yaw (deg) ~ U(-rot_deg, +rot_deg)')
    p.add_argument('--trans_std', type=float, default=0.05, help='Global XY translation std (m)')
    p.add_argument('--z_shift_std', type=float, default=0.03, help='Global Z shift std (m)')
    p.add_argument('--scale_std', type=float, default=0.0075, help='Global isotropic scale std (~0.5–1.0%)')
    p.add_argument('--xy_noise_std', type=float, default=0.005, help='Per-point XY jitter std (m)')
    p.add_argument('--z_noise_std', type=float, default=0.01, help='Per-point Z jitter std (m)')
    p.add_argument('--drop_p', type=float, default=0.0, help='Random per-point dropout probability (0–0.15 recommended)')

    p.add_argument('--seed', type=int, default=0, help='Base RNG seed (each run offsets this)')
    return p.parse_args()


# ---------- TTA helpers ----------
def sample_tta(args, rng):
    # global isotropic scale ~ N(1, std), clipped to a small band
    s = float(np.clip(rng.normal(1.0, args.scale_std), 1.0 - 3*args.scale_std, 1.0 + 3*args.scale_std))
    s = float(np.clip(s, 0.98, 1.02))

    # global yaw in radians
    theta_deg = rng.uniform(-args.rot_deg, args.rot_deg)
    theta = float(np.deg2rad(theta_deg))

    # global translation
    tx = float(rng.normal(0.0, args.trans_std))
    ty = float(rng.normal(0.0, args.trans_std))
    tz = float(rng.normal(0.0, args.z_shift_std))

    return dict(scale=s, theta=theta, tx=tx, ty=ty, tz=tz)


def apply_tta_points_inplace(points_np, tta, args, rng):
    """
    points_np: collated numpy array shaped [N, 5] = [batch_idx, x, y, z, intensity]
    Applies: global scale -> yaw rotZ -> global translation -> per-point jitter -> dropout
    """
    if points_np.size == 0:
        return

    # split columns
    xyz = points_np[:, 1:4]  # operates in-place later

    # global scale
    xyz *= tta['scale']

    # global rotation (Z)
    c, s = np.cos(tta['theta']), np.sin(tta['theta'])
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    xyz[:, 0] = c * x - s * y
    xyz[:, 1] = s * x + c * y

    # global translation
    xyz[:, 0] += tta['tx']
    xyz[:, 1] += tta['ty']
    xyz[:, 2] += tta['tz']

    # per-point jitter
    if args.xy_noise_std > 0:
        xyz[:, 0] += rng.normal(0.0, args.xy_noise_std, size=xyz.shape[0])
        xyz[:, 1] += rng.normal(0.0, args.xy_noise_std, size=xyz.shape[0])
    if args.z_noise_std > 0:
        xyz[:, 2] += rng.normal(0.0, args.z_noise_std, size=xyz.shape[0])

    # dropout
    if 0.0 < args.drop_p < 1.0:
        keep = rng.random(size=points_np.shape[0]) > args.drop_p
        # avoid dropping *everything*
        if not np.any(keep):
            keep[rng.integers(0, points_np.shape[0])] = True
        # in-place shrink by returning a view subset (caller must assign)
        return points_np[keep]
    return points_np


def inverse_map_boxes_world(boxes, tta):
    """
    boxes: [N,7] = [x,y,z, dx,dy,dz, yaw] in AUGMENTED frame.
    Return boxes back-mapped to ORIGINAL frame by inverting (T * R * S).
    """
    if boxes.size == 0:
        return boxes

    out = boxes.copy()

    # invert translation
    out[:, 0] -= tta['tx']
    out[:, 1] -= tta['ty']
    out[:, 2] -= tta['tz']

    # invert rotation (R^-1 = R^T)
    c, s = np.cos(-tta['theta']), np.sin(-tta['theta'])
    x = out[:, 0].copy()
    y = out[:, 1].copy()
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y

    # invert scale for centers and sizes
    out[:, 0:3] /= tta['scale']
    out[:, 3:6] /= tta['scale']

    # invert yaw (only Z-rotation assumed)
    out[:, 6] -= tta['theta']

    return out

def count_points_in_box(points_xyz, box):
    """
    Count how many points lie inside a 3D bounding box.
    Approximate using axis-aligned bounds after rotating into box frame.

    box: [x, y, z, dx, dy, dz, yaw]
    points_xyz: (N,3)
    """
    x, y, z, dx, dy, dz, yaw = box.astype(float)
    half = np.array([dx/2, dy/2, dz/2], dtype=np.float32)

    # Move points into box coordinate frame
    pts = points_xyz - np.array([x, y, z], dtype=np.float32)

    # Rotate by -yaw
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    pts_xy = pts[:, :2] @ R.T

    # Check inside bounds (AABB in box frame)
    inside_xy = (np.abs(pts_xy[:, 0]) <= half[0]) & (np.abs(pts_xy[:, 1]) <= half[1])
    inside_z  = (pts[:, 2] >= -half[2]) & (pts[:, 2] <= half[2])

    inside = inside_xy & inside_z
    return int(inside.sum())


# ---------------------------------
def main():
    args = parse_config()
    root_out = Path(args.out_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    # Load configs
    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset_cfg = EasyDict(yaml.safe_load(open(args.dataset_cfg_file, "r")))
    cfg.DATA_CONFIG = dataset_cfg.DATA_CONFIG

    # Logger
    logger = common_utils.create_logger()
    logger.info(f"Export with TTA to: {root_out}")
    logger.info(f"TTA params: rot_deg={args.rot_deg}, trans_std={args.trans_std}, "
                f"z_shift_std={args.z_shift_std}, scale_std={args.scale_std}, "
                f"xy_noise_std={args.xy_noise_std}, z_noise_std={args.z_noise_std}, drop_p={args.drop_p}")

    # Dataset (test split)
    dataset = MyDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=None,
        logger=logger
    )
    logger.info(f"Number of test samples: {len(dataset)}")

    # Model
    model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # Run multiple TTA passes
    base_seed = int(args.seed)
    for run_idx in range(args.num_runs):
        rng = np.random.default_rng(base_seed + run_idx)
        no_aug = (run_idx == 0)  # <- first run is raw (no augmentation)

        run_dir = root_out / f"run_{run_idx:02d}"
        pred_dir = run_dir / "preds"
        pred_dir.mkdir(parents=True, exist_ok=True)

        index = {
            "cfg_file": args.cfg_file,
            "dataset_cfg_file": args.dataset_cfg_file,
            "ckpt": args.ckpt,
            "split": "test",
            "conf_thresh_used": args.conf_thresh,
            "topk_used": args.topk,
            "tta": {
                "rot_deg": args.rot_deg,
                "trans_std": args.trans_std,
                "z_shift_std": args.z_shift_std,
                "scale_std": args.scale_std,
                "xy_noise_std": args.xy_noise_std,
                "z_noise_std": args.z_noise_std,
                "drop_p": args.drop_p,
                "seed": base_seed + run_idx,
                "no_aug": no_aug,
            },
            "frames": []
        }

        logger.info(f"\n=== TTA run {run_idx+1}/{args.num_runs} -> {run_dir} ===")
        tot_times = []
        for idx in range(len(dataset)):
            # prepare batch (numpy on CPU)
            single = dataset[idx]
            data_dict = dataset.collate_batch([single])

            pts = data_dict.get('points', None)
            if pts is None or pts.size == 0:
                logger.warning(f"[{idx}] Empty points array? Skipping frame.")
                continue
            
            # --- apply TTA only for augmented runs ---
            if not no_aug:
                tta = sample_tta(args, rng)
                new_pts = apply_tta_points_inplace(pts, tta, args, rng)
                if new_pts is not None and new_pts is not pts:
                    data_dict['points'] = new_pts
            else:
                tta = dict(scale=1.0, theta=0.0, tx=0.0, ty=0.0, tz=0.0)
            
            # -------------------------------------
            # IMPORTANT: UNCOMMENT FOR NUSCENES when running pretrained (not finetuned) model (ensures nuScenes-style timestamp exists (x,y,z,intensity,timestamp))
            # pts = data_dict['points']
            # if pts.shape[1] == 5:
            #     ts = np.zeros((pts.shape[0], 1), dtype=pts.dtype)
            #     data_dict['points'] = np.concatenate([pts, ts], axis=1)
            # -------------------------------------

            load_data_to_gpu(data_dict)

            # forward
            with torch.no_grad():
                start_time = time.time()
                pred_dicts, _ = model.forward(data_dict)
                tot_times.append(time.time()-start_time)
            points_xyz = data_dict['points'][:, 1:4].cpu().numpy()

            pred_boxes  = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].detach().cpu().numpy()
            
            # map predictions back to ORIGINAL frame
            pred_boxes = inverse_map_boxes_world(pred_boxes, tta)

            # confidence filtering
            if pred_labels.size > 0:
                keep = np.zeros_like(pred_labels, dtype=bool)
                for class_id, thr in args.conf_thresh.items():
                    keep |= (pred_labels == int(class_id)) & (pred_scores >= float(thr))
                pred_boxes  = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

            # top-k
            if pred_scores.size > 0:
                order = pred_scores.argsort()[::-1][:args.topk]
                pred_boxes  = pred_boxes[order]
                pred_scores = pred_scores[order]
                pred_labels = pred_labels[order]

            # ---- minimum point count filtering ----
            MIN_POINTS = 0  # change as desired

            if pred_boxes.size > 0:
                keep2 = []
                for i, box in enumerate(pred_boxes):
                    npts = count_points_in_box(points_xyz, box)
                    if npts >= MIN_POINTS:
                        keep2.append(i)

                pred_boxes  = pred_boxes[keep2]
                pred_scores = pred_scores[keep2]
                pred_labels = pred_labels[keep2]

            # Save TXT (GT-like)
            frame_name = dataset.frames[idx] 

            txt_dir = run_dir / "preds"
            txt_path = txt_dir / f"{frame_name}.txt"

            save_dets_txt(txt_path, pred_boxes, scores=pred_scores, class_name="box")

            index["frames"].append(frame_name)

            if idx % 1 == 0:
                logger.info(f"[run {run_idx:02d}] [{idx+1}/{len(dataset)}] {frame_name} "
                            f"-> {pred_boxes.shape[0]} boxes")

        # Write index.json for this run
        with open(run_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)

        avg_time = sum(tot_times) / len(tot_times)
        p50, p90, p99 = np.percentile(tot_times, [50, 90, 99])
        logger.info(f"p50={p50:.6f}s  p90={p90:.6f}s  p99={p99:.6f}s")
        logger.info(f"\nAverage inference time: {avg_time}")

    logger.info("\nAll runs complete.")


if __name__ == '__main__':
    main()

"""
Usage: 

python lidar_human_detection_models/inference.py \
  --cfg_file ./cfgs/nuscenes_voxelnext.yaml \
  --dataset_cfg_file ./cfgs/voxelnext_crane.yaml \
  --ckpt ./fine_tuned_ckpts/finetuned_voxelnext_250.pth \
  --out_dir outputs/preds_voxelnext \
  --conf_thresh '{"1": 0.45}' \
  --topk 5

"""