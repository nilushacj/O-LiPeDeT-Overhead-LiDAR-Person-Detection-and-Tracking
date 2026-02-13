#!/usr/bin/env python3

import argparse, json, csv
from pathlib import Path
import numpy as np

from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from mot_3d.data_protos import BBox
import yaml
import sys
import os
import time
# ---- I/O helpers ----
def yaw_to_rad_if_needed(yaw_val: float) -> float:
    """
    Heuristic: if abs(yaw) is > ~2*pi, it is probably in degrees -> convert to radians.
    """
    if abs(yaw_val) > 6.5:   # slightly above 2*pi
        return float(np.deg2rad(yaw_val))
    return float(yaw_val)

def wrap_to_pi(yaw):
    return (yaw + np.pi) % (2*np.pi) - np.pi

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_txt_frames(in_dir: Path, start: str = "", end: str = ""):
    files = sorted(in_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {in_dir}")

    if not start and not end:
        return files

    stems = [p.stem for p in files]
    stem_to_idx = {s: i for i, s in enumerate(stems)}

    def resolve_idx(name: str, default_idx: int):
        if not name:
            return default_idx
        return stem_to_idx.get(Path(name).stem, default_idx)

    k0 = resolve_idx(start, 0)
    k1 = resolve_idx(end, len(files) - 1)
    if k0 > k1:
        k0, k1 = k1, k0
    return files[k0:k1 + 1]


def load_boxes_from_txt(txt_path: Path, default_score: float = 1.0):
    """
    Input lines:
      box x y z dx dy dz yaw_deg
    or:
      box x y z dx dy dz yaw_deg score

    Returns:
      boxes: (N,7) float32 with yaw in RADIANS (SimpleTrack expects radians)
      scores: (N,) float32
    """
    boxes, scores = [], []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0].lower() != "box":
                continue
            if len(parts) not in (8, 9):
                raise ValueError(f"Bad line in {txt_path}: '{line}'")

            x, y, z = map(float, parts[1:4])
            dx, dy, dz = map(float, parts[4:7])
            yaw_deg = float(parts[7])
            yaw = np.deg2rad(yaw_deg)

            sc = float(parts[8]) if len(parts) == 9 else float(default_score)
            boxes.append([x, y, z, dx, dy, dz, yaw])
            scores.append(sc)

    if len(boxes) == 0:
        return np.zeros((0, 7), np.float32), np.zeros((0,), np.float32)

    return np.asarray(boxes, np.float32), np.asarray(scores, np.float32)


def save_tracks_txt(out_dir: Path, frame_stem: str, ids, boxes, scores):
    """
    Output lines:
      track id x y z dx dy dz yaw_deg score
    """
    out_path = out_dir / f"{frame_stem}.txt"
    with open(out_path, "w") as f:
        for tid, b, sc in zip(ids, boxes, scores):
            x, y, z, dx, dy, dz, yaw = [float(v) for v in b[:7]]
            yaw = wrap_to_pi(yaw)
            yaw_deg = np.rad2deg(yaw)
            f.write(
                f"track {int(tid)} {x:.8f} {y:.8f} {z:.8f} "
                f"{dx:.8f} {dy:.8f} {dz:.8f} {yaw_deg:.6f} {float(sc):.6f}\n"
            )

# ---- mot_3d adapters ----
def bbox_to_array(bb):
    if hasattr(BBox, "bbox2array"):
        arr = np.asarray(BBox.bbox2array(bb), dtype=np.float32).reshape(-1)

        # Expected SimpleTrack fork format:
        # [x, y, z, heading(yaw), l, w, h, score]
        if arr.size == 8:
            x, y, z, yaw, l, w, h, sc = arr.tolist()
            # Convert to our canonical format:
            # [x, y, z, dx, dy, dz, yaw]
            return np.array([x, y, z, l, w, h, yaw], dtype=np.float32)

        if arr.size == 7:
            x, y, z, yaw, l, w, h = arr.tolist()
            return np.array([x, y, z, l, w, h, yaw], dtype=np.float32)

        return arr[:7]

    def get_any(obj, names, default=0.0):
        for n in names:
            if hasattr(obj, n):
                return float(getattr(obj, n))
        return float(default)

    x = get_any(bb, ["x"])
    y = get_any(bb, ["y"])
    z = get_any(bb, ["z"])
    dx = get_any(bb, ["dx", "l"])
    dy = get_any(bb, ["dy", "w"])
    dz = get_any(bb, ["dz", "h"])
    yaw = get_any(bb, ["yaw", "ry", "heading"])
    return np.array([x, y, z, dx, dy, dz, yaw], dtype=np.float32)

def frame_data_from_dets(seq_id, frame_idx, timestamp, det_boxes, det_scores, obj_type=None):
    # Convert [x,y,z,dx,dy,dz,yaw] -> [x,y,z,o,l,w,h] where o=yaw
    det_arrs = []
    for b7 in det_boxes:
        x, y, z, dx, dy, dz, yaw = [float(v) for v in b7]
        det_arrs.append(np.array([x, y, z, yaw, dx, dy, dz], dtype=np.float32))

    ego = np.eye(4, dtype=np.float32)

    # Normalize common aliases
    t = (obj_type or "pedestrian").lower()
    alias = {
        "box": "pedestrian", 
        "person": "pedestrian",
        "ped": "pedestrian",
        "human": "pedestrian",
    }
    t = alias.get(t, t)

    det_types = [t] * len(det_arrs)

    aux_info = {
        "is_key_frame": True,
        "seq_id": seq_id,
        "frame_id": int(frame_idx),
    }

    fd = FrameData(
        dets=det_arrs,
        ego=ego,
        time_stamp=float(timestamp),
        pc=None,
        det_types=det_types,
        aux_info=aux_info
    )

    # Ensure scores exist (SimpleTrack uses det.s)
    for i, s in enumerate(det_scores):
        if i < len(fd.dets):
            fd.dets[i].s = float(s)

    return fd

def create_tracker(config_path: str, obj_type: str, name: str, det_name: str, score_threshold_override=None):
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    if isinstance(configs, dict):
        configs.setdefault("running", {})
        configs["running"].setdefault("obj_type", obj_type)
        configs["running"].setdefault("name", name)
        configs["running"].setdefault("det_name", det_name)
        configs.setdefault("redundancy", {})
        configs.setdefault("data_loader", {})

        if score_threshold_override is not None:
            configs["running"]["score_threshold"] = float(score_threshold_override)
        configs["redundancy"]["mode"] = "none"  
        configs["data_loader"]["nms"] = False
    try:
        return MOTModel(configs)
    except TypeError:
        return MOTModel(configs=configs)

# ----- main ----
def parse_args():
    ap = argparse.ArgumentParser("Run SimpleTrack on TXT detections")
    ap.add_argument("--in_dir", required=True, type=str,
                    help="Folder containing per-frame *.txt detections/labels")
    ap.add_argument("--out_dir", required=True, type=str,
                    help="Output directory")
    ap.add_argument("--start", default="", type=str, help="Optional start frame stem or filename")
    ap.add_argument("--end", default="", type=str, help="Optional end frame stem or filename")
    ap.add_argument("--config_path", required=True, type=str,
                    help="Path to SimpleTrack YAML config")
    ap.add_argument("--obj_type", default="pedestrian", type=str,
                    help="Object type (e.g., pedestrian/vehicle). We map 'box'->'pedestrian'.")
    ap.add_argument("--name", default="simpletrack", type=str)
    ap.add_argument("--det_name", default="txtsource", type=str)
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--default_score", type=float, default=1.0,
                    help="Used if txt has no score column")
    ap.add_argument("--score_threshold_override", type=float, default=0.0)
    return ap.parse_args()

def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    """
    Our sequences (start - end):
        cloud_stack3_1483300671_200699091.bin.txt - cloud_stack3_1483300675_700703144.bin.txt
        cloud_stack3_1483300723_700703144.bin.txt - cloud_stack3_1483300727_300698042.bin.txt
        cloud_stack3_1483391058_400703192.bin.txt - cloud_stack3_1483391073_400696039.bin.txt
    """

    # Select subrange
    frames = list_txt_frames(in_dir, start="cloud_stack3_1483391058_400703192.bin.txt", end="cloud_stack3_1483391073_400696039.bin.txt")

    tracker = create_tracker(
        config_path=args.config_path,
        obj_type=args.obj_type,
        name=args.name,
        det_name=args.det_name,
        score_threshold_override=args.score_threshold_override
    )

    out_txt_dir = out_root / "tracks_txt"
    ensure_dir(out_txt_dir)

    csv_path = out_root / "tracks.csv"
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

    with open(csv_path, "a", newline="") as fcsv:
        w = csv.writer(fcsv)
        if write_header:
            w.writerow(["frame","id","x","y","z","dx","dy","dz","yaw_deg","score","age","time_since_update","hits","hit_streak"])

        seq_id = "seq00"
        tot_times = []
        for k, txt_path in enumerate(frames):
            det_boxes, det_scores = load_boxes_from_txt(txt_path, default_score=args.default_score)

            timestamp = k / max(args.fps, 1e-6)

            fd = frame_data_from_dets(
                seq_id=seq_id,
                frame_idx=k,
                timestamp=timestamp,
                det_boxes=det_boxes,
                det_scores=det_scores,
                obj_type=args.obj_type
            )
            start_time = time.time()
            out = tracker.frame_mot(fd)
            tot_times.append(time.time()-start_time)

            # ---- Case 0: Nothing returned (empty) ----
            if out is None or (isinstance(out, (list, tuple)) and len(out) == 0):
                pred_bboxes, pred_ids, pred_scores = [], [], None

            # ---- Case 1: Returned as (bboxes, ids, [scores]) ----
            elif isinstance(out, tuple) and len(out) >= 2:
                pred_bboxes = out[0]
                pred_ids = out[1]
                pred_scores = out[2] if len(out) > 2 else None

            # ---- Case 2: Returned as a dict-like ----
            elif isinstance(out, dict):
                pred_bboxes = out.get("pred_bboxes", out.get("bboxes", []))
                pred_ids = out.get("pred_ids", out.get("ids", []))
                pred_scores = out.get("pred_scores", out.get("scores", None))

            # ---- Case 3: Returned as list of track objects/dicts ----
            elif isinstance(out, list):
                pred_bboxes, pred_ids, pred_scores_list = [], [], []

                for item in out:
                    # --- simpleTrack returns tuples: (bbox, id, state, cls) ---
                    if isinstance(item, tuple) and len(item) >= 4:
                        bb = item[0]
                        tid = item[1]
                        state = item[2]
                        clsname = item[3]
                        pred_bboxes.append(bb)
                        pred_ids.append(int(tid))

                        # score is stored in bbox.s
                        sc = getattr(bb, "s", None)
                        pred_scores_list.append(0.0 if sc is None else float(sc))
                        continue

                    # --- Existing support for dict-like results ---
                    if isinstance(item, dict):
                        tid = item.get("id", item.get("track_id", None))
                        bb = item.get("bbox", item.get("box", None))
                        sc = item.get("score", item.get("s", None))
                    else:
                        tid = getattr(item, "id", getattr(item, "track_id", None))
                        bb = getattr(item, "bbox", getattr(item, "box", None))
                        sc = getattr(item, "score", getattr(item, "s", None))

                    if tid is None or bb is None:
                        continue

                    pred_ids.append(int(tid))
                    pred_bboxes.append(bb)
                    pred_scores_list.append(0.0 if sc is None else float(sc))

                pred_scores = pred_scores_list

            # ---- Case 4: Some object with attributes ----
            else:
                pred_bboxes = getattr(out, "pred_bboxes", [])
                pred_ids = getattr(out, "pred_ids", [])
                pred_scores = getattr(out, "pred_scores", None)


            boxes_out = (np.stack([bbox_to_array(bb) for bb in pred_bboxes], axis=0)
                         if len(pred_bboxes) else np.zeros((0,7), np.float32))

            ids_out = (np.asarray(pred_ids, dtype=np.int32).reshape(-1)
                       if len(pred_ids) else np.zeros((0,), np.int32))
            if boxes_out.shape[0] > 0:
                print("Converted box_out[0] [x y z dx dy dz yaw]:", boxes_out[0])

            if pred_scores is None:
                s_list = []
                for bb in pred_bboxes:
                    s = None
                    for sname in ("s","score","confidence"):
                        if hasattr(bb, sname):
                            s = float(getattr(bb, sname))
                            break
                    s_list.append(0.0 if s is None else s)
                scores_out = (np.asarray(s_list, dtype=np.float32)
                              if len(s_list) else np.zeros((0,), np.float32))
            else:
                scores_out = np.asarray(pred_scores, dtype=np.float32).reshape(-1)

            save_tracks_txt(out_txt_dir, txt_path.stem, ids_out, boxes_out, scores_out)

            for tid, b, sc in zip(ids_out.tolist(), boxes_out, scores_out.tolist()):
                x, y, z, dx, dy, dz, yaw = [float(v) for v in b[:7]]
                yaw = wrap_to_pi(yaw)
                yaw_deg = np.rad2deg(yaw)
                w.writerow([txt_path.stem, int(tid), x, y, z, dx, dy, dz, yaw_deg, float(sc)])

            print(f"[{k+1}/{len(frames)}] {txt_path.name} -> tracks_out={len(ids_out)} dets_in={len(det_boxes)}")

    print(f"\nSaved per-frame tracks to: {out_txt_dir}")
    print(f"CSV summary: {csv_path}")
    avg_time = sum(tot_times) / len(tot_times)
    p50, p90, p99 = np.percentile(tot_times, [50, 90, 99])
    print(f"p50={p50:.6f}s  p90={p90:.6f}s  p99={p99:.6f}s")
    print(f"\nAverage inference time: {avg_time}")
    
if __name__ == "__main__":
    main()
"""
Usage:

python3 simpletrack_tracking.py \
  --in_dir /scratch/work/jayawin1/techboost/lidar-human-detection-models/outputs/preds_for_tracks_voxelnext/run_00/preds_txt \
  --out_dir outputs/simpletrack_txt_voxelnext \
  --config_path /scratch/work/jayawin1/techboost/lidar-human-tracking/simpletrack/SimpleTrack/configs/crane_ped.yaml \
  --obj_type box \
  --fps 3.0 \
  --default_score 1.0 \
  --score_threshold_override 0.0

"""