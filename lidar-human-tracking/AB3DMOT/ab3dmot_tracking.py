#!/usr/bin/env python3
import os, json, argparse, numpy as np, math, csv
from pathlib import Path
import time
# Optional SciPy for Hungarian; greedy fallback if missing
try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ---- helpers ----
def list_txt_frames(in_dir: Path, start: str = "", end: str = ""):
    """
    Returns sorted list of *.txt files (Path objects).
    If start/end provided, they are matched on stem (filename without suffix).
    """
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
    Reads lines like:
      box x y z dx dy dz yaw_deg
    or:
      box x y z dx dy dz yaw_deg score

    Returns:
      boxes: (N,7) float32 with yaw in RADIANS
      scores: (N,) float32
    """
    boxes = []
    scores = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0].lower() != "box":
                continue

            if len(parts) not in (8, 9):
                raise ValueError(f"Bad line in {txt_path}: '{line}' (expected 8 or 9 tokens)")

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


def save_tracks_txt(out_dir: Path, frame_stem: str, outputs):
    """
    Writes per-frame track file:
      track id x y z dx dy dz yaw_deg score
    """
    out_path = out_dir / f"{frame_stem}.txt"
    with open(out_path, "w") as f:
        for o in outputs:
            tid = int(o["id"])
            x, y, z, dx, dy, dz, yaw = [float(v) for v in o["box"]]
            yaw_deg = float(np.rad2deg(yaw))
            score = float(o["score"])
            f.write(f"track {tid} {x:.8f} {y:.8f} {z:.8f} {dx:.8f} {dy:.8f} {dz:.8f} {yaw_deg:.6f} {score:.6f}\n")

def wrap_angle(a):
    # wrap to [-pi, pi)
    return (a + np.pi) % (2 * np.pi) - np.pi

def bev_corners_xy(box):
    # box = [x,y,z,dx,dy,dz,yaw]  -> (4,2) in BEV (CCW)
    x,y,_,dx,dy,_,yaw = box
    c,s = np.cos(yaw), np.sin(yaw)
    hx, hy = dx*0.5, dy*0.5
    rect = np.array([[ hx,  hy],
                     [-hx,  hy],
                     [-hx, -hy],
                     [ hx, -hy]], dtype=np.float32)
    R = np.array([[c,-s],[s,c]], dtype=np.float32)
    pts = rect @ R.T
    pts[:,0] += x; pts[:,1] += y
    return pts

def iou_bev(b1, b2):
    # Get AABB extents by rotating corners then min/max
    p1 = bev_corners_xy(b1); p2 = bev_corners_xy(b2)
    x1min,x1max = p1[:,0].min(), p1[:,0].max()
    y1min,y1max = p1[:,1].min(), p1[:,1].max()
    x2min,x2max = p2[:,0].min(), p2[:,0].max()
    y2min,y2max = p2[:,1].min(), p2[:,1].max()
    ixmin, ixmax = max(x1min,x2min), min(x1max,x2max)
    iymin, iymax = max(y1min,y2min), min(y1max,y2max)
    iw, ih = max(0.0, ixmax-ixmin), max(0.0, iymax-iymin)
    inter = iw*ih
    a1 = (x1max-x1min)*(y1max-y1min)
    a2 = (x2max-x2min)*(y2max-y2min)
    union = a1 + a2 - inter + 1e-8
    return float(inter/union)

# ---- Kalman filter ----
class Kalman3D:
    """
    Constant-velocity Kalman filter on [x,y,z,vx,vy,vz].
    We measure [x,y,z] from detections.
    Size (dx,dy,dz) and yaw handled by simple EMA smoothing at Track level.
    """
    def __init__(self, dt=0.33,  # ~3 Hz default;
                 process_var_pos=1e-2,
                 process_var_vel=5e-2,
                 meas_var_pos=2.5e-3):
        # state x: [x,y,z,vx,vy,vz]^T
        self.x = np.zeros((6,1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 1e1

        self.F = np.eye(6, dtype=np.float32)
        self.F[0,3] = dt
        self.F[1,4] = dt
        self.F[2,5] = dt

        self.H = np.zeros((3,6), dtype=np.float32)
        self.H[0,0] = 1.0
        self.H[1,1] = 1.0
        self.H[2,2] = 1.0

        q_pos = process_var_pos
        q_vel = process_var_vel
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float32)

        r = meas_var_pos
        self.R = np.diag([r, r, r]).astype(np.float32)

        self.I = np.eye(6, dtype=np.float32)

    def init_state(self, x, y, z):
        self.x[:] = 0.0
        self.x[0,0] = x; self.x[1,0] = y; self.x[2,0] = z
        self.P = np.eye(6, dtype=np.float32) * 1.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        # z = [x,y,z]
        z = z.reshape(3,1).astype(np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

    def maha_distance(self, z):
        # Mahalanobis distance in measurement space
        z = z.reshape(3,1).astype(np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        try:
            invS = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            invS = np.linalg.pinv(S)
        d2 = float(y.T @ invS @ y)
        return d2


# ---- Track object ----
class Track:
    _next_id = 1

    def __init__(self, det_box, det_score, dt, ema_alpha_size=0.3, ema_alpha_yaw=0.3):
        # IDs start at 1 for convenient coloring
        self.id = Track._next_id
        Track._next_id += 1

        x,y,z,dx,dy,dz,yaw = det_box
        self.kf = Kalman3D(dt=dt)
        self.kf.init_state(x,y,z)

        # smooth attrs
        self.dx = float(dx); self.dy = float(dy); self.dz = float(dz)
        self.yaw = float(yaw)
        self.ema_a_size = float(ema_alpha_size)
        self.ema_a_yaw  = float(ema_alpha_yaw)

        self.score = float(det_score)

        self.age = 0           # total frames since birth
        self.time_since_update = 0
        self.hits = 1          # total successful updates
        self.hit_streak = 1    # consecutive updates

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, det_box, det_score):
        x,y,z,dx,dy,dz,yaw = det_box
        self.kf.update(np.array([x,y,z], dtype=np.float32))

        # EMA for size & yaw (handle yaw wrap)
        self.dx = self.ema_a_size*dx + (1-self.ema_a_size)*self.dx
        self.dy = self.ema_a_size*dy + (1-self.ema_a_size)*self.dy
        self.dz = self.ema_a_size*dz + (1-self.ema_a_size)*self.dz

        dyaw = wrap_angle(yaw - self.yaw)
        self.yaw = wrap_angle(self.yaw + self.ema_a_yaw*dyaw)

        self.score = 0.6*self.score + 0.4*float(det_score)

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state_box(self):
        x,y,z = self.kf.x[0,0], self.kf.x[1,0], self.kf.x[2,0]
        return np.array([x,y,z,self.dx,self.dy,self.dz,self.yaw], dtype=np.float32)

    def center(self):
        return np.array([self.kf.x[0,0], self.kf.x[1,0], self.kf.x[2,0]], dtype=np.float32)


# ---- Tracker ----
class MultiObjectTracker3D:
    def __init__(self,
                 dt=0.33,
                 max_age=10,
                 min_hits=2,
                 init_delay=0,
                 maha_gate=16.0,        # ~chi2 threshold for 3 dof (pos)
                 w_maha=1.0,
                 w_iou=0.3,             # small weight; mainly use motion
                 iou_floor=0.0,
                 score_norm=(0.0,1.0)): # not in cost by default, kept for future use
        self.dt = dt
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.init_delay = int(init_delay)

        self.maha_gate = float(maha_gate)
        self.w_maha = float(w_maha)
        self.w_iou  = float(w_iou)
        self.iou_floor = float(iou_floor)

        self.tracks = []

    def _build_cost(self, det_boxes, det_scores):
        """
        Cost matrix between current tracks (rows) and detections (cols).
        cost = w_maha * mahalanobis(pos)^0.5  +  w_iou * (1 - IoU_bev_clipped)
        Invalid pairs (beyond gate) set to a large number.
        """
        T = len(self.tracks)
        D = len(det_boxes)
        if T==0 or D==0:
            return np.zeros((T,D), dtype=np.float32)

        cost = np.full((T,D), 1e6, dtype=np.float32)

        for ti, trk in enumerate(self.tracks):
            trk_box = trk.get_state_box()
            for dj, det in enumerate(det_boxes):
                # Mahalanobis gating on (x,y,z)
                z = np.array(det[:3], dtype=np.float32)
                d2 = trk.kf.maha_distance(z)
                if d2 > self.maha_gate:
                    continue  # leave as large cost (invalid)

                c_maha = math.sqrt(max(d2, 0.0))

                iou = iou_bev(trk_box, det)
                iou = max(iou, self.iou_floor)
                c_iou = (1.0 - iou)

                cost[ti, dj] = self.w_maha*c_maha + self.w_iou*c_iou

        return cost

    def _assign(self, cost):
        T, D = cost.shape
        if T==0 or D==0:
            return [], list(range(T)), list(range(D))

        # mask invalid (very large) costs
        invalid = (cost > 1e5)

        if _HAVE_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost)
            matches = []
            unmatched_t = set(range(T))
            unmatched_d = set(range(D))
            for r,c in zip(row_ind, col_ind):
                if invalid[r,c]:
                    continue
                matches.append((r,c))
                unmatched_t.discard(r)
                unmatched_d.discard(c)
            return matches, list(unmatched_t), list(unmatched_d)

        # Greedy fallback
        matches = []
        unmatched_t = set(range(T))
        unmatched_d = set(range(D))
        # consider only valid pairs sorted by cost
        pairs = [(cost[i,j], i, j) for i in range(T) for j in range(D) if not invalid[i,j]]
        pairs.sort(key=lambda x: x[0])
        used_t, used_d = set(), set()
        for c,i,j in pairs:
            if i in used_t or j in used_d: continue
            matches.append((i,j)); used_t.add(i); used_d.add(j)
        for i in range(T):
            if i not in used_t: unmatched_t.add(i)
        for j in range(D):
            if j not in used_d: unmatched_d.add(j)
        return matches, list(unmatched_t), list(unmatched_d)

    def step(self, det_boxes, det_scores):
        # 1) predict
        for trk in self.tracks:
            trk.predict()
            trk.hit_streak = 0 if trk.time_since_update > 0 else trk.hit_streak

        # 2) associate
        cost = self._build_cost(det_boxes, det_scores)
        matches, un_tracks, un_dets = self._assign(cost)

        # 3) update matched
        for ti, dj in matches:
            self.tracks[ti].update(det_boxes[dj], det_scores[dj])

        # 4) create new tracks for unmatched detections
        for dj in un_dets:
            self.tracks.append(Track(det_boxes[dj], det_scores[dj], dt=self.dt))

        # 5) remove dead tracks
        kept = []
        for trk in self.tracks:
            if trk.time_since_update <= self.max_age:
                kept.append(trk)
        self.tracks = kept

        # output visible tracks (policy: return those with hits >= min_hits OR just updated)
        outputs = []
        for trk in self.tracks:
            if (trk.hits >= self.min_hits) or (trk.time_since_update == 0 and trk.age >= self.init_delay):
                outputs.append({
                    "id": trk.id,
                    "box": trk.get_state_box(),
                    "score": trk.score,
                    "age": trk.age,
                    "time_since_update": trk.time_since_update,
                    "hits": trk.hits,
                    "hit_streak": trk.hit_streak
                })
        return outputs


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_points(bin_path):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]  # xyz only


# ---- main ----
def parse_args():
    ap = argparse.ArgumentParser("3D MOT from TXT detections (AB3DMOT-style)")
    ap.add_argument("--in_dir", required=True, type=str,
                    help="Folder containing per-frame *.txt files (labelCloud format)")
    ap.add_argument("--out_dir", required=True, type=str,
                    help="Where to write per-frame tracked *.txt files")
    ap.add_argument("--start", default="", type=str,
                    help="Optional start frame stem (e.g., frame_000123)")
    ap.add_argument("--end", default="", type=str,
                    help="Optional end frame stem (e.g., frame_000200)")
    ap.add_argument("--fps", type=float, default=3.0, help="Frame rate to set Kalman dt=1/fps")
    # If txt lacks scores, use this:
    ap.add_argument("--default_score", type=float, default=1.0,
                    help="Score assigned to each input box if the txt has no score column")

    # tracker params
    ap.add_argument("--max_age", type=int, default=8)
    ap.add_argument("--min_hits", type=int, default=2)
    ap.add_argument("--init_delay", type=int, default=0)
    ap.add_argument("--maha_gate", type=float, default=16.0)
    ap.add_argument("--w_maha", type=float, default=1.0)
    ap.add_argument("--w_iou",  type=float, default=0.3)
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
    frames = list_txt_frames(in_dir, start="cloud_stack3_1483391058_400703192.bin.txt", end="cloud_stack3_1483391073_400696039.bin.txt")
    dt = 1.0 / max(1e-6, args.fps)

    tracker = MultiObjectTracker3D(
        dt=dt,
        max_age=args.max_age,
        min_hits=args.min_hits,
        init_delay=args.init_delay,
        maha_gate=args.maha_gate,
        w_maha=args.w_maha,
        w_iou=args.w_iou
    )

    # per-frame TXT outputs + CSV summary (optional but handy)
    out_txt_dir = out_root / "tracks_txt"
    ensure_dir(out_txt_dir)

    csv_path = out_root / "tracks.csv"
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

    with open(csv_path, "a", newline="") as fcsv:
        w = csv.writer(fcsv)
        if write_header:
            w.writerow(["frame","id","x","y","z","dx","dy","dz","yaw_deg","score","age","time_since_update","hits","hit_streak"])
        tot_times = []
        for k, txt_path in enumerate(frames):
            det_boxes, det_scores = load_boxes_from_txt(txt_path, default_score=args.default_score)
            start_time = time.time()
            outputs = tracker.step(det_boxes, det_scores)
            tot_times.append(time.time()-start_time)
            save_tracks_txt(out_txt_dir, txt_path.stem, outputs)

            for o in outputs:
                x,y,z,dx,dy,dz,yaw = o["box"]
                w.writerow([
                    txt_path.stem, int(o["id"]),
                    float(x), float(y), float(z),
                    float(dx), float(dy), float(dz),
                    float(np.rad2deg(yaw)),
                    float(o["score"]),
                    int(o["age"]), int(o["time_since_update"]),
                    int(o["hits"]), int(o["hit_streak"])
                ])

            print(f"[{k+1}/{len(frames)}] {txt_path.name} -> tracks_out={len(outputs)}")

    with open(out_root / "index.json", "w") as f:
        json.dump({
            "source_in_dir": str(in_dir),
            "frames": [p.name for p in frames],
            "fps": args.fps,
            "dt": dt,
            "default_score": args.default_score,
            "max_age": args.max_age,
            "min_hits": args.min_hits,
            "init_delay": args.init_delay,
            "maha_gate": args.maha_gate,
            "w_maha": args.w_maha,
            "w_iou": args.w_iou,
            "tracks_txt_dir": str(out_txt_dir),
            "csv": str(csv_path),
        }, f, indent=2)

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

python3 ab3dmot_tracking.py \
  --in_dir /PATH/TO/REPO/lidar-human-detection-models/outputs/preds_for_tracks_voxelnext/run_00/preds_txt \
  --out_dir outputs/ab3dmot_txt_voxelnext \
  --fps 3.0 \
  --max_age 3 --min_hits 2 --w_maha 1.0 --w_iou 0.3 \
  --default_score 1.0

"""