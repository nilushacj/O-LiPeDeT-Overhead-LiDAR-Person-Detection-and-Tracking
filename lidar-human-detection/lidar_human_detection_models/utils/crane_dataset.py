import os
import glob
import math
from types import SimpleNamespace
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
BEVBox3D = ml3d.datasets.utils.BEVBox3D

class CraneDataset(Dataset):
    """
    Expects:
      root_dir/
        training/
          pointclouds/*.bin      (float32, Nx4: x,y,z,intensity)
          labels/*.txt           (per pointcloud, lines: 'class x y z dx dy dz yaw_deg [optional_extra]')
        validation/...
    """
    def __init__(self, root_dir, split='training', classes=('box',), transforms=None):
        self.root = os.path.join(root_dir, split)
        self.pc_files = sorted(glob.glob(os.path.join(self.root, 'pointclouds', '*.bin')))
        self.label_files = sorted(glob.glob(os.path.join(self.root, 'labels', '*.txt')))
        assert len(self.pc_files) == len(self.label_files), "Mismatch between point clouds and labels"
        self.transforms = transforms
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.pc_files)

    def _read_points(self, bin_path) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = pts[:, :3].astype(np.float32)
        intensity = pts[:, 3:4].astype(np.float32)  # shape (N,1)
        return xyz, intensity

    def _read_labels(self, label_path) -> Tuple[List[BEVBox3D], np.ndarray]:
        bboxes = []
        labels = []
        with open(label_path, 'r') as f:
            for raw in [l for l in f.read().split('\n') if l.strip()]:
                vals = raw.strip().split()
                cls = vals[0]
                # Accept either 8 or >=8 fields; ignore extras
                nums = [float(x) for x in vals[1:]]
                if len(nums) < 8:
                    raise ValueError(f"Label line needs at least 8 numbers: {raw}")
                x, y, z, dx, dy, dz, yaw_deg = nums[:7]
                # Open3D BEVBox3D expects bottom-center (x,y,z), size=(w,l,h) or (l,h,w) per doc;
                # its to_xyzwhlr uses (x, y, z, w, l, h, yaw). We'll pass size as (w, l, h).
                # Here we treat dx=length (x-axis in LiDAR forward), dy=width (y), dz=height (z).
                # Yaw is around +Z; convert degrees -> radians.
                yaw = math.radians(yaw_deg)
                size = (dy, dx, dz)  # (w, l, h)
                center = (x, y, z)
                label_idx = self.class_to_idx.get(cls, None)
                if label_idx is None:
                    # unseen class -> skip
                    continue
                bboxes.append(BEVBox3D(center=center, size=size, yaw=yaw, label_class=label_idx, confidence=1.0))
                labels.append(label_idx)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        return bboxes, labels

    def __getitem__(self, idx):
        xyz, intensity = self._read_points(self.pc_files[idx])
        bboxes, labels = self._read_labels(self.label_files[idx])

        sample = {
            'point': xyz,          # (N,3) float32
            'feat': intensity,     # (N,1) float32 (optional but helpful)
            'bboxes': bboxes,      # list[BEVBox3D]
            'labels': labels,      # (Nb,) int
            'meta': {
                'id': os.path.splitext(os.path.basename(self.pc_files[idx]))[0],
                'path': self.pc_files[idx]
            }
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample

class Cfg(dict):
    """Dict that also allows attribute access (cfg.key) and still has .get()."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v
    def __delattr__(self, k):
        del self[k]

class CraneOpen3DDataset:
    """
    Minimal adapter that looks like an Open3D-ML dataset:
      - .cfg with at least .use_cache
      - .label_to_names (and get_label_to_names())
      - get_split('train'|'valid'|'validation'|'test'|'all')
    """
    def __init__(self, dataset_path: str, classes, use_cache=False, name='Crane'):
        self.dataset_path = dataset_path
        self.name = name
        self.classes = list(classes)
        self.label_to_names = {i: c for i, c in enumerate(self.classes)}
        self.cfg = _ml3d.utils.Config({
            "name": name,
            "dataset_path": dataset_path,
            "use_cache": use_cache,
            "classes": self.classes,
            # Optional knobs the pipeline sometimes reads via .get(...)
            "steps_per_epoch": None,
            "val_steps": None,
            "cache_dir": None,
        })


    def get_label_to_names(self):
        return self.label_to_names

    def get_split(self, split: str):
        split = {'train': 'training', 'valid': 'validation', 'val': 'validation'}.get(split, split)
        if split not in ('training', 'validation', 'test', 'all'):
            split = 'training'

        base = CraneDataset(self.dataset_path, split=split, classes=self.classes)

        class _Split:
            def __init__(self, base_ds: CraneDataset):
                self._base = base_ds
            def __len__(self):
                return len(self._base)
            def __getitem__(self, idx):
                return self._base[idx]

        return _Split(base)

    @staticmethod
    def is_tested(attr) -> bool:
        return False

    @staticmethod
    def save_test_result(results, attrs):
        pass