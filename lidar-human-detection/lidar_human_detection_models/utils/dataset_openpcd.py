import numpy as np
import os
from pcdet.datasets import DatasetTemplate

class MyDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.data_path = dataset_cfg.DATA_PATH
        self.split = 'training' if training else 'test'
        self.frames = sorted(os.listdir(os.path.join(self.data_path, self.split, "pointclouds")))
        # self.num_point_features = dataset_cfg.NUM_POINT_FEATURES
        # self.voxel_size = dataset_cfg.VOXEL_SIZE
        # self.grid_size = dataset_cfg.GRID_SIZE
        # self.num_point_features = 4
        if training:
            self.label_path = os.path.join(self.data_path, self.split, "labels")

    def get_gt_boxes(self, index):
        gt_names = []
        label_file = os.path.join(self.label_path, self.frames[index].replace('.bin', '.txt'))
        gt_boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                gt_names.append(parts[0])               # e.g., 'Pedestrian'
                box = [float(x) for x in parts[1:8]]    # x, y, z, dx, dy, dz, heading
                gt_boxes.append(box)

        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
            gt_names = np.array([], dtype=object)
        else:
            gt_boxes = np.array(gt_boxes, dtype=np.float32)
            gt_names = np.array(gt_names, dtype=object)  # or dtype=str

        return gt_boxes, gt_names

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        file_path = os.path.join(self.data_path, self.split, "pointclouds", self.frames[index])
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        # points = np.fromfile(file_path, dtype=np.float32).reshape(-1, self.num_point_features)
        input_dict = {
            'points': points,
            'frame_id': index,
        }
        if self.training:
            input_dict['gt_boxes'], input_dict['gt_names'] = self.get_gt_boxes(index)

        # limit = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        # xyz = points[:, :3]
        # in_range = ((xyz >= limit[:3]) & (xyz <= limit[3:6])).all(1)
        # print(f"[idx {index}] points: total={len(points)}, in_range={in_range.sum()}")
        # print("points xyz min:", xyz.min(0), "max:", xyz.max(0))

        data_dict = self.prepare_data(input_dict)
        return data_dict
