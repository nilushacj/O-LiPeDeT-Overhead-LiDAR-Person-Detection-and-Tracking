import torch
import torch.nn as nn
import open3d.ml.torch as ml3d
from types import SimpleNamespace

def collate_fn(batch):
    """Collate batch and convert labels to dict of tensors: cls, bbox, dir"""
    points_list = [torch.from_numpy(item['points']).float() for item in batch]      # each (Ni, 3)
    feat_list   = [torch.from_numpy(item['intensity']).float() for item in batch]   # each (Ni, C) or (Ni, 1)
    # Convert raw label tuples to dict per sample
    labels_raw = [item['labels'] for item in batch]
    labels = []
    for sample_labels in labels_raw:
        cls_list = []
        bbox_list = []
        dir_list = []
        for cls_str, bbox_data in sample_labels:
            # Map class string to int index
            cls_idx = 1 if cls_str.lower() in ('human', 'person') else 0
            cls_list.append(cls_idx)
            bbox_list.append(bbox_data)
            # Dummy direction class (e.g., 0)
            dir_list.append(0)
        labels.append({
            'cls':   torch.tensor(cls_list, dtype=torch.long),
            'bbox':  torch.tensor(bbox_list, dtype=torch.float),
            'dir':   torch.tensor(dir_list, dtype=torch.long)
        })
    batch_obj = SimpleNamespace(point=points_list, feat=feat_list)
    return {'items': batch_obj, 'labels': labels}

class PointPillars(nn.Module):
    def __init__(self, num_input_features=4):
        super().__init__()
        self.pfn = ml3d.models.PointPillars(
            voxelize={'voxel_size': [0.2, 0.2, 0.2]},
            num_input_features=num_input_features,
            augment={}
        )

    def forward(self, items):
        return self.pfn(items)
