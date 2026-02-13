import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from lidar_human_detection_models.utils.dataset_openpcd import MyDataset
import yaml
from easydict import EasyDict
from pcdet.datasets.dataset import DatasetTemplate

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import copy

def parse_config():
    parser = argparse.ArgumentParser(description='Fine-tune OpenPCDet model')
    parser.add_argument('--cfg_file', type=str, required=True, help='Config file path')
    parser.add_argument('--dataset_cfg_file', type=str, required=True, help='Dataset config file path')
    parser.add_argument('--data_root', type=str, default='./data/small_dataset', help='Dataset root directory')
    parser.add_argument('--pretrained_ckpt', type=str, required=True, help='Pretrained checkpoint path')
    parser.add_argument('--output_dir', type=str, default='./fine_tune_ckpts', help='Directory to save fine-tuned checkpoints')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--weight_decay', type=float, help='Weight decay (overrides config)')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    with open(args.dataset_cfg_file, 'r') as f:
        dataset_cfg = yaml.safe_load(f)
    dataset_cfg = EasyDict(dataset_cfg)
    if args.batch_size:
        cfg.OPTIMIZATION.BATCH_SIZE = args.batch_size
    if args.epochs:
        cfg.OPTIMIZATION.EPOCHS = args.epochs
    if args.lr:
        cfg.OPTIMIZATION.LR = args.lr
    if args.weight_decay:
        cfg.OPTIMIZATION.WEIGHT_DECAY = args.weight_decay
    cfg.DATA_CONFIG = dataset_cfg.DATA_CONFIG

    return args, cfg, dataset_cfg

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.sample_file_list = []

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        points = self.sample_file_list[index]
        input_dict = {'points': points, 'frame_id': index}
        return self.prepare_data(data_dict=input_dict)

    def add_sample(self, points):
        self.sample_file_list.append(points)

def main():
    args, cfg, dataset_cfg = parse_config()
    os.makedirs(args.output_dir, exist_ok=True)

    # Logger setup
    logger = common_utils.create_logger('logs/finetune.log')

    root = Path(args.data_root)  

    # --- Dataset & Dataloaders ---
    train_dataset = MyDataset(
        dataset_cfg=dataset_cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=True,              # keep True so gt_boxes are included
        root_path=root,             
        logger=logger
    )
    val_dataset = MyDataset(
        dataset_cfg=dataset_cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=True,              # keep True so loss path can run
        root_path=root,
        logger=logger
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.OPTIMIZATION.BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.OPTIMIZATION.BATCH_SIZE,
        shuffle=False,
        collate_fn=val_dataset.collate_batch
    )

    # --- Model setup ---
    model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_dataset)
    model.load_params_from_file(filename=args.pretrained_ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.OPTIMIZATION.LR,
        weight_decay=cfg.OPTIMIZATION.WEIGHT_DECAY
    )

    # --- Tracking losses ---
    train_losses = []
    val_losses   = []
    
    # --- Training loop ---
    for epoch in range(1, cfg.OPTIMIZATION.EPOCHS + 1):
        # --- Training phase ---
        model.train()
        running_train_loss = 0.0
        for idx, data_dict in enumerate(train_loader):
            load_data_to_gpu(data_dict)

            ret_dict, _, _ = model(data_dict)
            loss = ret_dict['loss'].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation (compute loss via training path under no_grad) ---
        running_val_loss = 0.0
        with torch.no_grad():
            was_training = model.training
            model.train()
            toggled_bns = []
            for m in model.modules():
                if hasattr(m, "track_running_stats"):   
                    toggled_bns.append((m, m.track_running_stats))        
                    m.track_running_stats = False
                       
            for val_data in val_loader:
                load_data_to_gpu(val_data)
                ret_dict, _, _ = model(val_data)
                val_loss = ret_dict['loss'].mean()
                running_val_loss += val_loss.item()

            # restore BN flags
            for m, flag in toggled_bns:
                m.track_running_stats = flag
            # restore model mode
            if not was_training:
                model.eval()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        logger.info(
            f"[Epoch {epoch}/{cfg.OPTIMIZATION.EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # --- Save checkpoint --- 
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }
        ckpt_path = os.path.join(args.output_dir, f'finetuned_voxelnext_{epoch}.pth')
        if epoch%50==0:
            torch.save(checkpoint, ckpt_path)
        logger.info(f'Saved checkpoint: {ckpt_path}')

    # --- Plot training & validation loss curves ---
    plt.figure(figsize=(4, 6))

    final_train = train_losses[-1] if len(train_losses) else float("nan")
    final_val   = val_losses[-1]   if len(val_losses) else float("nan")

    plt.plot(train_losses, label=f'Train Loss (final: {final_train:.4f})')
    plt.plot(val_losses,   label=f'Validation Loss (final: {final_val:.4f})')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LiDAR Person Detection: Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_curve_path = os.path.join('./logs', 'loss_curve_voxelnext.png')
    plt.savefig(loss_curve_path, dpi=200) 
    plt.close()
    logger.info(f'Saved loss curve: {loss_curve_path}')


if __name__ == '__main__':
    main()


"""
Usage:
python lidar_human_detection_models/train.py \
  --cfg_file ./cfgs/nuscenes_voxelnext.yaml \
  --dataset_cfg_file ./cfgs/voxelnext_crane.yaml \
  --pretrained_ckpt ./models/voxelnext_nuscenes_kernel1.pth \
  --output_dir ./fine_tuned_ckpts \
  --batch_size 4 \
  --epochs 200 \
  --lr 5e-4
"""