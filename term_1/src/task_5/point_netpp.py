import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from config import *

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batch_size = trans.size()[0]
    I = torch.eye(d, device=trans.device).unsqueeze(0).repeat(batch_size, 1, 1)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

# PointNet++ specific modules
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1).contiguous()
        if points is not None:
            points = points.permute(0, 2, 1).contiguous()

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1).contiguous()
        
        return new_xyz, new_points

def sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint
    
    centroids_idx = farthest_point_sample(xyz, S)
    new_xyz = index_points(xyz, centroids_idx)
    
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    
    return new_xyz, new_points

def sample_and_group_all(xyz, points):
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(xyz.device)
    grouped_xyz = xyz.view(B, 1, N, C)
    
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    
    return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1).contiguous()
        xyz2 = xyz2.permute(0, 2, 1).contiguous()
        
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = torch.topk(dists, 3, dim=-1, largest=False)
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            
            interpolated_points = torch.sum(
                index_points(points2.permute(0, 2, 1).contiguous(), idx) * weight.view(B, N, 3, 1), 
                dim=2
            )
            interpolated_points = interpolated_points.permute(0, 2, 1).contiguous()
        
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points

class PointNet2Seg(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS):
        super(PointNet2Seg, self).__init__()
        
        # First layer: 3 (rel xyz) + 3 (rgb) = 6
        self.sa1 = PointNetSetAbstraction(
            npoint=SA_PARAMS[0]['npoint'],
            radius=SA_PARAMS[0]['radius'],
            nsample=SA_PARAMS[0]['nsample'],
            in_channel=6,  # ← hardcoded to 6
            mlp=SA_PARAMS[0]['mlp']
        )
        
        self.sa2 = PointNetSetAbstraction(
            npoint=SA_PARAMS[1]['npoint'],
            radius=SA_PARAMS[1]['radius'],
            nsample=SA_PARAMS[1]['nsample'],
            in_channel=SA_PARAMS[0]['mlp'][-1] + 3,  # 64 + 3
            mlp=SA_PARAMS[1]['mlp']
        )
        
        self.sa3 = PointNetSetAbstraction(
            npoint=SA_PARAMS[2]['npoint'],
            radius=SA_PARAMS[2]['radius'],
            nsample=SA_PARAMS[2]['nsample'],
            in_channel=SA_PARAMS[1]['mlp'][-1] + 3,  # 128 + 3
            mlp=SA_PARAMS[2]['mlp']
        )
        
        self.sa4 = PointNetSetAbstraction(
            npoint=SA_PARAMS[3]['npoint'],
            radius=SA_PARAMS[3]['radius'],
            nsample=SA_PARAMS[3]['nsample'],
            in_channel=SA_PARAMS[2]['mlp'][-1] + 3,  # 256 + 3
            mlp=SA_PARAMS[3]['mlp']
        )
        
        # Decoder
        self.fp4 = PointNetFeaturePropagation(in_channel=FP_PARAMS[0]['in_channel'], mlp=FP_PARAMS[0]['mlp'])
        self.fp3 = PointNetFeaturePropagation(in_channel=FP_PARAMS[1]['in_channel'], mlp=FP_PARAMS[1]['mlp'])
        self.fp2 = PointNetFeaturePropagation(in_channel=FP_PARAMS[2]['in_channel'], mlp=FP_PARAMS[2]['mlp'])
        self.fp1 = PointNetFeaturePropagation(in_channel=FP_PARAMS[3]['in_channel'], mlp=FP_PARAMS[3]['mlp'])
        
        self.conv1 = nn.Conv1d(HEAD_MLP[0], HEAD_MLP[1], 1)
        self.bn1 = nn.BatchNorm1d(HEAD_MLP[1])
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(HEAD_MLP[1], num_classes, 1)

    def forward(self, xyz):
        xyz_input = xyz[:, :, :3]              
        if xyz.shape[2] > 3:
            features = xyz[:, :, 3:].transpose(1, 2).contiguous() 
        else:
            features = None

        xyz_input = xyz_input.transpose(1, 2).contiguous() 

        # Encoder
        l1_xyz, l1_points = self.sa1(xyz_input, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Decoder
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz_input, l1_xyz, features, l1_points)  # ← pass 'features', not None

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(x)
        seg_pred = self.conv2(x)
        return seg_pred

class S3DISDataset(Dataset):
    def __init__(self, points, labels, num_points=NUM_POINTS, augment=False):
        self.points = points
        self.labels = labels
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.points)

    def augment_pointcloud(self, pointcloud):
        if AUGMENT_ROTATION:
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            xyz = pointcloud[:, :3] @ rotation_matrix.T
        else:
            xyz = pointcloud[:, :3]
        
        scale = np.random.uniform(*AUGMENT_SCALE)
        xyz = xyz * scale
        
        xyz += np.clip(
            AUGMENT_JITTER * np.random.randn(*xyz.shape),
            -2 * AUGMENT_JITTER,
            2 * AUGMENT_JITTER
        )

        rgb = pointcloud[:, 3:]
        
        if np.random.rand() > 0.5:
            num_points = xyz.shape[0]
            keep_ratio = np.random.uniform(*AUGMENT_DROPOUT)
            keep_idx = np.random.choice(num_points, int(num_points * keep_ratio), replace=False)
            xyz = xyz[keep_idx]
            rgb = rgb[keep_idx]
            if len(keep_idx) < num_points:
                duplicate_idx = np.random.choice(len(keep_idx), num_points - len(keep_idx), replace=True)
                xyz = np.vstack([xyz, xyz[duplicate_idx]])
                rgb = np.vstack([rgb, rgb[duplicate_idx]])

        return np.hstack([xyz, rgb])

    def __getitem__(self, idx):
        pointcloud = self.points[idx].copy()
        label = self.labels[idx]

        if isinstance(label, np.ndarray) and label.ndim == 1 and len(label) == 1:
            label = np.full(pointcloud.shape[0], label[0])
        elif isinstance(label, (int, np.integer)):
            label = np.full(pointcloud.shape[0], label)
        
        if pointcloud.shape[0] > self.num_points:
            indices = np.random.choice(pointcloud.shape[0], self.num_points, replace=False)
            pointcloud = pointcloud[indices]
            label = label[indices]
        elif pointcloud.shape[0] < self.num_points:
            pad_size = self.num_points - pointcloud.shape[0]
            pointcloud = np.pad(pointcloud, ((0, pad_size), (0, 0)), mode='constant')
            label = np.pad(label, (0, pad_size), mode='constant')

        if self.augment:
            pointcloud = self.augment_pointcloud(pointcloud)

        return {
            'points': torch.from_numpy(pointcloud).float(),
            'labels': torch.from_numpy(label).long()
        }

def load_s3dis_area(area_path, num_points=NUM_POINTS):
    room_paths = [os.path.join(area_path, f) for f in os.listdir(area_path)
                  if os.path.isdir(os.path.join(area_path, f))]
    all_points = []
    all_labels = []

    for room_path in tqdm(room_paths, desc=f"Loading rooms from {os.path.basename(area_path)}"):
        ann_dir = os.path.join(room_path, "Annotations")
        if not os.path.exists(ann_dir):
            continue

        room_points = []
        room_labels = []

        for obj_file in glob.glob(os.path.join(ann_dir, "*.txt")):
            class_name = os.path.basename(obj_file).split("_")[0]
            if class_name not in S3DIS_CLASSES:
                continue
            cls_id = S3DIS_CLASSES.index(class_name)
            try:
                data = np.loadtxt(obj_file)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                room_points.append(data)
                room_labels.append(np.full(len(data), cls_id))
            except Exception:
                continue

        if room_points:
            points = np.concatenate(room_points, axis=0)
            labels = np.concatenate(room_labels, axis=0)

            xyz = points[:, :3]
            rgb = points[:, 3:] / 255.0
            centroid = np.mean(xyz, axis=0)
            xyz = xyz - centroid
            max_norm = np.max(np.linalg.norm(xyz, axis=1))
            if max_norm > 0:
                xyz = xyz / max_norm

            full_points = np.hstack([xyz, rgb])

            if len(full_points) >= num_points:
                idx = np.random.choice(len(full_points), num_points, replace=False)
            else:
                idx = np.random.choice(len(full_points), num_points, replace=True)

            all_points.append(full_points[idx])
            all_labels.append(labels[idx])

    return all_points, all_labels

def prepare_dataset():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if os.path.exists(DATA_PATH):
        return

    s3dis_root = "/content/drive/MyDrive/PointNet++/Stanford3dDataset_v1.2"
    if not os.path.exists(s3dis_root):
        raise FileNotFoundError(f"Dataset not found at {s3dis_root}")

    areas = ["Area_1"]
    all_points, all_labels = [], []

    for area in areas:
        area_path = os.path.join(s3dis_root, area)
        if not os.path.isdir(area_path):
            continue
        pts, lbls = load_s3dis_area(area_path, num_points=NUM_POINTS)
        all_points.extend(pts)
        all_labels.extend(lbls)

    if not all_points:
        raise RuntimeError("No valid point clouds were processed. Check your dataset paths and file formats.")

    all_points = np.array(all_points, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)

    with h5py.File(DATA_PATH, 'w') as file:
        file.create_dataset('points', data=all_points)
        file.create_dataset('labels', data=all_labels)
        file.create_dataset('classes', data=np.array(S3DIS_CLASSES, dtype='S'))

def train():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with h5py.File(DATA_PATH, 'r') as file:
        points = file['points'][:]
        labels = file['labels'][:]
        if 'classes' in file:
            classes = [name.decode('utf-8') for name in file['classes'][:]]
        else:
            classes = S3DIS_CLASSES
    
    print(f"Original labels shape: {labels.shape}")
    print(f"Points per sample: {points.shape[1]}")
    
    per_point_labels = np.array([
        np.full(points.shape[1], labels[i]) for i in range(len(labels))
    ])
    print(f"Converted labels shape: {per_point_labels.shape}")
    
    if points.shape[0] == 0:
        raise RuntimeError("HDF5 file is empty! Rebuild dataset.")
    
    dataset = S3DISDataset(points, per_point_labels, num_points=NUM_POINTS, augment=True)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    num_classes = len(classes)
    model = PointNet2Seg(num_classes=num_classes, input_channels=INPUT_CHANNELS).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    train_losses, val_losses = [], []
    train_oa, val_oa = [], []
    train_miou, val_miou = [], []
    best_miou = 0
    
    print(f"Starting PointNet++ segmentation training on {DEVICE}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {classes}")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_correct = 0
        total_points = 0
        confusion_mat = torch.zeros(num_classes, num_classes, device=DEVICE)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for batch in progress_bar:
            data = batch['points'].to(DEVICE)
            target = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.transpose(1, 2).contiguous()
            loss = criterion(seg_pred.view(-1, num_classes), target.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = seg_pred.argmax(dim=2)
            correct = (pred == target).sum().item()
            total_correct += correct
            total_points += target.numel()
            
            for cls_true in range(num_classes):
                for cls_pred in range(num_classes):
                    confusion_mat[cls_true, cls_pred] += ((target == cls_true) & (pred == cls_pred)).sum().item()
            
            current_oa = 100. * correct / target.numel()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'OA': f"{current_oa:.2f}%"})
        
        train_loss = total_loss / len(train_loader)
        train_accuracy = 100. * total_correct / total_points
        train_iou_per_class = []
        for i in range(num_classes):
            tp = confusion_mat[i, i].item()
            fp = confusion_mat[:, i].sum().item() - tp
            fn = confusion_mat[i, :].sum().item() - tp
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            train_iou_per_class.append(iou)
        train_miou_value = np.mean(train_iou_per_class) * 100
        
        train_losses.append(train_loss)
        train_oa.append(train_accuracy)
        train_miou.append(train_miou_value)
        
        model.eval()
        val_total_loss = 0
        val_total_correct = 0
        val_total_points = 0
        val_confusion_mat = torch.zeros(num_classes, num_classes, device=DEVICE)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False):
                data = batch['points'].to(DEVICE)
                target = batch['labels'].to(DEVICE)
                seg_pred = model(data)
                seg_pred = seg_pred.transpose(1, 2).contiguous()
                loss = criterion(seg_pred.view(-1, num_classes), target.view(-1))
                val_total_loss += loss.item()
                pred = seg_pred.argmax(dim=2)
                val_total_correct += (pred == target).sum().item()
                val_total_points += target.numel()
                for cls_true in range(num_classes):
                    for cls_pred in range(num_classes):
                        val_confusion_mat[cls_true, cls_pred] += ((target == cls_true) & (pred == cls_pred)).sum().item()
        
        val_loss = val_total_loss / len(val_loader)
        val_accuracy = 100. * val_total_correct / val_total_points
        val_iou_per_class = []
        for i in range(num_classes):
            tp = val_confusion_mat[i, i].item()
            fp = val_confusion_mat[:, i].sum().item() - tp
            fn = val_confusion_mat[i, :].sum().item() - tp
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            val_iou_per_class.append(iou)
        val_miou_value = np.mean(val_iou_per_class) * 100
        
        val_losses.append(val_loss)
        val_oa.append(val_accuracy)
        val_miou.append(val_miou_value)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"Train Loss: {train_loss:.4f}, Train OA: {train_accuracy:.2f}%, Train mIoU: {train_miou_value:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val OA: {val_accuracy:.2f}%, Val mIoU: {val_miou_value:.2f}%")
        print("Per-class IoU (Val):")
        for i, (cls_name, iou) in enumerate(zip(classes, val_iou_per_class)):
            print(f"    {cls_name:15s}: {iou*100:6.2f}%")
        
        if val_miou_value > best_miou:
            best_miou = val_miou_value
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou_value,
                'val_oa': val_accuracy,
                'train_miou': train_miou_value,
                'train_oa': train_accuracy,
                'classes': classes,
                'per_class_iou': val_iou_per_class
            }, MODEL_PATH)
            print(f"New best model saved with mIoU: {best_miou:.2f}%")
        
        scheduler.step()
    
    print("\n" + "="*50)
    print("FINAL TESTING ON TEST SET")
    print("="*50)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_total_correct = 0
    test_total_points = 0
    test_confusion_mat = torch.zeros(num_classes, num_classes, device=DEVICE)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            data = batch['points'].to(DEVICE)
            target = batch['labels'].to(DEVICE)
            seg_pred = model(data)
            seg_pred = seg_pred.transpose(1, 2).contiguous()
            pred = seg_pred.argmax(dim=2)
            test_total_correct += (pred == target).sum().item()
            test_total_points += target.numel()
            for cls_true in range(num_classes):
                for cls_pred in range(num_classes):
                    test_confusion_mat[cls_true, cls_pred] += ((target == cls_true) & (pred == cls_pred)).sum().item()
    
    test_accuracy = 100. * test_total_correct / test_total_points
    test_iou_per_class = []
    for i in range(num_classes):
        tp = test_confusion_mat[i, i].item()
        fp = test_confusion_mat[:, i].sum().item() - tp
        fn = test_confusion_mat[i, :].sum().item() - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        test_iou_per_class.append(iou)
    test_miou_value = np.mean(test_iou_per_class) * 100
    
    print(f"\nTest Results:")
    print(f"Overall Accuracy (OA): {test_accuracy:.2f}%")
    print(f"Mean IoU (mIoU): {test_miou_value:.2f}%")
    print(f"\nPer-class IoU:")
    for i, (cls_name, iou) in enumerate(zip(classes, test_iou_per_class)):
        print(f"  {cls_name:15s}: {iou*100:6.2f}%")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.plot(train_oa, label='Train OA', marker='o')
    plt.plot(val_oa, label='Val OA', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Overall Accuracy (%)')
    plt.title('Training and Validation OA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.plot(train_miou, label='Train mIoU', marker='o')
    plt.plot(val_miou, label='Val mIoU', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU (%)')
    plt.title('Training and Validation mIoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
    plt.close()
    print(f"\nTraining history plot saved to {os.path.join(OUTPUT_DIR, 'training_history.png')}")

if __name__ == "__main__":
    prepare_dataset()
    train()
