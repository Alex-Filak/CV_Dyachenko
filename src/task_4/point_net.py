import os
import sys
import time
import zipfile
import requests
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import open3d as o3d
import h5py

from config import *

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batch_size = trans.size()[0]
    I = torch.eye(d, device=trans.device).unsqueeze(0).repeat(batch_size, 1, 1)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3.bias.data.zero_()
        self.fc3.bias.data[:k*k].copy_(torch.eye(k).view(-1))

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Identity transformation
        identity = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)

        # Shared MLP
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # Batch norms
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Classifier
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.3)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x[:, :, :3]

        # Input transform
        input_transform = self.input_transform(x.transpose(2, 1))
        x = torch.bmm(x, input_transform)

        # First shared MLP
        x = x.transpose(2, 1)  # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)

        # Feature transform (on 64-dim features)
        feature_transform = self.feature_transform(x)  # TNet(k=64)
        x = x.transpose(2, 1)  # (B, N, 64)
        x = torch.bmm(x, feature_transform)
        x = x.transpose(2, 1)  # (B, 64, N)

        # Continue MLP
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 1024, N)

        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]  # (B, 1024, 1)
        x = x.view(-1, 1024)

        # Classifier
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x, input_transform, feature_transform

class ModelNetDataset(Dataset):
    def __init__(self, points, labels, augment=False):
        self.points = points
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.points)

    def augment_pointcloud(self, pointcloud):
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        scale = np.random.uniform(0.8, 1.2)

        xyz = pointcloud[:, :3]
        rgb = pointcloud[:, 3:]

        xyz = (xyz @ rotation_matrix.T) * scale
        xyz += np.clip(0.01 * np.random.randn(*xyz.shape), -0.02, 0.02)

        if np.random.rand() > 0.5:
            num_points = xyz.shape[0]
            keep_idx = np.random.choice(num_points, int(num_points * np.random.uniform(0.9, 1.0)), replace=False)
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

        if self.augment:
            pointcloud = self.augment_pointcloud(pointcloud)

        return torch.from_numpy(pointcloud).float(), torch.tensor(label).long()

# Convert mesh to point cloud
def read_off(filename):
    with open(filename, 'r') as f:
        header = f.readline().strip()
        if 'OFF' not in header:
            raise ValueError('Not a valid OFF file')
        if header != 'OFF':
            n_verts, n_faces, _ = map(int, header[3:].split())
        else:
            n_verts, n_faces, _ = map(int, f.readline().strip().split())
        
        verts = []
        for _ in range(n_verts):
            line = f.readline().strip()
            if line:
                verts.append(list(map(float, line.split())))
        
        for _ in range(n_faces):
            f.readline()
    
    return np.array(verts)

def mesh_to_pointcloud(mesh_path, num_points=1024):
    try:
        vertices = read_off(mesh_path)
        
        if vertices.shape[0] == 0:
            print(f"No vertices found in {mesh_path}")
            return None
            
        if len(vertices) < num_points:
            indices = np.random.choice(len(vertices), num_points, replace=True)
        else:
            indices = np.random.choice(len(vertices), num_points, replace=False)
        
        points = vertices[indices]
        
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        
        rgb = np.full((num_points, 3), 0.5, dtype=np.float32)
        return np.hstack([points, rgb])
    
    except Exception as e:
        print(f"Skipping {mesh_path}: {e}")
        return None

def download_dataset(dest_dir="ModelNet10"):
    if os.path.exists(dest_dir):
        print(f"✅ {dest_dir} already exists!")
        return

    url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    zip_path = "ModelNet10.zip"
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192), total=total_size//8192):
            f.write(chunk)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

def prepare_dataset():
    # Ensure directories exist
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    if os.path.exists(DATA_PATH):
        return


    modelnet_path = "ModelNet10"
    if not os.path.exists(modelnet_path):
        download_dataset()
        if not os.path.exists(modelnet_path):
            raise FileNotFoundError(f"ModelNet10 not found at {modelnet_path} even after download attempt")

    classes = [i for i in os.listdir(modelnet_path) 
               if os.path.isdir(os.path.join(modelnet_path, i)) and not i.startswith('.')]
    classes.sort()
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    print(f"Classes found: {classes}")

    all_points = []
    all_labels = []

    for cls in classes:
        cls_idx = class_to_idx[cls]
        print(f"\nProcessing class: {cls}")

        # Process train and test examples 
        for split in ["train", "test"]:
            dir_path = os.path.join(modelnet_path, cls, split)
            if not os.path.exists(dir_path):
                print(f"  ⚠️ {split} directory not found for {cls}, skipping")
                continue
                
            files = [i for i in os.listdir(dir_path) if i.endswith('.off')]
            print(f"  Found {len(files)} .off files in {split} set")
            
            for i in tqdm(files[:20], desc=f"{split} - {cls}"):
                mesh_path = os.path.join(dir_path, i)

                try:
                    points_colors = mesh_to_pointcloud(mesh_path, NUM_POINTS)
                    if points_colors is not None:
                        all_points.append(points_colors)
                        all_labels.append(cls_idx)
                    else:
                        print(f"  ⚠️ Failed to process {mesh_path}")

                except Exception as e:
                    print(f"  ❌ Error processing {mesh_path}: {e}")

    if len(all_points) == 0:
        raise RuntimeError("No valid point clouds were processed. Check your dataset paths and file formats.")

    # Save as HDF5
    all_points = np.array(all_points, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)

    with h5py.File(DATA_PATH, 'w') as file:
        file.create_dataset('points', data=all_points)
        file.create_dataset('labels', data=all_labels)
        file.create_dataset('classes', data=np.array(classes, dtype='S'))


# Training function    
def train():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    with h5py.File(DATA_PATH, 'r') as file:
        points = file['points'][:]
        labels = file['labels'][:]
        
        if 'classes' in file:
            classes = [name.decode('utf-8') for name in file['classes'][:]]
        else:
            classes = [
                'bathtub', 'bed', 'chair', 'desk', 'dresser',
                'monitor', 'night_stand', 'sofa', 'table', 'toilet'
            ]

    if points.shape[0] == 0:
        raise RuntimeError("HDF5 file is empty! Rebuild dataset.")

    # Create dataset 
    dataset = ModelNetDataset(points, labels, augment=True)
    train_size = int(TRAIN_FRAC * len(dataset))
    test_size = len(dataset) - train_size
    print(f"Dataset split: {train_size} training, {test_size} testing samples")

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize model
    model = PointNet(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training history
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_acc = 0

    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for data, target in progress_bar:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()

            # Forward pass
            output, input_transform, feature_transform = model(data)
            loss = criterion(output, target)
            loss += 0.001 * feature_transform_regularizer(input_transform)
            loss += 0.001 * feature_transform_regularizer(feature_transform)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()  # CORRECTED: was resetting instead of accumulating
            total += target.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss/len(train_loader):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluation
        model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Eval]", leave=False):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output, _, _ = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        test_loss = total_loss / len(test_loader)
        test_acc = 100. * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
                'classes': classes
            }, MODEL_PATH)
            best_preds = np.array(all_preds)
            best_targets = np.array(all_targets)
            print(f"💯 New best model saved with accuracy: {best_acc:.2f}%")

        scheduler.step()

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
    plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(best_targets, best_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    return model, classes, test_dataset, best_preds, best_targets

def visualize_predictions(model, test_dataset, classes, num_examples=5):
    model.eval()
    # Get random indices from test dataset
    indices = np.random.choice(len(test_dataset), min(num_examples, len(test_dataset)), replace=False)

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        points, label = test_dataset[idx]
        points_batch = points.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output, _, _ = model(points_batch)
        
        pred = output.argmax(dim=1).item()
        points_np = points.cpu().numpy()
        
        ax = plt.subplot(2, 3, i+1, projection='3d')
        x, y, z = points_np[:, 0], points_np[:, 1], points_np[:, 2]
        colors = points_np[:, 3:6]
        if colors.max() > 1:
            colors = colors / 255.0
        
        ax.scatter(x, y, z, c=colors, s=1, alpha=0.6)
        title = f"True: {classes[label]}\nPred: {classes[pred]}"
        if pred != label:
            title += " (INCORRECT)"
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'predictions.png'))
    plt.close()

    try:
        for idx in indices[:3]:
            points, label = test_dataset[idx]
            points_batch = points.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output, _, _ = model(points_batch)
            
            pred = output.argmax(dim=1).item()
            points_np = points.cpu().numpy()
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np[:, :3])
            
            colors = points_np[:, 3:6]
            if colors.max() > 1:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            title = f"True: {classes[label]}, Pred: {classes[pred]}"
            if pred != label:
                title += " (INCORRECT)"
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=title, width=800, height=600)
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            
            screenshot_path = os.path.join(OUTPUT_DIR, f'interactive_{idx}.png')
            vis.capture_screen_image(screenshot_path)
            vis.destroy_window()
            print(f"Interactive visualization screenshot saved: {screenshot_path}")
            
    except Exception as e:
        print("Visualization skiped.")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    # Download and prepare dataset
    download_dataset()
    prepare_dataset()
    
    # Train model
    model, classes, test_dataset, best_preds, best_targets = train()
    
    # Load best model
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Visualize predictions
    visualize_predictions(model, test_dataset, classes)
    
    cm = confusion_matrix(best_targets, best_preds)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    worst_classes = np.argsort(class_accuracies)[:3]
    
    
    # Save results for download
    result_files = [
        MODEL_PATH,
        os.path.join(OUTPUT_DIR, "training_history.png"),
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
        os.path.join(OUTPUT_DIR, "predictions.png"),
        DATA_PATH
    ]
    
    # Filter out files that don't exist
    existing_files = [f for f in result_files if os.path.exists(f)]
    
    zip_path = os.path.join(OUTPUT_DIR, "pointnet_results.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for f in existing_files:
            zf.write(f, os.path.basename(f))
    
    print(f"Results saved to {zip_path}")
    
    # Download in Colab
    try:
        from google.colab import files
        files.download(zip_path)
    except Exception as e:
        print("Error while loading files from colab")
