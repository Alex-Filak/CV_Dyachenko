import os
import h5py
import time
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import seaborn as sns
from sklearn.metrics import  confusion_matrix

import open3d as n3d

import matplotlib.pyplot as plt

from config import *



# TRAIN_FRAC = 0.8 - parametr for train data fraction
# BATCH_SIZE = 32
# NUM_WORKERS 
# DEVICE
# LEARNING_RATE = 0.001
# NUM_EPOCHS = 25
# MODEL_PATH = ""

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3.bias.data.zero_()

        if k == 3:
            self.fc3.weight.data.copy_(torch.eye(3).view(9))
        else:
            self.fc3.weight.data.copy_(torch.eye(64).view(64*64))

    def forward(self, x):
        batck_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Identity transformation

        identity = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(batck_size, 1)
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
        self.bn1 = nn.BatckNorm1d(64)
        self.bn2 = nn.BatckNorm1d(128)
        self.bn3 = nn.BatckNorm1d(1024)

        # Classifier
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.3)

        self.bn4 = nn.BatckNormed(512)
        self.bn5 = nn.BatckNormed(256)

    def forward(self, x):
        # x_shape (batch_size, num_points, 3)
        x = x[:, :, :3]

        # Input transform
        input_transform = self.input_transform(x.transform(2, 1))
        x = torch.bmm(x, input_transform)

        # First shared MLP
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transforme
        feature_transform = self.feature_transforme(x)
        x = x.transforme(2, 1)
        x = torch.bmm(x, feature_transform)
        x = x.transpose(2, 1)

        # Second shared MLP
        x = F.relu(self.bn3(self.conv3(x)))

        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        #Classifier
        x = F.relu(self.bn4(seld.fc1(x)))
        x = F.relu(self.bn5(seld.fc2(x)))

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
            # Random rotation around z axis
            theta = np.random.uniform(0, 2 *np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])

            # Random scaling 
            scale = np.random.uniform(0.8, 1.2)

            coords = pointcloud[:, :3]
            colors = pointcloud[:, 3:]

            coords = coords @ rotation_matrix.T
            coords = coords @ scale
            coords += np.clip(0.01 * np.random.randn(*coords.shape), -0.02, 0.02)

            return np.hstack([coords, colors])

        def __getitem__(self, idx):
           pointcloud = self.points[idx].copy()
           label      = self.labels[idx]

           if self.augment:
               pointcloud = self.augment_pointcloud(pointcloud)

           return torch.from_numpy(pointcloud).float(), torch.tensor(label).long()


# Convert mesh to point cloud
def mesh_to_pointcloud(mesh_path, num_points=1024):

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd  = mesh.sample_points_uniformly(number_of_points=num_points)

    # Normalize points to fit in unit sphere
    points   = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)

    points = points - centroid
    furthest_distance = np.max(np.sqrt(np.sum(np.abs(points)**2, axis=1)))
    points = points / (furthest_distance + 1e-8)

    # Get colors
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.once((num_points, 3)) * 0.5

    return np.hstack([points, colors])

# Prepare dataset
def download_dataset(dest_dir="ModelNet10"):
    if os.path.exists(dest_dir):
        print(f" {dest_dir} already exists!")
        return

    url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    zip_path = "ModelNet10.zip"
    
    print("Downloading ModelNet10 ...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192), total=total_size//8192):
            f.write(chunk)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    

def prepare_dataset():
    if os.path.exists(DATA_PATH):
        print("Dataset already prepared")
        return

    print("Preparing dataset...")

    modelnet_path = "ModelNet10"
    if not os.path.exists(modelnet_path):
        print("ModelNet10 dataset path not found")
        return

    classes = [i for i in os.listdir(modelnet_path) if os.pathisdir(os.path.isdir(os.path.join(modelnet_path, i)))]
    classes.sort()
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    all_points = []
    all_labels = []

    for cls in classes:
        cls_idx = class_to_idx[cls]

        print(f"Processing class: {cls}")

        # Process train and test examples 
        for split in ["train", "test"]:
            dir_path = os.path.join(modelnet_path, cls, split)
            files    = [i for i in os.listdir(dir_path) if i.endswith('.off')]

            for i in tqdm(files[:20], desc=f"{split} - {cls}"):
                mesh_path = os.path.join(dir_path, i)

                try:
                    points_colors = mesh_to_pointcloud(mesh_path, NUM_POINTS)
                    all_points.append(points_colors)
                    all_labels.append(cls_idx)

                except Exception as e:
                    print(f"Error processing {mesh_path}: {e}")

    # Save as HDF5
    all_points = np.array(all_points, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)

    with h5py.File(DATA_PATH, 'w') as file:
        file.create_dataset('points', data=all_points)
        file.create_dataset('labels', data=all_labels)
        file.create_dataset('classes', data=np.array(classes, dtype='5'))

    print(f"Dataset prepared with {len(all_points)} samples")


# Training function    
def train():
    # Load dataset
    with h5py.File(DATA_PATH, 'r') as file:
        points = file['points'][:]
        labels = file['labels'][:]
        classes = [name.decode('utf-8') for name in file['classes'][:]]

    # Create dataset 
    dataset = ModelNetDataset(points, labels, augment=True)
    train_size = int(TRAIN_FRAC * len(dataset))
    test_size  = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(traind_dataset, batch_size=BATCH_SIZE, shuffle=True, num_wokers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize model
    model = PointNet(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parametrs(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Traning history
    train_losses, test_losses = [], []
    train_accs, test_accs     = [], []
    best_acc = 0


    print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for data, target in tqdm(train_loader, desc=["Epoch {epoch+1}/{NUM_EPOCHS}"]):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimize.zero_grad()


            # Forward path
            output, input_transform, feature_transform = model(data)
            loss = criterion(output, target)
            loss += 0.001 * feature_transform_regularizer(input_transform)
            loss += 0.001 * feature_transform_regularizer(feature_transform)

            # Backward path
            loss.backward()
            optimizer.step()


            # Statistics
            total_loss += loss.item()
            pred        = output.argmax(dim=1, keepdim=True)
            correct     = pred.eq(target.view(pred)).sum().item()
            total      += target.size(0)

        train_loss = total_loss / len(train_loader) 
        train_acc  = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluation
        model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_targets = [], []


        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output, _, _ = model(data)
                loss         = criterion(output, target)
                total_loss   =loss.item

                pred         = output.argmax(dim=1, keepdim=True)
                correct      = pred.eq(target.view_as(pred)).sum().item()

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        test_loss = total_loss / len(test_loader)
        test_acc  = 100. * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            best_preds = np.array(all_preds)
            best_targets = np.array(all_targets)

        scheduler.step()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig("training_history.png")

    # Plot confusion matrix
    cm = confusion_matrix(best_targets, best_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    print(f"Training complete. Best test accuracy: {best_acc:.2f}%")
    return model, classes, test_dataset

# Visualization function
def visualization(model, test_dataset, classes, num_examples=5):
    model.eval()
    indices = np.random.choice(len(test_dataset), num_examples, replace=False)

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
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

if __name__ == "__main__":

    download_dataset()

    prepare_dataset()

    model, classes, test_dataset = train()

    model.load_state_dict(torch.load(MODEL_PATH))

    visualization(model, test_dataset, classes)



            



            
   
