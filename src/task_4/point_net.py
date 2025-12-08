import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import open3d as n3d

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

    def(self, x):
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
    
    

        


           


            
            
    
        

        


        
        
