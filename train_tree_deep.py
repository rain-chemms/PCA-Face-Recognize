#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习人脸特征提取训练脚本（适用于小样本场景）
使用预训练的 FaceNet 或 ResNet50 + ArcFace 风格特征提取器
支持单张/少量图像训练（每人 ≥1 张即可）

依赖：torch, torchvision, facenet_pytorch, opencv-python, numpy, scikit-learn
"""
#产生文件:
# deep_face_model.pkl
# deep_features.npy
# recognize_tree_deep.py在识别是会读取上述的两个文件


import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN

# ======================
# 配置参数（与您原配置一致）
# ======================
DATASET_PATH = 'dataset'  # 可自行修改路径
IMAGE_SIZE = (100, 100)   # 输入图像尺寸
PCA_COMPONENTS = 10       # 保留主成分数量（用于后处理降维，可选）
MODEL_SAVE_PATH = 'deep_face_model.pkl'
FEATURE_SAVE_PATH = 'deep_features.npy'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# ======================
# 数据加载与预处理
# ======================
class FaceDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.image_paths = []
        self.labels = []
        self.label_names = {}
        self.person_ids = {}
        
        label_id = 0
        for person_name in sorted(os.listdir(dataset_path)):
            person_dir = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            self.label_names[label_id] = person_name
            self.person_ids[person_name] = label_id
            
            for img_name in sorted(os.listdir(person_dir)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label_id)
            
            label_id += 1
        
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),  # FaceNet 要求 160x160
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# ======================
# 特征提取器（使用 FaceNet）
# ======================
def get_face_encoder(device='cpu'):
    """加载预训练 FaceNet 编码器"""
    print("正在加载预训练 FaceNet 模型...")
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    resnet.to(device)
    return resnet

# ======================
# 训练函数
# ======================
def train_deep_face_recognizer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载数据集
    dataset = FaceDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    if len(dataset) == 0:
        raise ValueError("未找到任何人脸图像！请检查 DATASET_PATH 目录结构。")
    
    print(f"共加载 {len(dataset)} 张图像，{len(dataset.label_names)} 个人物")
    
    # 2. 初始化编码器
    encoder = get_face_encoder(device)
    
    # 3. 提取所有特征
    features_list = []
    labels_list = []
    
    print("开始提取深度特征...")
    with torch.no_grad():
        for batch_imgs, batch_labels in dataloader:
            batch_imgs = batch_imgs.to(device)
            embeddings = encoder(batch_imgs)
            features_list.append(embeddings.cpu().numpy())
            labels_list.append(batch_labels.numpy())
    
    features = np.vstack(features_list)
    labels = np.hstack(labels_list)
    
    print(f"特征形状: {features.shape} (样本数 × 特征维度)")
    
    # 4. （可选）PCA 降维（适用于后续 KNN 或可视化）
    if PCA_COMPONENTS < features.shape[1]:
        from sklearn.decomposition import PCA
        print(f"执行 PCA 降维到 {PCA_COMPONENTS} 维...")
        pca = PCA(n_components=PCA_COMPONENTS)
        features_reduced = pca.fit_transform(features)
        print(f"PCA 解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        features_reduced = features
        pca = None
    
    # 5. 训练分类器（Nearest Centroid 更适合小样本）
    print("训练最近中心分类器...")
    classifier = NearestCentroid()
    classifier.fit(features_reduced, labels)
    
    # 6. 保存模型
    model_data = {
        'classifier': classifier,
        'pca': pca,
        'label_names': dataset.label_names,
        'feature_dim': features.shape[1],
        'input_size': IMAGE_SIZE,
        'num_classes': len(dataset.label_names),
        'sample_counts': {name: np.sum(labels == lid) for lid, name in dataset.label_names.items()}
    }
    
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    # 7. 保存原始特征（便于调试/增量更新）
    np.save(FEATURE_SAVE_PATH, {
        'features': features,
        'features_reduced': features_reduced,
        'labels': labels,
        'label_names': dataset.label_names
    })
    
    print(f"\n✅ 训练完成！")
    print(f"模型已保存至: {MODEL_SAVE_PATH}")
    print(f"特征已保存至: {FEATURE_SAVE_PATH}")
    print(f"人物列表:")
    for lid, name in dataset.label_names.items():
        count = np.sum(labels == lid)
        print(f"  ID={lid:2d}: {name:<12} | 样本数={count}")
    
    # 8. 简单验证（留一法或交叉验证）
    if len(np.unique(labels)) > 1:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features_reduced, labels, test_size=0.2, stratify=labels, random_state=42
        )
        acc = classifier.score(X_test, y_test)
        print(f"\n📌 验证准确率（20%测试集）: {acc:.2%}")

# ======================
# 使用说明 & 增强技巧（小样本关键！）
# ======================
if __name__ == '__main__':
    print("="*60)
    print("🚀 深度学习人脸训练脚本 (小样本优化版)")
    print("="*60)
    
    # 检查目录结构
    if not os.path.exists(DATASET_PATH):
        print(f"❌ 错误：数据集目录 '{DATASET_PATH}' 不存在！")
        print("请按以下结构组织数据：")
        print("dataset/")
        print("├── ZhangSan/")
        print("│   ├── 1.jpg")
        print("│   └── 2.jpg")
        print("└── LiSi/")
        print("    └── 1.jpg")
        exit(1)
    
    # 检查是否有足够样本
    person_dirs = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    if not person_dirs:
        print("⚠️  警告：未检测到任何人脸子文件夹！")
        exit(1)
    
    # 运行训练
    try:
        train_deep_face_recognizer()
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()