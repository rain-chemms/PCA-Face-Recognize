#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习实时人脸检测与识别脚本
基于 OpenCV 和 FaceNet 模型，支持摄像头实时识别

依赖：torch, torchvision, facenet_pytorch, opencv-python, numpy, scikit-learn
"""

import cv2
import numpy as np
import pickle
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
import sys
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

# ======================
# 配置参数
# ======================
MODEL_SAVE_PATH = 'deep_face_model.pkl'  # 深度学习模型路径
CAMERA_INDEX = 0  # 摄像头索引 (通常为 0)
CONFIDENCE_THRESHOLD = 0.7  # 识别置信度阈值 (0.0~1.0)
WINDOW_NAME = "Tree_Deep_Recognize"

# 预处理变换（与训练时保持一致）
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Lambda(fixed_image_standardization)  # FaceNet 标准化
])

# ======================
# 加载人脸检测器 (OpenCV Haar Cascade)
# ======================
def load_face_detector():
    """加载 OpenCV 内置的人脸检测器"""
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        print(f"❌ 人脸检测器文件不存在: {face_cascade_path}")
        sys.exit(1)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    return face_cascade

# ======================
# 加载深度学习模型
# ======================
def load_deep_model():
    """加载训练好的深度学习模型"""
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"❌ 模型文件不存在: {MODEL_SAVE_PATH}")
        print("请先运行 train_deep.py 训练模型！")
        sys.exit(1)
    
    with open(MODEL_SAVE_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"✅ 成功加载模型，共 {model_data['num_classes']} 个类别")
    return model_data

# ======================
# 提取人脸特征（深度学习）
# ======================
def extract_deep_features(img, encoder, device='cpu'):
    """
    使用深度学习模型提取人脸特征
    Args:
        img: BGR 彩色图像 (H, W, C)
        encoder: FaceNet 编码器
        device: 计算设备
    Returns:
        特征向量 (1, feature_dim)
    """
    if img is None or img.size == 0:
        return None
    
    # 转换为 RGB 并应用预处理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 应用预处理
    img_tensor = transform(img_rgb)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    
    # 提取特征
    with torch.no_grad():
        embedding = encoder(img_tensor)
    
    return embedding.cpu().numpy()

# ======================
# 主识别函数
# ======================
def main():
    # 1. 加载模型和检测器
    print("🔍 正在初始化...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    face_cascade = load_face_detector()
    model_data = load_deep_model()
    
    # 初始化 FaceNet 编码器
    encoder = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
    
    # 2. 打开摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {CAMERA_INDEX}")
        sys.exit(1)
    
    print(f"🎥 摄像头已打开，按 'q' 退出")
    
    # 3. 主循环
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面")
            break
        
        # 检测人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # 处理每个人脸
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_roi = frame[y:y+h, x:x+w]
            
            # 提取深度特征
            features = extract_deep_features(face_roi, encoder, device)
            if features is None:
                continue
            
            # 如果模型使用了 PCA，需要转换特征
            if model_data['pca'] is not None:
                features = model_data['pca'].transform(features)
            
            # 预测
            classifier = model_data['classifier']
            prediction = classifier.predict(features)[0]
            distances = classifier.decision_function(features)[0]
            
            # 计算置信度
            max_dist = np.max(distances)
            min_dist = np.min(distances)
            # 归一化置信度 (0~1)，越大越可信
            confidence = (max_dist - min_dist) / (max_dist + 1e-8)  # 防止除零
            confidence = np.clip(confidence, 0.0, 1.0)
            
            # 获取姓名
            if prediction in model_data['label_names']:
                name = model_data['label_names'][prediction]
            else:
                name = "未知"
            
            #if isinstance(name, bytes):
            #    name = name.decode('utf-8', errors='replace')
            #elif not isinstance(name, str):
            #    name = str(name)
            
            #print("已识别人脸:"+name)
            # 根据置信度设置颜色和显示文本
            if confidence >= CONFIDENCE_THRESHOLD:
                color = (0, 255, 0)  # 绿色 - 高置信度
                text = f"{name} ({confidence:.2f})"
            else:
                color = (0, 0, 255)  # 红色 - 低置信度或未知
                text = f"UnKnow ({confidence:.2f})"
            
            #text = text.encode('utf-8', errors='replace').decode('utf-8')
            # 绘制矩形框和文字
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame, text, 
                (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, color, 2
            )
        
        # 显示帧
        cv2.imshow(WINDOW_NAME, frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 4. 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("👋 识别结束")

if __name__ == '__main__':
    print("="*50)
    print("🚀 深度学习实时人脸检测识别")
    print("="*50)
    main()