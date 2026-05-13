import cv2
import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# 配置参数
DATASET_PATH = 'dataset' # 可自行修改文件夹路径
#文件夹内部结构如下
#├── DATASET_PATH/                 # 存放用于训练的人脸图像集
#│   ├── ZhangSan/            # 以人名命名的子文件夹
#│   │   ├── 1.jpg
#│   │   ├── 2.jpg
#│   │   └── ... (建议每人准备10-20张不同角度/表情的照片,不少于PCA_COMPONENTS)
#│   └── LiSi/
#│       ├── 1.jpg
#│       └── ...
IMAGE_SIZE = (100, 100)  # 统一将人脸缩放到100x100像素
PCA_COMPONENTS = 10      # PCA降维保留的主成分个数,训练时需要保证图片数量大于当前数值

# 训练完成后
# 会在train.py的同级目录中生成
#   knn_model.pkl
#   pca_model.pkl
#   label_names.pkl
# 三个文件
# 这三个文件再运行recognize.py时会自动加载

def load_dataset():
    print("正在加载数据集...")
    faces = []
    labels = []
    label_names = {}
    label_id = 0
    
    # 遍历dataset文件夹下的所有人名子文件夹
    for name in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, name)
        if os.path.isdir(folder_path):
            label_names[label_id] = name
            print(f"正在处理人物：{name} (标签ID: {label_id})")
            
            # 读取每个人的所有照片
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # 转为灰度图
                if img is None:
                    continue
                
                # 统一图片尺寸并展平成一维向量
                img_resized = cv2.resize(img, IMAGE_SIZE)
                faces.append(img_resized.flatten())
                labels.append(label_id)
            label_id += 1   
    return np.array(faces), np.array(labels), label_names

def train_and_save():
    # 1. 加载数据
    X, y, label_names = load_dataset()
    if len(X) == 0:
        print("错误：未在 dataset 文件夹中找到有效图片！请检查路径和图片格式。")
        return

    # 2. 使用PCA进行降维
    print(f"正在进行PCA降维，保留 {PCA_COMPONENTS} 个主成分...")
    pca = PCA(n_components=PCA_COMPONENTS, whiten=True) # whiten=True进行白化处理，提升KNN效果
    X_pca = pca.fit_transform(X)

    # 3. 使用KNN进行分类训练
    print("正在训练KNN分类器...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_pca, y)

    # 4. 保存模型和标签
    with open('pca_model.pkl', 'wb') as f:
        pickle.dump(pca, f)
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(knn, f)
    with open('label_names.pkl', 'wb') as f:
        pickle.dump(label_names, f)

    print("训练完成！模型已保存为 pca_model.pkl, knn_model.pkl 和 label_names.pkl")

if __name__ == '__main__':
    train_and_save()