import cv2
import numpy as np
import pickle

#该脚本会加载同级目录下的 
# pca_model.pkl
# knn_model.pkl 
# label_names.pkl 
#三个文件,请务必提前准备好

# 加载OpenCV自带的人脸检测器（用于在画面中框出人脸）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_models():
    try:
        with open('pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)
        with open('knn_model.pkl', 'rb') as f:
            knn = pickle.load(f)
        with open('label_names.pkl', 'rb') as f:
            label_names = pickle.load(f)
        print("模型加载成功！")
        return pca, knn, label_names
    except FileNotFoundError:
        print("错误：找不到模型文件！请先运行 train.py 进行训练。")
        return None, None, None

def recognize():
    pca, knn, label_names = load_models()
    if pca is None:
        return

    cap = cv2.VideoCapture(0) # 打开默认摄像头
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    print("实时识别已启动，按 'q' 键退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转为灰度图用于人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # 截取人脸区域并预处理
            face_roi = gray[y:y+h, x:x+w]
            try:
                face_resized = cv2.resize(face_roi, (100, 100))
                face_vector = face_resized.flatten().reshape(1, -1)
                
                # PCA降维 + KNN预测
                face_pca = pca.transform(face_vector)
                label_id = knn.predict(face_pca)[0]
                name = label_names[label_id]
                
                # 在画面上画出人脸框和名字
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                pass # 忽略识别过程中的偶发错误

        cv2.imshow('PCA Real-time Face Recognition', frame)
        
        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    recognize()