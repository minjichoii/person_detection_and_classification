import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# 모델 로드
model = load_model('newVGG16.h5', compile=False)

# 테스트 데이터셋 준비
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    'val/images',  # 테스트 데이터 경로
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# 예측 수행
# steps = np.floor(test_generator.samples / test_generator.batch_size)

y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()  # 0.5 기준으로 이진 분류

# 실제 라벨 가져오기
y_true = test_generator.classes

# 예측 길이를 실제 라벨 길이에 맞추기
y_pred_classes = y_pred_classes[:len(y_true)]

# 길이 확인
print("Length of y_true:", len(y_true))
print("Length of y_pred_classes:", len(y_pred_classes))
print("Number of samples in test_generator:", test_generator.samples)


# 성능 지표 계산 및 출력
print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred_classes))

print('Classification Report')
target_names = ['no_person', 'person']  # 클래스 이름 정의
print(classification_report(y_true, y_pred_classes, target_names=target_names))
