from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

# 사전 학습된 모델 불러오기
input_tensor = Input(shape=(224, 224, 3))
model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

# 모델 Layer 데이터화
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# Layer 추가
x = layer_dict['block5_pool'].output
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

# new model 정의
new_model = Model(inputs=model.input, outputs=x)

# CNN Pre-trained 가중치를 그대로 사용할 때
for layer in new_model.layers[:19]:
    layer.trainable = False

# 모델 컴파일
new_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

# 데이터 경로 설정
train_dir = 'train/images'
test_dir = 'test/images'

# 데이터 증강 (학습용)
# 일반화된 성능을 발휘할 수 있도록 함
train_image_generator = ImageDataGenerator(
    rescale=1./255, # 모든 픽셀 값을 0~255->0~1로 조정
    rotation_range=20,# 이미지가 최대 +-20도 회전 가능하게 설정
    width_shift_range=0.2, # 너비가 최대 +-20%까지 좌우로 이동가능하게 설정
    height_shift_range=0.2, # 높이의 최대 +-20%까지 상하로 이동
    shear_range=0.2, # 20%ㅇ의 전단 변환 적용
    zoom_range=0.2, # 20%까지 확대, 축소
    horizontal_flip=True, # 이미지를 좌우 반전하여 사용
    fill_mode='nearest' # 변환 후 이미지에 생긴 빈 픽셀을 가장 가까운 픽셀 값으로 채워줌
)

# 데이터 증강 (테스트용은 rescale만)
test_image_generator = ImageDataGenerator(rescale=1./255)

# 데이터 구조 생성
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=16,
    directory=train_dir,
    shuffle=True,
    target_size=(224, 224),
    class_mode='binary',
    classes=['no_person', 'person']
)

test_data_gen = test_image_generator.flow_from_directory(
    batch_size=16,
    directory=test_dir,
    target_size=(224, 224),
    class_mode='binary',
    classes=['no_person', 'person']
)

# 콜백 설정
checkpoint = ModelCheckpoint(
    'best_model_vgg16.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 클래스 가중치 설정
# class_weight = {0: 1, 1: 3}  # 'no_person': 1, 'person': 3 비율로 설정
# 이렇게 학습하니 no_person 정확도만 올라가고 사람은 아예 인식못함..

# 모델 학습
history = new_model.fit(
    train_data_gen,
    epochs=100,
    validation_data=test_data_gen,
    callbacks=[checkpoint],
)

# 모델 저
new_model.save("newVGG16.h5")

# 최종 결과 리포트
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# 저장된 최적의 모델 불러오기
from keras.models import load_model

best_model = load_model("newVGG16.h5")
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
