import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

def collate_fn(batch): # 데이터 로더에서 배치로 묶을 때 사용. 여기서는 이미지와 바운딩 박스 묶기 위해.
    return tuple(zip(*batch))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    # 바운딩 박스 색상 설정
    color = (255, 0, 0)

    # 앉은 사진 삭제하기!
    # 이미지와 라벨 파일 경로 설정
    image_dir = "train/image/path"
    label_dir = "train/label/path"
    # test_image_dir = "test/image/path"

    # 하이퍼파라미터 설정
    num_epochs = 100
    learning_rate = 0.001 # 학습률
    batch_size = 4 # 배치 크기
    target_size = (640, 640)
    num_classes = 2

    # 데이터셋에서 바운딩 박스와 라벨을 저장할 리스트
    all_bboxes = []
    all_labels = []

    # 이미지 파일 목록 가져오기
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    n = 0

    for idx, img_file in enumerate(image_files): # 이미지 및 바운딩 박스 데이터 로드
        print(f"Processing image {idx + 1}/{len(image_files)}: {img_file}")
        img_path = os.path.join(image_dir, img_file)# 이미지 경로+이미지 이름 => 전체 경로
        img0 = cv2.imread(img_path)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")

        bbox0 = []  # 바운딩 박스 원본 좌표
        labels = []  # 라벨 저장 리스트

        # 라벨 파일에서 바운딩 박스 읽기
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines() #파일의 모든 줄 읽어 lines 리스트에 저장
                for line in lines:
                    line = line.strip() # 양쪽 끝 공백 제거
                    if line == "":
                        continue
                    try:
                        class_id, x, y, width, height = map(float, line.split())
                    except ValueError:
                        print(f"Invalid line in {label_path}: {line}")
                        continue

                    x1 = int((x - width / 2) * img0.shape[1])
                    y1 = int((y - height / 2) * img0.shape[0])
                    x2 = int((x + width / 2) * img0.shape[1])
                    y2 = int((y + height / 2) * img0.shape[0])

                    # 바운딩 박스 그리기
                    cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)

                    # bbox0에 추가
                    bbox0.append([x1, y1, x2, y2])
                    # labels에 클래스 추가
                    labels.append(int(class_id))

        if n < 5:
            # 이미지 시각화
            plt.imshow(img0)
            plt.title(img_file)
            plt.axis('off')
            plt.show()
            n += 1

        # 크기 조정
        img_resized = cv2.resize(img0, dsize=target_size, interpolation=cv2.INTER_CUBIC)

        # bbox0과 labels 배열로 변환
        bbox0 = np.array(bbox0)
        labels = np.array(labels)

        # 바운딩 박스 좌표 변환
        ratioList = [target_size[0] / img0.shape[1], target_size[1] / img0.shape[0], target_size[0] / img0.shape[1], target_size[1] / img0.shape[0]]
        bbox = []

        for box in bbox0:
            box = [int(a * b) for a, b in zip(box, ratioList)]
            bbox.append(box)

        bbox = np.array(bbox)

        # bbox와 labels 저장
        all_bboxes.append(bbox)
        all_labels.append(labels)

    # 모델 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

    # ROI 헤드 수정 (클래스 수를 2로 설정)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 백본 레이어도 학습 가능하게 설정
    for param in model.backbone.parameters():
        param.requires_grad = True

    model.to(device)

    # 데이터 로더 생성
    train_data = []
    for img_file, bboxes, lbls in zip(image_files, all_bboxes, all_labels):
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])(img).to(device)

        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32).to(device),
            "labels": torch.tensor(lbls, dtype=torch.int64).to(device)
        }
        train_data.append((img_tensor, target))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # Optimizer 설정
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    # 학습 루프
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        loss_classifier_epoch = 0
        loss_box_reg_epoch = 0

        for images, targets in train_loader:
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            loss_classifier_epoch += loss_dict['loss_classifier'].item()
            loss_box_reg_epoch += loss_dict['loss_box_reg'].item()

        # 에포크마다 한 번씩만 출력
        avg_total_loss = total_loss / len(train_loader)
        avg_loss_classifier = loss_classifier_epoch / len(train_loader)
        avg_loss_box_reg = loss_box_reg_epoch / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {avg_total_loss:.4f}, "
              f"Classification Loss: {avg_loss_classifier:.4f}, Box Regression Loss: {avg_loss_box_reg:.4f}")

    print("Training complete.")

    # 모델 저장
    model_path = "faster_rcnn_model_resnet101.pth"
    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)

if __name__ == '__main__':
    main()
