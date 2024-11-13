import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.ops import box_iou

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # OpenMP 오류 해결을 위한 설정

def main():
    # 모델 평가 및 시각화
    # 이미지와 라벨 파일 경로 설정
    test_image_dir = "test/image/path"
    label_dir = "test/label/path"

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 설정 및 로드
    num_classes = 2
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)  # MobileNet을 사용
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 학습된 가중치 로드
    model.load_state_dict(torch.load('faster_rcnn_mobilenet_v3.pth', map_location=device))
    model.to(device)
    model.eval()

    # 평가 변수 초기화
    all_pred_bboxes = []  # 예측 바운딩 박스
    all_pred_labels = []  # 예측 라벨
    all_true_bboxes = []  # 실제 바운딩 박스
    all_true_labels = []  # 실제 라벨
    all_pred_scores = []  # 예측 점수

    # 테스트 이미지 목록 가져오기
    test_image_files = sorted(f for f in os.listdir(test_image_dir) if f.endswith('.jpg'))

    # 이미지 평가
    for img_file in test_image_files:
        img_path = os.path.join(test_image_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img).to(device)

        with torch.no_grad():
            predictions = model([img_tensor])

        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()
        pred_labels = predictions[0]['labels'].cpu().numpy()

        # 예측 결과를 필터링 (신뢰도 0.5 이상)
        valid_indices = np.where(pred_scores > 0.5)[0]
        all_pred_bboxes.append(pred_boxes[valid_indices])
        all_pred_labels.append(pred_labels[valid_indices])
        all_pred_scores.append(pred_scores[valid_indices])

        # 실제 라벨 파일에서 바운딩 박스 읽기
        true_label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
        true_bboxes = []
        true_labels = []

        if os.path.exists(true_label_path):
            with open(true_label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id, x, y, width, height = map(float, line.split())
                    x1 = int((x - width / 2) * img.width)
                    y1 = int((y - height / 2) * img.height)
                    x2 = int((x + width / 2) * img.width)
                    y2 = int((y + height / 2) * img.height)
                    true_bboxes.append([x1, y1, x2, y2])
                    true_labels.append(int(class_id))

        all_true_bboxes.append(np.array(true_bboxes))
        all_true_labels.append(np.array(true_labels))

    # mAP 계산 함수
    def calculate_map(predictions, ground_truths, iou_threshold=0.5):
        all_ap = []
        all_ious = []

        for (boxes, scores, pred_labels), (gt_boxes, gt_labels) in zip(predictions, ground_truths):
            if len(boxes) == 0 or len(gt_boxes) == 0:
                continue
            ious = box_iou(torch.tensor(boxes), torch.tensor(gt_boxes))
            for pred_idx in range(len(boxes)):
                if scores[pred_idx] < 0.5:
                    continue
                iou = ious[pred_idx]
                matched_gt_indices = (iou > iou_threshold).nonzero(as_tuple=True)[0].tolist()
                if matched_gt_indices:
                    all_ap.append(1)
                    all_ious.append(iou[matched_gt_indices].max().item())
                else:
                    all_ap.append(0)

        return sum(all_ap) / len(all_ap) if all_ap else 0, all_ious

    # Precision, Recall 계산 함수
    def calculate_precision_recall(all_predictions, all_ground_truths, iou_threshold=0.5):
        tp, fp, fn = 0, 0, 0

        for (boxes, scores, pred_labels), (gt_boxes, gt_labels) in zip(all_predictions, all_ground_truths):
            if len(boxes) == 0 and len(gt_boxes) == 0:
                continue

            if len(gt_boxes) == 0:
                fp += len(boxes)  # GT가 없는 경우 모든 예측은 FP로 처리
                continue

            if len(boxes) == 0:
                fn += len(gt_boxes)  # 예측이 없는 경우 모든 GT는 FN으로 처리
                continue

            ious = box_iou(torch.tensor(boxes), torch.tensor(gt_boxes))
            matched_gt = set()  # 이미 매칭된 GT를 추적하기 위한 집합

            for pred_idx in range(len(boxes)):
                if scores[pred_idx] < 0.5:
                    continue
                iou = ious[pred_idx]
                matched_gt_indices = (iou > iou_threshold).nonzero(as_tuple=True)[0].tolist()

                if matched_gt_indices:
                    for gt_idx in matched_gt_indices:
                        if gt_idx not in matched_gt:
                            tp += 1
                            matched_gt.add(gt_idx)
                            break
                else:
                    fp += 1

            fn += len(gt_boxes) - len(matched_gt)  # 매칭되지 않은 GT는 FN으로 처리

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return precision, recall

    # mAP 계산
    map_score, all_ious = calculate_map(zip(all_pred_bboxes, all_pred_scores, all_pred_labels),
                                        zip(all_true_bboxes, all_true_labels))
    print(f"mAP: {map_score:.4f}")

    # Precision, Recall 계산
    precision, recall = calculate_precision_recall(zip(all_pred_bboxes, all_pred_scores, all_pred_labels),
                                                   zip(all_true_bboxes, all_true_labels))
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

    # 결과 시각화
    for img_file, pred_bboxes, true_bboxes in zip(test_image_files, all_pred_bboxes, all_true_bboxes):
        img_path = os.path.join(test_image_dir, img_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 예측 바운딩 박스 그리기
        for box in pred_bboxes:
            cv2.rectangle(img_rgb, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # 실제 바운딩 박스 그리기
        for bbox in true_bboxes:
            cv2.rectangle(img_rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        # 결과 시각화
        plt.imshow(img_rgb)
        plt.title(img_file)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
