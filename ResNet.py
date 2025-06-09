# RetNet18을 이용한 전이학습
# 사람, 동물, 사물을 분류하는 모델 구현
# 학습 곡선, 혼동 행렬, 평가 지표 시각화
# 모델: resnet18
# 데이터: Open Image V7, 41000개, 8:2 분할
# 옵티마이저: AdamW
# 스케줄러: CosineAnnealingLR
# 손실 함수: Focal loss
# layer4, fc 에 대한 파인 튜닝 적용

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    print(torch.cuda.is_available()) # 만약 false라면 pytorch 라이브러리를 설치할때 현재 GPU의 CUDA 버전을 확인하고 해당 버전을 명시해서 재설치해야 한다.

    # 장치 지정
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # 하이퍼파라미터
    BATCH_SIZE = 128 
    EPOCH = 50
    NUM_WORKERS = 8 # 멀티프로세싱 파라미터, CPU 사양에 따라 조절하여 사용
    LR = 1e-5
    EARLY_STOP = 3 # 학습 조기 종료 지점 
    
    # 데이터 셋 로드
    train_loader = DataLoader(datasets.ImageFolder('dataset/train', transforms.Compose([ # 학습 데이터
        transforms.RandomHorizontalFlip(), # 이미지 반전 랜덤 적용
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), # 밝기, 대비, 채도, 색조 조정
        transforms.ToTensor(), # 타입 변환
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])), # 정규화 
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS) # 배치 사이즈 적용, 순서 랜덤, 병렬처리
    
    val_loader = DataLoader(datasets.ImageFolder('dataset/val', transforms.Compose([ # 평가 데이터
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])), 
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    class_name = train_loader.dataset.classes # 레이블 이름

    # 모델 
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # resnet18 로드, pretrained = true, DEFAULT = 'IMAGENET1K_V1', resnet50 사용 시 18부분을 50으로 변경
    for name, param in model.named_parameters():
        param.requires_grad = 'layer4' in name or 'fc' in name # layer4와 완전연결층에 대해서만 학습, 나머지 층은 학습하지 않음 (freeze)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 3)) # 완전연결층 드롭아웃 적용, 3개의 클래스 구분으로 변경 
    model.to(DEVICE) # 장치 사용

    # 손실 함수 Focal Loss 정의
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2): 
            super().__init__() # 생성자
            self.alpha, self.gamma = alpha, gamma # alpha = 클래스별 중요도, gamma = 강조 정도
            self.ce = nn.CrossEntropyLoss(reduction='none') # 크로스엔트로피 객체 생성, 개별 계산 적용

        def forward(self, inputs, targets): 
            logpt = self.ce(inputs, targets) # 크로스엔트로피 계산 -log(pt)
            pt = torch.exp(-logpt) # pt 계산
            return (self.alpha * (1 - pt) ** self.gamma * logpt).mean() # Focal Loss = -α⋅(1−pt)^γ⋅log(pt) 의 평균값
    
    criterion = FocalLoss()

    # 옵티마이저 및 스케줄러 정의
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=LR) # freeze 되지 않은 층에 대해서만 적용, 학습률, 가중치 감쇠 0.00001 설정
    scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6) # 코사인함수에 따라 학습률이 조정, 최소 0.000001 까지 수렴, 사이클 = 25 에포크

    # 학습 및 평가
    BEST_LOSS = float('inf') # 최고 손실율 기록, float('inf')는 양의 무한대를 의미, 초기에 무조건 손실율 기록하기 위함
    COUNT = 0 # 조기 종료 계산
    train_losses, val_losses = [], [] # 손실률 모음
    train_accuracies, val_accuracies = [], [] # 정확도 모음

    for epoch in range(EPOCH): # 50 에포크 진행
        model.train() # 학습 모드
        train_loss, train_correct = 0, 0 # 학습 데이터에 대한 손실, 정답
        for data, target in train_loader: # 학습 데이터 적용
            data, target = data.to(DEVICE), target.to(DEVICE) # 장치 사용
            optimizer.zero_grad() # 기울기 초기화
            output = model(data) # 출력
            loss = criterion(output, target) # 손실 함수 계산
            loss.backward() # 손실률에 대한 역전파 적용 
            optimizer.step() # 옵티마이저 업데이트
            train_loss += loss.item() * data.size(0) # 손실률 저장, 실제 배치 사이즈를 곱하여 정확도를 높임
            train_correct += output.argmax(1).eq(target).sum().item() # 정답과 예측값 비교 후 맞춘 개수 저장

        train_loss /= len(train_loader.dataset) # 손실률 평균값
        train_acc = train_correct / len(train_loader.dataset) # 정확도 계산

        model.eval() # 평가 모드
        val_loss, val_correct = 0, 0 # 평가 데이터에 대한 손실, 정답 
        with torch.no_grad(): # 기울기 계산 하지 않음
            for data, target in val_loader: # 평가 데이터 적용
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                val_correct += output.argmax(1).eq(target).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        scheduler.step() # 스케줄러 업데이트

        print(f"[{epoch+1:02d}] 학습: [Loss: {train_loss:.4f}, Acc: {train_acc:.4f}] | 평가: [Loss: {val_loss:.4f}, Acc: {val_acc:.4f}]") # 에포크 별 학습 및 평가 상황 출력
        
        # 손실률, 정확도 저장
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_loss < BEST_LOSS: # 최고 손실률 갱신
            BEST_LOSS, COUNT = val_loss, 0 # 갱신 및 초기화
            torch.save(model.state_dict(), 'ResNet18.pt') # 최고 모델 저장, resnet50 사용 시 18을 50으로 변경
        else: # 손실률 개선이 없음
            COUNT += 1 
            if COUNT >= EARLY_STOP: # 조기 종료 발생
                print("학습 조기 종료")
                break

    # 최고 성능 모델 평가
    model.load_state_dict(torch.load('ResNet18.pt', weights_only=True)) # resnet50 사용 시 ResNet50.pt로 수정한다.
    model.eval()
    all_pred, all_true = [], []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(DEVICE)
            output = model(data)
            pred = output.argmax(1).cpu() 
            all_pred.extend(pred.tolist())
            all_true.extend(target.tolist())

    os.makedirs("model_info", exist_ok=True) # 시각화 자료 저장 폴더 생성

    # 학습 곡선 생성 및 저장
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss Curve")
    plt.xlabel("EPOCH")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.title("Accuracy Curve")
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("model_info/학습곡선.png") # 필요 시 파일 이름을 바꾸어 구분 처리, ResNet18 확인 후 ResNet50 실행 시 파일이 덮어쓰기 됨
    plt.show()

    # 혼동 행렬 생성 및 저장 (최고 성능 모델 이용)
    cm = confusion_matrix(all_pred, all_true)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_name, yticklabels=class_name)
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("model_info/혼동행렬.png") # 필요 시 파일 이름을 바꾸어 구분 처리, ResNet18 확인 후 ResNet50 실행 시 파일이 덮어쓰기됨
    plt.show()

    # 재현율 및 F1-Score 지표 표시 및 저장
    report = classification_report(all_pred, all_true, target_names=class_name) # 각 클래스 별 평가 지표 표시
    print(report)
    with open("model_info/모델 성능 지표.txt", "w", encoding="utf-8") as f: # 필요 시 파일 이름을 바꾸어 구분 처리, ResNet18 확인 후 ResNet50 실행 시 파일이 덮어쓰기됨
        f.write(report)