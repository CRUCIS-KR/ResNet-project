# 테스트 코드
# 224 사이즈 이미지 식별 결과 확인

import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
import os
from PIL import Image

# 장치 사용
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_name = ['animal', 'object', 'person']  # 예측 클래스 이름

# 모델 불러오기
model = models.resnet18(weights=None) # resnet50으로 테스트 시 18을 50으로 변경
model.fc = nn.Sequential(torch.nn.Dropout(0.3), nn.Linear(model.fc.in_features, 3))
model.load_state_dict(torch.load("ResNet18.pt", map_location=DEVICE, weights_only=True)) # resnet50으로 테스트 시 18을 50으로 변경
model.to(DEVICE)
model.eval()

# 이미지 예측 및 시각화
path = sorted(os.listdir('test_image'))[:10]  # 테스트 이미지 10장 로드, 폴더에 정렬되어 있는 순서, 이미지 10장 이상 혹은 이하 사용 시 10을 원하는 숫자로 변경, 다만 사전에 제공된 이미지는 12장
plt.figure(figsize=(15, 8)) # 테스트 이미지가 많아서 공간이 부족하면 사이즈를 수정하면 된다, 기존은 10장에 최적화 된 사이즈이다, figsize(가로, 세로), 실 사이즈는 길이X100px

for i, name in enumerate(path): # 10장 예측 및 시각화
    img_path = os.path.join('test_image', name) # 이미지 경로
    img = Image.open(img_path).convert("RGB") # RGB로 이미지 객체 생성

    input = transform(img).unsqueeze(0).to(DEVICE) # 입력 이미지 텐서 변환 및 배치 추가

    with torch.no_grad():
        pred = model(input).argmax(1).item() # 예측
        pred_label = class_name[pred] # 예측 값 클래스 이름으로 변환

    # 시각화
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.xlabel(f"Pred: {pred_label}", fontsize=10)
    plt.xticks([])
    plt.yticks([])

# 결과 출력 및 저장
os.makedirs("model_info", exist_ok=True)
plt.tight_layout()
plt.savefig("model_info/테스트 시연.png") # 필요 시 파일 이름을 바꾸어 구분 처리, ResNet18.pt로 테스트 후 ResNet50.pt로 테스트 시 파일이 덮어쓰기 됨
plt.show()