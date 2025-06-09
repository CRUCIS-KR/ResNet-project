# 데이터 가공
# OpenCV 라이브러리를 사용하여 데이터를 읽어와 원본 이미지에서 필요한 부분만 잘라 224x224 px크기로 사이즈 변환
# 객체 식별이 가능한 크기만 추출하며 비율을 유지하여 변환, 나머지 빈 공간은 평균값 계산으로 이미지에 어울리는 배경 임의 생성
# 20000장 이상의 데이터를 순차적으로 처리하면 시간이 많이 소요되기 때문에 멀티프로세싱을 이용하여 단축

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 환경에 맞게 수정 필요
data = Path("") # 다운로드 한 이미지가 있는 데이터 폴더, data_find.py를 통해 가져온 이미지 데이터 경로 지정
result = Path("person") #Path("animal") #Path("object"), 가져온 이미지 레이블에 따라 수정, 가공된 이미지가 저장되는 경로, 현재 폴더에 저장
annotation = Path("?/detections.csv") # 이미지 메타 데이터 (이미지 고유 ID, 객체 클래스, 객체 바운딩 박스 좌표), ?는 data_find.py를 통해 생성된 데이터 폴더 안에 있는 detections.csv가 있는 경로

size = 224
m_box = 0.01 # 최소 바운딩 박스 정규화 크기 정의

# 추출 클래스 고유 번호 목록 - csv 파일 참고, 다운로드 한 데이터 폴더에 클래스 고유 ID를 명시한 csv가 있다.
# 만약 data_find.py 에서 다운로드하는 클래스를 변경했다면 추가하거나 삭제한 항목에 대한 ID를 확인하고 리스트안의 내용을 수정해야 한다.
p_class = {"/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655"} # 사람

a_class = {"/m/0czz2", "/m/01h44", "/m/01dws", "/m/01yrx", "/m/01xq0k1", "/m/09b5t", "/m/09kx5", "/m/0bt9lr", "/m/09csl", "/m/0f6wt", "/m/0306r", "/m/03fwl", "/m/0dbvp", "/m/03qrc", 
           "/m/03k3r", "/m/012074", "/m/0c29q", "/m/04rmv", "/m/0dbzx", "/m/0cn6p", "/m/09d5_", "/m/0gv1x", "/m/068zj", "/m/06mf6", "/m/06j2d", "/m/07final_imgp", "/m/078jl", "/m/071qp", 
           "/m/0dftk", "/m/07dm6", "/m/09dzg"} # 동물
 
o_class = {"/m/0cmf2", "/m/012n7d", "/m/01j51", "/m/01btn", "/m/02zn6n", "/m/0199g", "/m/019jd", "/m/025dyy", "/m/0584n8", "/m/01bjv", "/m/0ph39", "/m/0k4j", "/m/02gzp", "/m/02068x", 
           "/m/0gxl3", "/m/04ctx", "/m/01prls", "/m/01c648", "/m/04_sv", "/m/02rdsp", "/m/0k1tl", "/m/05gqfk", "/m/09rvcxw", "/m/06msq", "/m/076bq", "/m/06nrc", "/m/04vv5k", 
           "/m/01x3jk", "/m/03kt2w", "/m/074d1", "/m/01s55n", "/m/07cmd", "/m/07r04", "/m/0h2r6"} # 사물

# 다운로드한 이미지 파일에서 ID만 추출, 확장자 제거
download = {p.stem for p in data.glob("*")}

# 모든 Openimage V7 데이터에 대한 메타데이터에서 원하는 항목만 필터링, pandas 데이터 프레임으로 정렬
frame = pd.read_csv(annotation)
frame = frame[frame["ClassName"].isin(o_class)] # 필요한 클래스 번호만 필터링, isin(안에는 다운로드한 이미지 레이블에 맞는 ID 리스트를 작성, p_class, o_class, a_class)
frame = frame[frame["ImageID"].isin(download)] # 다운로드한 이미지만 필터링
matadata = frame.groupby("ImageID") # 이미지 ID로 정렬
image_id = list(matadata.groups.keys()) # 정렬된 메타데이터에서 ID만 추출

# 박스 크기 계산
def box_size(frame):
    return (frame["XMax"] - frame["XMin"]) * (frame["YMax"] - frame["YMin"]) # 박스에 대한 X, Y 정규화 좌표(0 ~ 1)가 detections.csv에 명시되어 있음

# 이미지 배경 색깔 계산
def background_color(img):
    xy = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]] # 이미지의 꼭짓점 좌표 (y, x)에 대한 RGB 색상 [B, G, R] 리스트트
    return tuple(np.median(xy, axis=0).astype(int)) # x, y 좌표의 RGB 색상의 평균을 계산해서 인트형으로 저장

# 추출 및 가공
def image_change(image_id):
    road = matadata.get_group(image_id).copy() # 이미지 ID 불러오기
    road.loc[:, "area"] = box_size(road) # 계산된 박스 크기 데이터 프레임에 추가

    # 넓이 비교
    bigone = road.loc[road["area"].idxmax()] # 추가한 넓이 데이터에서 가장 큰 값만 저장
    if bigone["area"] < m_box:
        return f"{image_id}: 객체 크기 문제 발생 (area={bigone['area']})" # 박스 크기가 해상도 비율에 비해 너무 작으면 객체 식별이 어려우니 추출하지 않음

    # 이미지 파일 찾기
    img_path = data/f"{image_id}.jpg" # 이미지 파일 경로 정의
    if not img_path.exists(): # 다운로드한 데이터가 있는지 확인
        return f"{image_id}: 이미지 없음"
    
    # 이미지 불어오기
    img = cv2.imread(str(img_path)) # 해당 경로에 있는 이미지 로드
    if img is None: # 읽어오지 못하면 표시
        return f"{image_id}: 이미지 불러오기 실패" 

    # 실제 픽셀 비교
    h, w = img.shape[:2] # 이미지 실제 크기 슬라이싱
    # 박스의 x, y 정규화 좌표를 실제 픽셀 크기로 변환
    x1 = int(bigone["XMin"] * w)
    x2 = int(bigone["XMax"] * w)
    y1 = int(bigone["YMin"] * h)
    y2 = int(bigone["YMax"] * h)
    cut = img[y1:y2, x1:x2] # 원본 이미지에서 해당 범위만큼 추출
    ch, cw = cut.shape[:2] # 추출 이미지 실제 크기 슬라이싱
    if ch < 30 or cw < 30: # 30px 보다 작으면 식별이 어려우니 학습 데이터로 사용하지 않음
        return f"{image_id}: 객체 크기 문제 발생 (px={cw}x{ch})"

    # 배경 생성, 이미지 사이즈 변환
    color = background_color(img) # 이미지 배경 RGB 평균값 추출
    final_img = np.full((size, size, 3), color, dtype=np.uint8) # 224x224px의 RGB 색상을 사용하는 배경 생성, 8비트 포맷
    ratio = min(size / ch, size / cw) # 사이즈 변환 시 추출 된 이미지의 기존 비율은 유지하기 위해 비교
    cut_resize = cv2.resize(cut, (int(cw * ratio), int(ch * ratio))) # 추출 된 이미지를 계산된 비율에 맞게 사이즈 변환
    
    # 랜덤 위치 생성, 이미지 삽입
    rh, rw = cut_resize.shape[:2] # 변환 이미지 실제 크기 슬라이싱 
    # 224x224 크기 안에 이미지를 삽입 할 수 있는 랜덤한 좌표 생성
    rx = np.random.randint(0, size - rw + 1)
    ry = np.random.randint(0, size - rh + 1)
    final_img[ry:ry+rh, rx:rx+rw] = cut_resize # 해당 위치에 변환된 이미지 삽입

    # 저장
    result.mkdir(parents=True, exist_ok=True) # 이미지 저장할 폴더 생성 있다면 무시 
    final_path = result/f"{image_id}.jpg" # 최종 이미지 경로 정의
    final = cv2.imwrite(str(final_path), final_img) # 최종 경로에 저장
    if not final: 
        return f"{image_id}: 변환 이미지 저장 실패" # 저장에 실패하면 표시
    
    return None # 성공

# 실행
if __name__ == "__main__":
    print(f"총 {len(image_id)}개 이미지 처리 시작") # 갯수 확인
    with Pool(min(16, cpu_count())) as pool: # 빠른 처리를 위해 멀티프로세싱 사용, 최대 16코어, 본인 CPU 사양에 맞게 조절이 필요, cpu_count()는 본인 cpu의 총 코어 갯수
        fix = list(tqdm(pool.imap(image_change, image_id), total=len(image_id))) # 이미지 추출 및 가공을 병렬로 실행 및 결과 저장, tqdm 진행도 확인

    # 실행 결과 확인
    error = [e for e in fix if e] # 문제 발생 시 에러 메세지를 남김
    if error:
        with open("추출 실패 목록", "w") as txt: # 텍스트 파일 작성
            txt.write("\n".join(error)) # 모든 발생 문제 작성
        print(f"결과: 일부 실패, 실패 {len(error)}건 기록")
    else: # 에러 메시지가 없는 경우
        print("결과: 모든 이미지 처리 완료")