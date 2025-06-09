# fiftyone을 이용한 Openimage dataset V7 다운로드
# fiftyone.zoo로 데이터셋 로드
# Openimage에서 제공하는 V7 훈련 데이터를 가지고와 사용
# 객체 구분을 위해 detections 레이블 이용
# 프로젝트 목적에 맞는 클래스로 수집 - 클래스 이름은 Openimage 웹사이트 제공, 확인 가능
# 다운로드 시 이미지 구분을 위해 하나의 레이블만 다운받을 수 있게 일부 주석 처리
# 다운로드 후 다운받은 이미지를 확인하고 분류시켜 놓고 실행한 부분은 주석 처리하고 나머지 중 하나의 주석을 제거하여 실행을 권장
# 만약 다운로드하는 클래스를 추가하거나 기존의 것을 삭제했다면 data_fix에서 정의된 클래스 고유 ID 목록을 수정해야 한다.

import fiftyone.zoo

# person data, 한번 사용 후 주석 처리 
data = fiftyone.zoo.load_zoo_dataset (
    "open-images-v7", # 가져올 데이터 셋
    split="train", # 진행하는 프로젝트에서는 큰 의미는 없으나 데이터 다운로드를 위해 명시
    label_types="detections", # 이미지의 라벨링 타입, 이미지에 존재하는 객체에 대한 클래스와 바운딩 박스 좌표 모음 
    classes=["Man", "Woman", "Boy", "Girl"], # 다운로드할 이미지의 클래스 이름
    max_samples=18000 # 데이터 수
)

# animal data
animal = [
    "Antelope", 
    "Bat (Animal)", 
    "Bear",
    "Cat",
    "Cattle",
    "Chicken",
    "Deer",
    "Dog",
    "Eagle",
    "Falcon",
    "Fox",
    "Goat",
    "Goose",
    "Hamster",
    "Horse",
    "Magpie",
    "Leopard",
    "Mouse",
    "Mule",
    "Otter",
    "Owl",
    "Parrot",
    "Pig",
    "Rabbit"
    "Raven",
    "Sheep",
    "Snake",
    "Squirrel",
    "Swan",
    "Tiger",
    "Turtle",
]

# 사용 시 코드 앞에 주석을 제거, 사용 후 다시 주석 처리

#for C in animal :
    #data = fiftyone.zoo.load_zoo_dataset (
        #"open-images-v7",
        #split="train",
        #label_types="detections",
        #classes=C,
        #max_samples=540,
    #)      

# object data
ob = [
    "Airplane",
    "Ambulance",
    "Balloon",
    "Barge",
    "Barrel",
    "Bicycle",
    "Boat",
    "Box",
    "Briefcase",
    "Bus",
    "Canoe",
    "Car",
    "Dagger",
    "Gondola",
    "Handgun",
    "Knife",
    "Land vehicle",
    "Laptop",
    "Motorcycle",
    "Office supplies",
    "Pen",
    "Plastic bag",
    "Rocket",
    "Sculpture",
    "Segway",
    "Shotgun",
    "Snowplow",
    "Snowmobile",
    "Stationary bicycle",
    "Submarine",
    "Suitcase",
    "Tank",
    "Truck",
    "Van",
]

# 사용 시 코드 앞에 주석을 제거, 사용 후 다시 주석 처리

#for C in ob :
    #data = fiftyone.zoo.load_zoo_dataset (
        #"open-images-v7",
        #split="train",
        #label_types="detections",
        #classes=C,
        #max_samples=500,
    #)      