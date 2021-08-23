# Hallym University Capstone Design
캡스톤디자인-5인이상집합금지

# Create fake faces through deepfakes and discriminate fake face images

-----

## Directory
- Team A : 이미지 생성을 위한 Gan모델
- Team B : 이미지 판별을 위한 CNN모델
- VisualizeFake : 생성한 이미지

## Contents
1. [소개 : 프로젝트에 대한 간략한 소개 및 목적](#Introduction)
2. [팀 : 팀원소개 및 역할](#Team)
3. [구축한 모델 : 프로젝트 해결을 위한 구축모델 정의]
    - [GAN(Team A) : Generative Adversarial Network](#GAN-Model)
    - [CNN(Team B) : Convolutional Neural Network](#CNN-Model)
4. [문제해결 : 프로젝트 과정](#Method)
    - [Data : 사용 데이터 셋](#Data)
    - [DCGAN : 가짜 사람얼굴 데이터 생성모델]
    - [CNN : 가짜 이미지 판별기 모델]

5. [결과 : 프로젝트의 결과](#Result)
6. [기대효과 : 프로젝트를 의의 및 확장성](#Benefit)
----

## Introduction

본 프로젝트는 GAN(Generative Adversarial Network) 모델과 CNN(Convolutional Neural Network) 모델을 동시적으로 학습시키는 방법을 고안했다.

AI 기술이 빠르게 발전하고 대중화 되면서 AI분야 중 딥러닝을 이용하여 인위적으로 얼굴을 조작하여 Deepfake 뒤에 숨어 타인의 명예를 훼손하거나 허위 사실들을 유포하는 등 범죄에도 사용되는 사례가 발생하고있다. 
프로젝트에서 제안하는 방법은 두 네트워크 간의 적대적 학습방법을 통해 Team A은 육안을 넘어 컴퓨터가 인지하기 어려운 가짜 얼굴을 만들어 내고 TeamB은 생성된 이미지를 인식하여 real과 fake이미지를 구분하는 모델을 구성하여, TeamA와 TeamB은 서로 경쟁을 하며 성능이 개선된 생성기(Generator)와 판별기(Discriminator)를 만드는 것이 주된 목적이다.

## Team
5인이상 집합금지

|Name|Department|Contact|
|---|---|---|
| Lee Kyung | Hallym Univ | kevin960317@naver.com
| Choi Min Gi | Hallym univ | chlalsrl98@naver.com
| Eu Hyung Gyu | Hallym univ | ehg2652@naver.com
| Gu Young Mo | Hallym univ | dudah0776@gmail.com
| Hong Seung Taek | Hallym univ | tmdxor97@naver.com

Professor
|Name|Department|Contact|
|---|---|---|
| Lim sung hoon | Hallym Univ(Prof.) | shlim@hallym.ac.kr

![generator](https://user-images.githubusercontent.com/79297596/128826502-5e2613b5-128d-4e23-a99f-f8e6277eef71.jpg)
![Discriminator](https://user-images.githubusercontent.com/79297596/128826523-b4556ada-d031-4e74-a8eb-4c6e61a48013.jpg)

위에서 정의한 2가지 모델을 다음과 같은 구조로 경쟁을 하며 네트워크의 성능 발전을 시키고자 한다.

##  Method
### Data

- 각 Team 모델학습에 사용한 데이터
    |Model|학습데이터|
    |---|---|
    |Team A | Real Face 52600장 |
    |Team B| Real Face 15000장 / Fake Face 15000|

-----
## GAN-Model
-----
 - 사람얼굴을 비지도 학습을 통하여 가짜얼굴(Deepfake)를 생성하는 적대적 신경망 구조
 - GAN 구조

#### Network Structure

<div display="inlie-block">
<center><img뭐시기>
</div>

GAN은 DCGAN구조를 참고하여 구현하였다.
GAN (Generative Adversarial Network)은 딥러닝 모델 중 이미지 생성에 널리 쓰이는 모델이다. 기본적인 딥러닝 모델인 CNN (Convolutional Neural Network)은 이미지에서 개인지 고양이인지 구분하는 이미지 분류 (image classification) 문제에 널리 쓰이고 있다. GAN은 CNN과 달리 개는 라벨 0이 하고, 고양이는 라벨 1이라하는 것 처럼 진행하는 이미지 분류 문제보다 더 복잡하다. GAN 모델이 데이터셋과 유사한 이미지를 만들도록 하는 것이다.


GAN은 Generator (생성자)와 Discriminator (판별자) 두 개의 모델이 동시에 적대적인 과정으로 학습한다. 생성자 G는 실제 데이터 분포를 학습하고, 판별자 D는 원래의 데이터인지 생성자로부터 생성이 된 것인지 구분한다. 생성자 G의 학습 과정은 이미지를 잘 생성해서 속일 확률을 높이고 판별자 D가 제대로 구분하는 확률을 높이는 과정이라고 볼 수 있다.

본 실험에서 사용한 DCGAN의 Generator의 구조는 Random-Noise를 Input으로 넣으면 최종 출력 이미지는 64X64크기로 출력 된다.

DCGAN의 Discriminator의 구조는 64X64크기의 이미지를 입력 받아 True와 False의 결과를 출력한다.

활성화함수로는 아래 그림에서 확인할 수 있듯이 LeakyReLU를 사용한다. LeakyReLU는 기존 ReLU와 달리 음수영역의 값을 버리지 않고 가져온다.

## CNN-Model
-----
 - real/fake로 이진분류를 통해 GAN모델에서 만든 가짜얼굴(Deepfake)을 판별하는 모델 구조
 - Binary Classification

#### NetWork Structure

<div display="inlie-block">
<center><img뭐시기>
</div>

 - CNN(Convolutional Neural Network)
Fully Connected Layer 만으로 구성된 인공 신경망의 입력 데이터는 1차원(배열) 형태로 한정된다. 한 장의 컬러 사진은 3차원 데이터이다. 배치 모드에 사용되는 여러장의 사진은 4차원 데이터이다. 사진 데이터로 전연결(FC, Fully Connected) 신경망을 학습시켜야 할 경우에, 3차원 사진 데이터를 1차원으로 평면화시켜야 한다. 사진 데이터를 평면화 시키는 과정에서 공간 정보가 손실될 수밖에 없다. 결과적으로 이미지 공간 정보 유실로 인한 정보 부족으로 인공 신경망이 특징을 추출 및 학습이 비효율적이고 정확도를 높이는데 한계가 있고, 이미지의 공간 정보를 유지한 상태로 학습이 가능한 모델이 바로 CNN(Convolutional Neural Network) 이다.


 -Binary Classification
딥러닝에서 말하는 Binary Classification은 구분하고자 하는 결과 값이 2가지인 경우, 예를 들면 고양이/개를 분류하는 것과 같다. CNN이란 Convolutional Neural Network에서 이미지나 영상과 같은 데이터를 처리하는 방법이다. 본 프로젝트에서는 CNN을 사용해 2가지 label을 가진 데이터를 분류를 하고자 하였고, GAN 네트워크를 통해 만들어낸 Deepfake와 실제 사람 얼굴을 5:5비율로 맞추어 경쟁을 하며 accuracy의 값이 0.5이상일 경우 TeamB의 알고리즘이 더 잘 인식하는 것으로 판정한다.


### Output
 - GAN을 이용한 가짜 사람 얼굴(Deepfake) 데이터 생성 과정 및 결과
![FakeFaceAnimation](https://user-images.githubusercontent.com/79297596/129441754-fa501f94-442c-4d85-9a00-0e2467dc9411.gif)
-----

## Result
본 프로젝트에서는 실제 데이터와 GAN 모델을 통해 생성한 Deepfake데이터를 다른 데이터로 학습한 CNN Classification 모델의 Test셋으로 하여 더욱 잘 속이고, 더욱 잘 판별하는 두가지의 네트워크의 구현을 해보았다.
여러번의 실험을 통해 나온 정확도
1. 0.532... -> TeamB 승
2. 0.434... -> TeamA 승
.....

컴퓨터가 인식하지 못하는 데이터를 생성하고 발전하였다.

## Benefit
1. GAN 알고리즘을 통해 가짜라고 인지하기 어려운 Deepfake 생성
2. 범죄로 악용될 수 있는 Deepfake를 판별을 통한 단점 보안
3. 두 네트워크의 성능 향상
