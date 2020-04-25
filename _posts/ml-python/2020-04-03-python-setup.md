---
layout: post
title:  "머신러닝 개발환경(Python) 구축하기"
subtitle:   "python-setup"
categories: machine-learning
tags: python tutorial
---
> Anaconda를 이용하여 python 개발환경을 구축하고 머신러닝 관련 패키지들을 설치하는 방법을 포스팅합니다.

## 머신러닝 분야를 공부해보고 싶어요!
---

Machine Learning 분야를 공부하고 개발해볼 수 있는 언어로는 R, Java 등 다양하고 언어에 맞는 개발환경(IDE)들도
많지만 그 중에서 가장 핫한 Python 언어와 개발환경을 구축하는 것을 정리하겠습니다. Python이 핫한 이유는 머신러닝 플랫폼을 포함하여
데이터 과학에 필요한 패키지들을 많이 포함하고 있기 때문으로 생각됩니다.

아래 과정을 따라 시작해보시길 바랍니다.


## Step 1. Python 개발환경 구축 (Anaconda 설치)
---
[아나콘다(Anaconda)](https://ko.wikipedia.org/wiki/%EC%95%84%EB%82%98%EC%BD%98%EB%8B%A4_(%ED%8C%8C%EC%9D%B4%EC%8D%AC_%EB%B0%B0%ED%8F%AC%ED%8C%90))가 무엇인지 설명은 생략하고 Python을 포함한 것을 설치한다고 생각하시면 됩니다.

### 1. Anaconda 다운로드 (Anaconda3-2020.02 권장)
(https://repo.continuum.io/archive/) 이곳에서 다운로드 받습니다. 제 PC는 Windows 운영체제 64bit 이므로 [Anaconda3-2020.02-Windows-x86_64.exe](https://repo.continuum.io/archive/Anaconda3-2020.02-Windows-x86_64.exe)를 받았습니다.

![](https://laboputer.github.io/assets/img/ml/python/setup-python/1.JPG)

### 2. Anaconda 설치
exe파일을 실행하여 설치합니다. 아무것도 건드리지 않고 Next만 눌러도 무방합니다.

![](https://laboputer.github.io/assets/img/ml/python/setup-python/2.JPG)

### 3. (완료) 실행 확인
설치가 완료되면 "Anaconda Prompt"가 보일 것입니다. 실행한 후 아래 명령어를 입력하시면 Python 버전을 확인할 수 있습니다.

```
python --version
```

![](https://laboputer.github.io/assets/img/ml/python/setup-python/3.JPG)


Python 개발환경이 만들어졌습니다. 같이 설치된 Jupyter Notebook으로 Python 언어로 실제 개발을 시작할 수 있습니다.
포스팅 마지막에 다시 소개하겠습니다.

## Step 2. 머신러닝 패키지 설치
---

Python에서 머신러닝 패키지는 다양하지만 구글의 [TensorFlow](https://www.tensorflow.org/)를 포함한 자주 사용하는 패키지들을 함께 설치해보겠습니다. 개발할때 필요한 패키지는 추후에도 같은 방식으로 설치하시면 됩니다.

### 1. pip 최신버전 업그레이드

패키지 설치시 [pip](https://pypi.org/project/pip/)를 이용하여 설치하기 때문에 pip 버전 먼저 업그레이드 합니다.
Anaconda Prompt에서 아래 명령어를 실행하세요.

```
python -m pip install --upgrade pip
```

Anaconda 2020.02 버전을 설치하였으면 포스팅 작성일 기준 pip가 최신 버전이므로, 아래와 같이 나옵니다.
![](https://laboputer.github.io/assets/img/ml/python/setup-python/4.JPG)


### 2. 머신러닝 관련 패키지 설치

아래 명령어를 이용하여 패키지를 설치할 수 있습니다.
```
pip install (패키지 이름)
```

패키지가 많기 때문에 제가 자주 사용하는 Tensorflow 2.0을 포함한 머신러닝 관련 패키지 모음을 텍스트 파일로 정리해놓았습니다.

[머신러닝 패키지 모음 다운로드](https://laboputer.github.io/assets/img/ml/python/setup-python/requirements.txt)

or

```
numpy
scipy
matplotlib
ipython
scikit-learn
pandas
pillow
wrapt
tensorflow==2.0.0
mglearn
```
텍스트파일(requirements.txt)을 생성하여 위 내용을 입력하셔도 됩니다.

그 다음 아래와 같은 명령어를 입력하면 한번에 설치됩니다. 이미지는 다운로드 받은 파일을 **C:/ 경로**에 복사했는데 만약 파일의 위치가 다르면 명령어에서 경로만 변경하여 입력하시면 됩니다.

```
pip install -r C:/requirements.txt
```

![](https://laboputer.github.io/assets/img/ml/python/setup-python/5.JPG)

### (참고) 머신러닝 학습 시 GPU를 사용하고 싶어요
실제로 공부를 하시다가 많은 양의 데이터를 학습하다보면 시간이 오래 걸리다보니 GPU을 이용하여 빠르게 학습시키고 싶으시게 되실 겁니다.
본인 PC의 GPU을 사용할 수도 있지만 구글이나 아마존의 클라우드 서비스를 이용하시는 것이 좋습니다.

(TensorFlow 기준) 본인 PC에서 GPU를 사용하시고자 한다면: 

 [TensorFlow-GPU Support](https://www.tensorflow.org/install/gpu) 공식홈페이지에 한글로 잘 설명되어 있으니 따라하시면 됩니다.

클라우드 서비스를 이용하고 싶다면:

[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) 또는 [AWS](https://aws.amazon.com/ko/ec2/) 을 추천합니다.

Colab 사용방법은 [이곳](https://laboputer.github.io/machine-learning/2020/04/04/colab/)에 별도로 포스팅하였습니다.

## (설치완료) 주피터 노트북 실행 확인
---
[Jupyter Notebook(주피터 노트북)](https://jupyter.org/)으로도 Python을 개발하고 테스트해볼 수 있는 좋은 툴입니다.
Anaconda 설치 시 자동으로 함께 설치가 됩니다.

만약 다른 IDE를 별도로 설치하고 싶으신 분들은 [Visual Studio Code](https://code.visualstudio.com/), [PyCharm](https://www.jetbrains.com/pycharm/) 등을 알아보시기 바랍니다.

이번 포스팅에서는 주피터 노트북을 실행하신 후에 Tensorflow 버전을 확인하면서 마무리하겠습니다.

Jupyter Notebook 실행:

![](https://laboputer.github.io/assets/img/ml/python/setup-python/6.JPG)


아래 명령어로 tensorflow의 버전을 확인할 수 있습니다.
```
import tensorflow as tf
tf.__version__
```

![](https://laboputer.github.io/assets/img/ml/python/setup-python/7.JPG)

