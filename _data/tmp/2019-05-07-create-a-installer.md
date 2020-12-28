---
layout: post
title:  "C# 윈도우 응용프로그램 설치파일 만들기"
subtitle:   "matplotlib"
categories: machine-learning
tags: python tutorial matplotlib
---

> 직접 개발한 프로그램을 Windows Installer를 이용하여 사용자 PC에 설치 파일을 생성해보는 과정을 포스팅합니다.

> Visual Studio에서 제공하는 Installer로 설치 배포파일을 쉽게 만드실 수 있으며, C# WinForm 프로그램에 대한 설치파일을 만드는 예제로 설명합니다.

## 개요
---

프로그램 개발을 완료한 이후에 사용자에게 설치파일을 배포해야 하는 경우, 우리는 보통 2가지 방식을 따릅니다.

1. .exe, .dll 등을 Zip하여 배포
2. 설치파일을 만들어 배포

1번은 간단한 토이프로젝트 또는 일부 사용자에게 일회성으로 전달할 때는 유용합니다. 하지만 불특정 다수의 사용자에게 배포할 때는 보통 2번의 경우를 선택합니다.

여기에서는 2번의 경우처럼 설치파일을 만들어서 정식(?) 프로그램을 배포하는 방법에 대해
정리합니다.

Visual Studio Installer를 이용하여 테스트용 프로그램(C# WinForm)에 대한 설치파일(Setup.exe)을 만들 것입니다. 이 설치파일만 실행하면 자동으로 사용자PC에 설치되고 바탕화면에 바로가기 파일까지 만드는 것을 목표로 합니다.

## 1. Visual Studio Installer 설치
---

Visual Studio 에서 `Microsoft Visual Studio Installer Projects`라는 Extension을 설치하면 됩니다.

> 이미 설치가 되어 있는 분들은 넘어가셔도 됩니다.

### 설치 방법

1. Visual Studio > Tools > Extensions and Updates 메뉴
2. "installer" 검색 
3. Microsoft Visual Studio Installer Projects 설치 (아마도 제일 상단에 노출됨)


---
이 포스팅에서 사용한 코드는 [이곳](https://github.com/Laboputer/LearnML/blob/master/02.%20%5BPOST%5D/52.%20%5BCode%5D%20Matplotlib%20Tutorial.ipynb)에 공개되어 있으며 다운로드 받으실 수 있습니다.