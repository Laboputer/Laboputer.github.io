---
layout: post
title:  "C# 윈도우 응용프로그램 배포용 설치파일 만들기"
subtitle:   "create-a-installer"
categories: csharp
tags: csharp-common
---

> 사용자에게 프로그램을 배포하기 위해 Windows Installer 형태로 사용자 PC에 설치할 수 있는 설치파일을 생성해보는 과정을 포스팅합니다.

> Visual Studio에서 제공하는 Installer Project로 설치 배포파일을 쉽게 만들 수 있으며 C# WinForm 프로그램에 대한 설치파일을 만드는 예제로 설명합니다.

## 개요
---

사용자에게 설치파일을 배포해야 하는 경우 보통 아래와 같이 합니다.

1. 실행파일 등(.exe, .dll 등)을 Zip하여 배포
2. 설치파일을 만들어 배포

1번은 간단한 토이프로젝트 또는 일부 사용자에게 일회성으로 전달할 때는 유용합니다. 하지만 불특정 다수의 사용자에게 배포할 때는 보통 2번의 경우를 선택합니다.

이 글은 2번의 경우처럼 설치파일을 만들어서 정식(?) 프로그램을 배포하는 방법에 대해
정리합니다.

테스트용 프로그램(C# WinForm)에 대한 설치파일(Setup.exe)을 만들기 위해 Visual Studio Installer를 사용할 것 입니다. 이 설치파일을 통해 사용자PC에 프로그램이 설치되고 바탕화면에 바로가기 파일까지 만드는 것을 목표로 합니다.

## 1. Visual Studio Installer 설치
---

Visual Studio 에서 `Microsoft Visual Studio Installer Projects`라는 Extension을 다운로드하면 설치파일을 만들 수 있는 프로젝트를 추가할 수 있습니다. 이 Installer Project를 이용할 것이므로 설치하시면 됩니다.

> 이미 설치가 되어 있는 분들은 넘어가셔도 됩니다.

### 설치 방법

1. Visual Studio > Tools > Extensions and Updates 메뉴
2. "installer" 검색 
3. Microsoft Visual Studio Installer Projects 설치 (아마도 제일 상단에 노출됨)

![]


## 2. Setup Project 추가
---

위에서 설치한 Visual Studio Installer 의 'Setup Project' 를 솔루션에 추가합니다. 이 프로젝트에서 설치파일에 어떤 항목들을 추가할지, 어떤 과정으로 설치할지 등을 설정할 수 있게 됩니다.

1. Solution > Add New Project > Other Project Types > Visual Studio Installer
2. Setup Project 추가

![]

## 3. Setup Project 설정
---

Setup Project > View > File System 에서 설치 파일에 어떤 것을 포함할지 설정할 수 있습니다.

- Application Folder : 설치 목록에 포함될 파일
- User's Desktop : 사용자 PC 바탕화면에 추가할 파일
- User's Programs Menu : 사용자 PC 윈도우 시작메뉴에 추가할 파일

![]

위에서 원하는 설정을 하면 됩니다. 여기서 할 것은 테스트 프로그램을 사용자 PC에 설치하고, 바탕화면과 시작메뉴에 설치된 프로그램의 바로가기까지 만들려고 합니다.

설정을 하기 전에 배포할 테스트 프로그램은 다음과 같습니다.
- myApp : C# WinForm Application (.exe)
- myAppLibrary : reference project (.dll)

![]

일일이 파일을 하나씩 추가해도 되지만 myApp 프로젝트의 기본 Output 파일을 설치목록으로 설정하겠습니다.

1. Application Folder > Add > Project Output
2. Project 선택(myApp) 후 Primary output 선택 

![]

그 다음은 사용자 PC 바탕화면과 시작메뉴에 바로가기(Shortcut)를 추가합니다.

1. User's Desktop > (우측 클릭) > Create Shortcut
2. User's Programs Menu > (우측 클릭) > CReate Shortcut

![]

바로가기의 파일명은 myApp-Shortcut 으로 설정했습니다.

## (완료) Setup Project 빌드
---

설정이 완료된 후 Setup Project를 빌드하면 Setup.exe 와 .msi 파일이 생성됩니다.
- Setup.exe : 프로그램 파일
- .msi : 사용자 PC에 설치할 Installer

결론적으로는 어떤 프로그램을 실행하든 .msi 파일을 통해 프로그램이 설치되므로 아무거나 실행하시면 됩니다.

> 자세한 답변은 [differences betweetn msi and setup exe file - stackoverflow](~~) 참고

사용자는 설치파일을 이용하여 Windows Installer 형식으로 프로그램을 설치할 수 있습니다.

![]

이렇게 우리의 목표였던 설치프로그램은 완성됐습니다.

## (Optional) 추가 셋팅 
---

이대로 끝나면 아쉬우니까 프로그램이 좀 더 그럴싸해지도록 몇 가지만 더 바꿔봅시다.

### 프로그램 아이콘

아이콘은 프로그램 Ui titlebar 내 아이콘과 실행파일 아이콘, 바로가기 아이콘, 시작메뉴 아이콘, 윈도우 프로그램 추가/삭제의 아이콘 등 모두 각각 셋팅해야 합니다. 모두 다 같은 아이콘으로 바꿉니다.

![]()

### 프로그램 부가정보

Setup Project > Properties 에서 기본적인 부가정보들을 변경할 수 있습니다.

![]()

변경하면 .msi 파일의 속성에 나오게 됩니다.

![]()


## 마치며..
---

Setup Project 의 File System 만 다뤘는데 이 외에도 다양한 옵션들이 많이 있습니다.
사용자 PC의 레지스트리 설정(Registry), Install 과정 커스터마이징(User interface) 등 필요에 따라 찾아보시면 될 것 같습니다.