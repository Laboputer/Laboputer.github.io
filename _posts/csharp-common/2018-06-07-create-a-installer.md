---
layout: post
title:  "C# 윈도우 응용프로그램 배포용 설치파일 만들기"
subtitle:   "create-a-installer"
categories: csharp
tags: csharp-common
---

> 사용자에게 프로그램을 배포하기 위해 Windows Installer 형태로 사용자 PC에 설치할 수 있는 설치파일을 생성해보는 과정을 포스팅합니다.

> Visual Studio에서 제공하는 Installer Project로 설치 배포파일을 쉽게 만들 수 있으며 C# WindowsForms 프로그램에 대한 설치파일을 만드는 예제로 설명합니다.

## 개요
---

사용자에게 설치파일을 배포해야 하는 경우 아래와 같이 할 수 있습니다.

- 실행파일 등(.exe, .dll 등)을 Zip하여 배포
- 설치파일을 만들어 배포

첫번째는 규모가 비교적 작은 프로그램을 누군가에게 주거나 일부 사용자에게 일회성으로 전달할 때는 유용합니다. 하지만 불특정 다수의 사용자에게 배포할 때는 보통 두번째 경우를 선택합니다.

이 글은 설치파일을 만들어서 정식(?)으로 프로그램을 배포하는 방법에 대해 정리합니다.

테스트용 프로그램(C# WinForm)에 대한 설치파일(Setup.exe)을 만들기 위해 Visual Studio Installer를 사용할 것 입니다. 이 설치파일을 통해 사용자PC에 프로그램이 설치되고 바탕화면에 바로가기 파일까지 만드는 것을 목표로 합니다.

## 1. Visual Studio Installer 설치
---

Visual Studio 에서 `Microsoft Visual Studio Installer Projects` 라는 Extension을 다운로드하면 설치파일을 만들 수 있는 프로젝트를 추가할 수 있습니다. 이 Installer Project를 이용할 것이므로 설치하시면 됩니다.

> 위 Extension이 이미 설치가 되어 있는 분들은 넘어가셔도 됩니다.

### 설치 방법

1. Visual Studio > Tools > Extensions and Updates 메뉴
2. "installer" 검색 
3. Microsoft Visual Studio Installer Projects 설치 (아마도 제일 상단에 노출됨)

![](https://laboputer.github.io/assets/img/csharp/common/00_setup_01.png)

## 2. Setup Project 추가
---

설치한 Visual Studio Installer 의 `Setup Project` 를 솔루션에 추가합니다. 이 프로젝트에서 설치파일에 어떤 항목들을 포함할지, 어떤 과정으로 설치할지 등을 정할 수 있게 됩니다.

1. Solution > Add New Project > Other Project Types > Visual Studio Installer
2. Setup Project 추가

![](https://laboputer.github.io/assets/img/csharp/common/00_setup_02.png)

## 3. Setup Project 설정
---

설정을 하기 전에 배포할 테스트 프로그램은 다음과 같습니다.
- myApp : C# WinForm Application (.exe)
- myAppLibrary : reference project (.dll)

![](https://laboputer.github.io/assets/img/csharp/common/00_setup_03.png)

그리고 Setup Project > View > File System 에서 설치 파일에 어떤 것을 포함할지 결정합니다.

- Application Folder : 설치 목록에 포함될 파일
- User's Desktop : 사용자 PC 바탕화면에 추가할 파일
- User's Programs Menu : 사용자 PC 윈도우 시작메뉴에 추가할 파일

위에서 원하는 설정을 하면 됩니다. 여기서 할 것은 테스트 프로그램을 사용자 PC에 설치하고, 바탕화면과 시작메뉴에 설치된 프로그램의 바로가기까지 만들고자 합니다.

일일이 파일을 하나씩 추가해도 되지만, 간편하게 myApp 프로젝트의 기본 Output 파일을 설치목록으로 설정하면 간단합니다.

1. Application Folder > (우측클릭) > Add > Project Output
2. Project 선택(myApp) 후 Primary output 선택 

![](https://laboputer.github.io/assets/img/csharp/common/00_setup_04.png)

그 다음은 사용자 PC 바탕화면과 시작메뉴에 바로가기(Shortcut)를 추가합니다.

1. Application Folder 에서 'Primary output form myApp' > (우측 클릭) > Create Shortcut 
2. 1.에서 생성한 파일 User's Desktop 으로 옮김 (Drag and Drop)
3. 1을 다시 한번 한후에 User's Programs Menu 으로 옮김 (Drag and Drop)

저는 바로가기의 파일명을 'myApp-Shortcut' 으로 변경했습니다.

![](https://laboputer.github.io/assets/img/csharp/common/00_setup_05.png)

모든 설정이 완료되면 위 그림처럼 됩니다. 

## (완료) 4. Setup Project 빌드
---

설정을 완료한 후 Setup Project를 빌드하면 `Setup.exe` 와 `.msi` 파일이 생성됩니다.
- Setup.exe : 프로그램 파일
- .msi : 사용자 PC에 설치할 Installer

![](https://laboputer.github.io/assets/img/csharp/common/00_setup_06.png)

결론적으로 어떤 프로그램을 실행하든 `.msi` 파일을 통해 프로그램이 설치되므로 아무거나 실행하시면 됩니다.

> Setup.exe, .msi 파일에 대한 자세한 내용은 [Stackoverflow](https://stackoverflow.com/questions/1789530/what-are-the-specific-differences-between-msi-and-setup-exe-file) 참고

실제로 `.msi` 파일을 실행하면 아래와 같이 Windows Installer 형식으로 프로그램이 설치가 됩니다.

![](https://laboputer.github.io/assets/img/csharp/common/00_setup_07.png)

여기까지 하면 우리의 목표였던 설치파일 만들기는 완료입니다.

## (Optional) 추가 작업
---

이대로 끝나면 아쉬우니까 프로그램이 좀 더 그럴싸해지도록 몇 가지만 더 바꿔봅시다.

### 프로그램 아이콘

아이콘은 프로그램 UI titlebar 내 아이콘과 실행파일 아이콘, 바로가기 아이콘, 시작메뉴 아이콘, 윈도우 프로그램 추가/삭제의 아이콘 등 모두 각각 셋팅해야 합니다. 모두 다 같은 아이콘으로 바꿉니다.

1. 프로그램 UI : Form > Properties > Icon 추가
2. 실행파일(.exe) : Project > Properties > Applcation > Icon 추가
3. Shortcut : 각 Shortcut > Properties > Icon 추가 
4. 프로그램 추가/삭제 : Setup Project > Properties > AddRemoveProgramsIcon 추가

> 아이콘은 [이곳](https://icon-icons.com/icon/Adobe-CC-Creative-Cloud/78300)에서 다운받았습니다.

![](https://laboputer.github.io/assets/img/csharp/common/00_setup_08.png)

### 프로그램 부가정보

Setup Project > Properties 에서 기본적인 부가정보들을 변경할 수 있습니다.

- Author : 제작자
- Description : 프로그램 설명
- Manufacturer : 제조사
- ProductName : 제품명

저는 이정도만 변경하였습니다.

![](https://laboputer.github.io/assets/img/csharp/common/00_setup_09.png)

변경하게 되면 위 그림과 같이 `.msi` 파일에서 정보를 확인할 수 있습니다. 그리고 설치되는 기본 경로가 `Manufacturer`와 `ProductName`에 따라 변합니다. 물론 기본경로도 변경할 수 있지만요.

## 마치며..
---

이 포스팅에서는 배포할 프로그램에 대한 `Windows Installer`을 만드는 것을 정리하였습니다. 이 외에도 다양한 옵션들이 많이 있습니다. 사용자 PC의 레지스트리 변경이 필요하다거나 설치 과정 커스터마이징을 통해 라이센스 확인이나 사용자 체크사항 확인 등 많은 것이 가능합니다. 필요에 따라 찾아 추가해보시길 바랍니다.

---
이 포스팅에서 사용된 소스 파일 다운로드: [create-a-installer.zip](https://laboputer.github.io/assets/zips/create-a-installer.zip)