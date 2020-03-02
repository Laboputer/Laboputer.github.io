---
layout: post
title:  "[C# 문법] 1) .NET Framework"
subtitle:   "dotent framework"
categories: csharp
tags: language
---

> `[C# 문법]` 시리즈 형식으로 C# 프로그래밍 기본적인 문법에 대해 포스팅합니다. C# 언어를 처음 접하시는 분, C# 개발을 하고 있지만 문법이 익숙치 않으신 분들이 읽기에 좋습니다. 전체를 한번에 읽으시기 보단 필요한 챕터를 찾아 읽는 것을 권장합니다.
> 본 시리즈는 `뇌를 자극하는 C# 4.0 프로그래밍 - 박상현` 책 내용을 바탕으로 정리하였습니다.

> 이번 포스팅에서는 C# 언어로 만들어진 프로그램이 어떻게 실행될 수 있는지를 배웁니다.

## .NET Framework
---

.NET Framework는 Microsoft에서 개발한 프레임워크로 윈도우용 프로그램 개발 및 실행 환경입니다.
웹/윈도우 기반 응용프로그램을 개발 할 수 있으며 .NET 표준에 따르는 프로그래밍 언어로 개발된 프로그램의 실행환경입니다.

.NET 표준에 따르는 프로그래밍 언어 중 하나가 C# 이며 .NET Framework만 설치되어 있으면 C# 뿐만 아니라 다른 언어들로 개발된 프로그램들도 실행이 가능하게 합니다.

.NET Framework는 다양한 언어를 지원하지만 운영체제에 따른 제약이 있습니다. Linux, MacOS 등에서는 기본적으로 동작하지 않지만 다른 운영체제에서도 .NET Framework가 동작하도록 하는 프로젝트가 있습니다. 대표적으로 [MONO Project](https://www.mono-project.com/)이 있습니다. 이 포스팅에서는 다루지 않으니 관심있으신 분들은 찾아보시길 바랍니다.

## .NET Framework 구조
---

![](https://laboputer.github.io/assets/img/csharp/01-1.PNG)

.NET 표준을 따르는 프로그래밍 언어 C# 외에도 VB .NET , Managed C++ , Jscript .NET , j# 등으로 ASP.NET으로 웹 서비스를, Windows Forms으로 윈도우 응용프로그램을 개발할 수 있습니다.

- ASP.NET Web Forms Web Services : Web service, Web apps 개발 
- Windows Forms : 윈도우용 응용 프로그램 개발
- ADO .NET and XML : XML 및 데이터베이스 활용
- Base Class Library : CLR 지원을 위한 핵심 클래스 라이브러리

## Common Language Runtime(CLR)
---

.NET 언어로 작성된 프로그램을 실행하고 관리하는 실행환경이며 Java의 [Virtual machine](https://ko.wikipedia.org/wiki/%EC%9E%90%EB%B0%94_%EA%B0%80%EC%83%81_%EB%A8%B8%EC%8B%A0)과 비슷합니다.

CLR은 .NET에서 동작하는 프로그램을 적재하고, 프로그램의 동적 컴파일, 프로그램의 실행, 메모리관리 (Garbage Collection), 프로그램의 예외처리, 언어 간의 상속 지원, COM과의 상호 운영성 지원 등을 가능하게 합니다. .NET 언어 즉, C#뿐만 아니라 위에서 나열하였던 VB .NET 등 .NET 표준을 따르는 다양한 언어들은 CLR을 통해 프로그램을 실행할 수 있게 됩니다.

![](https://laboputer.github.io/assets/img/csharp/01-2.PNG)

이것이 가능한 이유는 Intermediate Language (IL, 중간언어) 라는 기계어로 변환하기 쉬운 중간 단계의 언어로 .NET에서 실행되기 위해 IL 형태로 컴파일을 하는데 IL을 기계어로 바꾸는 번역기만 제공되면 어떤 플랫폼에서도 실행가능합니다. JIT (Just-In-Time) 컴파일러를 통해 IL을 동적으로 컴파일하는데 .NET 에서는 이와같이 프로그램을 2번 컴파일 합니다. Assembly는 이 때 IL로 컴파일된 결과 파일들을 패키징 한 것을 말합니다.

또 .NET 언어가 지켜야 하는 스펙으로 Common Language Specifications (CLS)로 다른 .NET 언어로 작성된 것도 호환되어 동작 가능하게 합니다. 이 과정에서 Common Type System(CTS)라는 .NET 언어마다 Data type이 다를 수 있는데 이를 언어나 시스템 환경에 관계없이 동일한 Data type을 유지하기 위한 규약이 사용되기도 합니다.

간단히 요약하면, C# 컴파일러는 우리가 작성한 코드를 IL로 작성된 실행파일을 만들게 되고, 이 파일을 실행시키면 CLR이 IL을 읽어 다시 OS에 이해할수 있는 코드로 컴파일하여 실행시킵니다. 이렇게 서로 다른 언어가 만나는 지점이 IL언어이고, CLR이 다시 자신이 설치되어 있는 플랫폼에 최적화시켜 컴파일한 후 실행하게 됩니다. 이를 통해 플랫폼에 최적화된 코드를 만들어 내는 장점이 있으나, 실행시에 이루어지는 컴파일이 부담일 수 있습니다.

---

추상적인 개념이라 이해하기 어렵지만 C# 언어로 개발하다보면 도움이 될만한 내용이므로 한번쯤은 읽어보면 좋을 것 같습니다.