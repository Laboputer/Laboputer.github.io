---
layout: post
title:  "C# log4net을 이용하여 쉽게 로그 남기기"
subtitle:   "log4net-tutorial"
categories: csharp
tags: csharp-common
---

> log4net 이라는 라이브러리를 통해 C# Windows Application 에서 로그를 쉽게 남기는 방법에 대해서 정리합니다.

## log4net
---

[log4net](https://logging.apache.org/log4net/)은 [Apache Software Foundation](https://www.apache.org/) 에서 개발한 .NET (C# 포함)에서 사용할 수 있는 Logging Framework 입니다.

프로그램에서 어떤 에러나 상황이 발생했는지를 로그를 남기고자 할 때 사용할 수 있습니다. 직접 로그파일을 생성하거나 어떤 상황이 발생할 때마다 긴 코드를 작성하지 않고도 `log4net`을 사용하면 쉽게 로그를 관리할 수 있습니다.

log4net은 로그를 다양한 공간(File, DB, Email 등)에 다양한 로그(Info, Debug, Error 등)를 만들 수 있으며 로그 생성 구조를 변경하고 싶을 때도 특별한 코드 수정없이 Config 파일만 변경하면 되어 편리합니다.

log4net에는 `Logger`, `Appender`, `Layout` 라는 컴포넌트가 있는데
`Logger`를 통해 코드상에서 로그를 출력할 수 있으며, 이 `Logger` 안에 `Appender` 들을 파일이나 DB 또는 Console 등 여러 가지를 붙여 다양한 로그를 만들 수 있습니다. 그리고 `layout`을 통해 메시지를 어떤 형태로 남길 것인지 정의할 수도 있습니다. 

> 자세한 설명은 [Apache log4net](https://logging.apache.org/log4net/release/manual/introduction.html) 을 참고하세요.

그럼 바로 설치하고 직접 로그를 남겨봅시다!

## 1. log4net 설치
---

Visual Studio 에서 Nuget Package 를 통해 설치할 수 있습니다. Nuget Package Console을 통해 설치해봅시다.

### Nuget Package Console 에서 아래 명령어 실행

```
PM> Install-Package log4net
```

![](https://laboputer.github.io/assets/img/csharp/common/01_log4net_1.PNG)

## 2. log4net.config 파일 추가
---

log4net 은 `log4net.config` 라는 Config 파일에 의해 로그를 어떻게 기록할지 결정됩니다. 따라서 이 config 파일을 추가하고 아래와 같은 설정을 진행하세요.

### Step 1. Add > New Item > Application Configuration File 추가

프로젝트에 "log4net.config" 파일을 추가하세요.

### Step 2. Copy to Output = "Copy always" 로 변경

프로젝트가 빌드되면 config 파일이 생성되도록 변경해야 합니다.

![](https://laboputer.github.io/assets/img/csharp/common/01_log4net_2.PNG)

### Step 3. Config 파일 내용 수정

아래 코드는 `Console` 과 `File` 2가지에 로그를 남길 수 있도록 설정한 것입니다.

아래 코드를 복사/붙여넣기 하세요.
```C#
<?xml version="1.0" encoding="utf-8" ?>
<configuration>
  <log4net>
    <root>
      <level value="ALL" />
      <appender-ref ref="console" />
      <appender-ref ref="file" />
    </root>
    <appender name="console" type="log4net.Appender.ConsoleAppender">
      <layout type="log4net.Layout.PatternLayout">
        <conversionPattern value="%date %level %logger - %message%newline" />
      </layout>
    </appender>
    <appender name="file" type="log4net.Appender.RollingFileAppender">
      <file value="myapp.log" />
      <filter type="log4net.Filter.LevelRangeFilter">
        <levelMin value="ERROR" />
        <levelMax value="FATAL" />
      </filter>
      <appendToFile value="true" />
      <rollingStyle value="Size" />
      <maxSizeRollBackups value="5" />
      <maximumFileSize value="10MB" />
      <staticLogFileName value="true" />
      <layout type="log4net.Layout.PatternLayout">
        <conversionPattern value="%date [%thread] %level %logger - %message%newline" />
      </layout>
    </appender>
  </log4net>
</configuration>
```

여기서 설정한 내용을 조금 설명하면 root 라는 `Logger`를 만들고 그 안에 Console과 File에 입력을 하는 `Appender` 2개를 생성하였습니다. 각각의 `Appender`는 `PatternLayout`에 쓰여진 형태로 로그를 남깁니다. 또한 `RollingFileAppender`는 `myapp.log` 라는 파일을 생성하면서 `filter`를 통해 `ERROR` ~ `FATAL` 형태의 로그만 남기도록 설정하였습니다.

잠시 후 샘플코드를 보면 이해되실 것입니다.

### Step 4. AssemblyInfos.cs 수정

프로그램이 `log4net.config` 파일을 읽을 수 있도록 있어야 합니다. `AssemblyInfo.cs` 파일에서 제일 하단에 아래 코드를 추가합니다.

```
[assembly: log4net.Config.XmlConfigurator(ConfigFile = "log4net.config")]
```

![](https://laboputer.github.io/assets/img/csharp/common/01_log4net_3.PNG)

## (완료) 3. 로그 출력하기
---

아래와 같은 방식으로 원하는 로직에서 로그를 출력할 수 있습니다.

```C#
public partial class Form1 : Form
{
    private static readonly log4net.ILog log = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

    public Form1()
    {
        InitializeComponent();
    }

    private void button1_Click(object sender, EventArgs e)
    {
        log.Info("Clicked!");

        try
        {
            DoSomething();	
        }
        catch (Exception)
        {
            log.Error("Error occured.");
        }

        log.Info("Ended!");
    }

    private void DoSomething()
    {
        throw new NotImplementedException();
    }
}
```

이 코드는 root에 해당하는 `Logger` 객체를 불러오는 것입니다.
```C#
private static readonly log4net.ILog log = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);
```

이 코드를 통해 해당 `Logger` 에서 `Info()` 레벨의 로그를 남길 수 있습니다.
```C#
log.Info("Clicked!");
```

> Log 레벨은 5가지 DEBUG > INFO > WARN > ERROR > FATAL 가 있으며 Filter를 통해 ALL 또는 OFF 로 전부 키거나 끌 수도 있습니다.
 
이 프로그램에서 Button Click을 하면 다음과 같은 결과가 나옵니다.

![](https://laboputer.github.io/assets/img/csharp/common/01_log4net_4.PNG)

Console은 모든 레벨의 로그를 출력하고, File에는 ERROR ~ FATAL 레벨의 로그만 출력되는 것을 확인하실 수 있습니다.


## 마치며..
---

프로그램이 커지면 커질수록 코드상의 로그 뿐만 아니라 로그의 출력위치가 다양해지고 변경될 때마다 큰 리소스가 들 수 있습니다. 이때 이런 Logging Framework를 이용하면 보다 편하게 관리하실 수 있을 것입니다.

---
Reference : 
- https://stackify.com/log4net-guide-dotnet-logging/
- https://logging.apache.org/log4net/release/manual/configuration.html