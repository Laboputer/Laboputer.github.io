---
layout: post
title:  "[C# 기초] 6) 인터페이스(Interface)"
subtitle:   "interface"
categories: csharp
tags: language
---

> C#은 인터페이스라는 것을 통해 아직 정의되지도 않은 클래스를 상속받아 다형성을 가질수도 있습니다. 

## 인터페이스 (Interface)
---

인터페이스는 클래스의 선언부 역할을 합니다. 인터페이스는 메소드, 이벤트, 인덱서, 프로퍼티만을 가질 수 있습니다. 구현부가 없기 때문에 당연히 인스턴스를 생성할 수 없습니다. 클래스는 인터페이스를 상속받아 인터페이스의 모든 메소드를 구현해야만 사용할 수 있습니다.

말 그대로 클래스의 인터페이스가 되는 것입니다. 인터페이스는 아직 정의도 되지 않은 클래스의 형태를 가지고 프로그래머에게 최소한의 형태만 갖추어지면 인터페이스로써 다형성을 가질수도 있고 인터페이스를 통해 이미 구현된 기능을 본인의 클래스를 정의하여 사용하기도 합니다.

인터페이스의 사용방법:
```
interface 인터페이스이름
{
	반환형식 메소드이름1 (매개변수);
	반환형식 메소드이름2 (매개변수);
}
```

예제코드:
```
interface IMonitor
{
    void PowerButton();
    void InputDevice(int Device);
}

class Monitor : IMonitor
{
    bool Power;
    int Devices;

    public Monitor()
    {
        this.Power = false;
        this.Devices = 0;
    }

    public void PowerButton()
    {
        if (Power)
        {
            Power = false;
            Console.WriteLine("Off");
        }
        else
        {
            Power = true;
            Console.WriteLine("On");
        }
    }

    public void InputDevice(int Device)
    {
        if (Power)
        {
            this.Devices = Device;

            switch (this.Devices)
            {
                case 0:
                    Console.WriteLine("TV");
                    break;
                case 1:
                    Console.WriteLine("Computer");
                    break;
            }
        }
    }
}

class Program
{
    static void Main(string[] args)
    {
        IMonitor imonitor = new Monitor();
        imonitor.PowerButton();
        imonitor.InputDevice(1);
    }
}
```

## 인터페이스를 상속하는 인터페이스
---

인터페이스를 상속할 수 있는 것은 클래스 뿐만이 아니라 구조체와 인터페이스도 인터페이스를 상속받을 수 있습니다.

보통 상속하려는 인터페이스가 소스코드가 아닌 어셈블리로 제공될 경우, 상속하려는 인터페이스의 소스코드를 가지고 있어도 이미 이 인터페이스를 상속하여 사용하는 클래스가 있어 수정할 수 없는 경우에는 인터페이스를 상속받아 수정하여 사용할 수 있습니다.
C#에서는 클래스에 대한 다중상속을 지원하지 않지만, 인터페이스는 죽음의 다이아몬드 문제가 발생하지 않으므로 인터페이스를 다중 상속받을 수 있습니다.

## 추상클래스 (abstract class)
---

추상클래스는 인터페이스와 클래스 사이의 개념이라고 보시면 됩니다.

추상클래스는 구현부를 가질 수 있으나 인터페이스를 만들지 못하고 추상 메소드(abstract method)를 가질 수 있습니다.
추상 메소드는 인터페이스의 메소드처럼 선언부만 가진 메소드입니다. 따라서 C# 컴파일러는 추상 메소드가 private가 아닌 그 외에 접근한정자로 수식할 것을 요구합니다.

위의 인터페이스 코드를 추상클래스 코드로 바꾸어 보겠습니다.
```
abstract class AbsMonitor
{
    protected bool Power;
    protected int Devices;
        
    public AbsMonitor()
    {
        this.Power = false;
        this.Devices = 0;
    }

    public void PowerButton()
    {
        if (Power)
        {
            Power = false;
            Console.WriteLine("Off");
        }
        else
        {
            Power = true;
            Console.WriteLine("On");
        }
    }

    abstract public void InputDevice(int Device);
}

class Monitor : AbsMonitor
{    
    public override void InputDevice(int Device)
    {
        if (Power)
        {
            this.Devices = Device;

            switch (this.Devices)
            {
                case 0:
                    Console.WriteLine("TV");
                    break;
                case 1:
                    Console.WriteLine("Computer");
                    break;
            }
        }
    }
}

class Program
{
    static void Main(string[] args)
    {
        AbsMonitor imonitor = new Monitor();
        imonitor.PowerButton();
    }
}
```

추상클래스는 '파생클래스를 만들어 사용하세요. 그리고, 주어진 추상메소드를 오버라이딩하여 사용해야 합니다' 라는 프로그래머의 메시지입니다.

---
인터페이스와 추상클래스라는 개념으로 상속과 다형성을 더 다양하게 활용할 수 있도록 있습니다.