---
layout: post
title:  "[C# 기초] 5. 상속과 다형성"
subtitle:   "inheritance"
categories: csharp
tags: language
---

> 클래스를 상속받아 코드를 재활용하고 상속받은 여러 클래스들을 한번에 핸들링이 가능하게 하는 다형성이라는 특징에 대해 배웁니다. 

## 상속성(Inheritance), 코드 재활용하기
---

상속은 기반 클래스(base class)로 부터 필드, 메소드 등을 그대로 물려 받아 새로운 파생 클래스(derived class)를 만드는 것입니다.

this는 자기자신 객체를 가리키듯이, base는 기반 클래스를 가리킵니다. this() 생성자와 같이 base() 생성자도 마찬가지입니다.
파생클래스에서 base 키워드를 사용하지 않아도 상속받은 필드, 메소드 등이 노출되지만 (private가 아닐 경우) 명확하게 표현하는 것은 좋은 습관이므로 base 키워드를 사용하는 것을 권합니다.

참고로, C#에서는 죽음의 다이아몬드 문제(The Deadly Diamond of Death)로 클래스에 대한 다중 상속을 지원하지 않습니다.

C#에서는 sealed 한정자로 클래스를 작성하게 되면 상속을 봉인할 수 있습니다. (해당 클래스를 상속할 경우 컴파일에러)

```
sealed class Base { ... }
class Derived : base  // 컴파일 에러
{ ... }
```

## 다형성(Polymorphism), 오버라이딩을 해보자
---

다형성은 객체가 여러 형태를 가질 수 있음을 의미합니다.

파생클래스의 인스턴스는 기반클래스의 인스턴스로 사용할 수 있습니다. 이를 이용하기 위해서는 캐스팅을 해야하는데 C#에서는 안전한 캐스팅 방법으로 다음을 제공합니다.

| 연산자 |                                                                      설명                                                                      |
|:------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|
| is     | 객체가 해당 형식에 해당하는지를 검사하여 그 결과를 bool 값으로 반환합니다.                                                                     |
| as     | 형식 변환 연산자와 같은 역할을 합니다. 다만 형변환 연산자가 변환에 실패하는 경우 예외를 던지는  반면에 as연산자는 객체 참조를 null로 만듭니다. |


기반 클래스에 있는 메소드를 파생 클래스에서 사용할 때 다른 기능을 하고 싶다면 오버라이딩(Overriding)을 할 수 있습니다. 즉 파생 클래스에서 기능이 바뀔 수 있기 때문에 재정의하는 것을 말합니다. 이 때 조건은 오버라이딩할 메소드가 virtual 키워드로 되어있어야 합니다.(안쓸 경우 컴파일 에러 발생하므로 이해하고만 있으면 됩니다.)

기반 클래스로 파생클래스들을 받아 파생클래스의 오버라이딩된 메서드들을 기반클래스에서 파생클래스로 다시 캐스팅할 필요 없이(알아서 객체를 인식하고) 호출할 수 있습니다. 단 오버라이딩되지 않은 메서드를 호출하면 기반 클래스에 있는 virtual을 그대로 호출합니다. 만약 기반 클래스에 있는 virtual이 구현할 필요가 없는 추상적인 내용이라면 abstract로 만들면 반드시 파생클래스는 오버라이딩하여 구현해야 합니다.

따라서 virtual로 메소드를 정의 한다는 것은 팀프로젝트 단위에서 여러명이서 개발을 진행할 때 이 기반클래스를 상속받아 사용할때 재정의하여 사용하라는 프로그래머의 지시가 될 수 있습니다.

상속을 봉인했던 것처럼 오버라이딩 또한 봉인할 수 있습니다. 파생클래스에서 오버라이딩하여 사용하여 정의하였습니다. 하지만 이 파생클래스를 다시 상속받아 사용할때 오버라이딩을 할 수 없도록 봉인하는 것입니다.

```
class Base
{
	public virtual void SealMe()
	{}
}

class Derived : Base
{
	public sealed override void SealMe()
	{}
}

class DerivedDerived : Derived
{
	public override void SealMe()   // 컴파일 에러
	{}
}

```

봉인 메소드는 파생 클래스의 작성자를 위한 배려입니다. 혹시라도 파생 클래스의 작성자가 오버라이딩 했을 경우 클래스의 다른 부분이 오작동할 가능성이 있다고 판단될 때 사용할 수 있습니다.

## 상속성과 다형성, 코드로 이해하기
---

모든 사람을 Human 객체로 생각해봅시다. 모든 사람은 전부 다 성격이 다르고, 개개인마다 개성이 있습니다. 축구선수 박지성, 야구선수 류현진, 영화배우 송강호 등을 모두 표현하기 위해 Jisung, Hyunjin, Gangho 라는 클래스를 만들어 필드(이름,나이 등)와 기능; 메소드(개개인 특징; 연기력,운동실력 등)를 정의 하는 것은 아주 비효율적입니다.

따라서 Human 이라는 객체를 기반으로 하고, 이를 상속받아 SoccerPlayer, BasebollPlayer, Actor의 클래스에 각각의 추가적인 기능(연기,운동) 등을 추가하거나 재정의하기만 하면 되는 것입니다. 그리고 이 객체를 사용할 때는, 모두 각각의 객체로 생각하는 것이 아니라 사람이라는 객체로 간주하여, 형변환을 통해 사람이라는 기반클래스로 파생클래스 전부를 컨트롤 할 수 있습니다.

예제 코드:
```
class Human
{
    string Name;

    public Human(string name="")
    {
        this.Name = name;
    }

    public void PrintMyName()
    {
        Console.WriteLine(Name);
    }

    public virtual void Play()
    {
        Console.WriteLine("Nothing");
    }
}

class SoccerPlayer : Human
{
    public SoccerPlayer(string name="") : base(name)
    { }

    public override void Play()
    {
        Console.WriteLine("Soccer");
    }

    public void Training()
    {
        Console.WriteLine("Soccer Training");
    }
}

class Actor : Human
{
    public Actor(string name="") : base(name)
    { }

    public override void Play()
    {
        Console.WriteLine("Action");
    }
}

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("===1번째 방식===");
        Human human1 = new SoccerPlayer("박지성");
        human1.PrintMyName();
        // human1.Training(); 
        // human1은 Human Type이므로 SoccerPlayer의 인스턴스 및 메소드를 접근할 수 없다.
        human1.Play();

        Console.WriteLine("===2번째 방식===");
        Human human2 = new SoccerPlayer("박지성");
        SoccerPlayer Soccer2 = (SoccerPlayer)human2;
        Soccer2.PrintMyName();
        Soccer2.Training();  // ==> SoccerPlayer로 캐스팅 후 방식1을 해결할 수 있다.
        Soccer2.Play();

        Console.WriteLine("===3번째 방식===");
        Human human3 = new SoccerPlayer("박지성");
        SoccerPlayer Soccer3 = human3 as SoccerPlayer;
        Soccer3.PrintMyName();
        Soccer3.Training();
        Soccer3.Play();

        Console.WriteLine("===객체를 잘 모를 때 안전한 형변환1===");
        Human human4 = new Actor("송강호");
        Actor actor4;
        if (human4 is Actor) // Human 객체가 SoccerPlayer 형식임을 확인한 후 형변환
        {
            actor4 = (Actor)human4;
            actor4.PrintMyName();
            actor4.Play();
        }

        Console.WriteLine("===객체를 잘 모를 때 안전한 형변환2===");
        actor4 = human4 as Actor; // 형변환이 실패할 경우 null 반환
        if( actor4!= null)
        {
            actor4.PrintMyName();
            actor4.Play();
        }
    }
}
```

위의 코드를 보면 Human 기반 클래스를 중심으로 SoccerPlayer, Actor를 상속받아서 SoccerPlayer는 Human 이라는 클래스로도 표현할 수 있습니다.

이렇게 다형성이 있다는 것이 강력한 이유를 맛보기로 보여드리면 SoccerPlayer와 Actor 등 모든 Human들이 일관되게 Play()를 동작하고 싶을 경우, 아래 코드로 손쉽게 가능합니다.

```
List<Human> people = new List<Human>() { human1, Soccer3, actor4 };

foreach (Human human in people)
{
    human.Play();
}
```

---
실제 상속과 다형성은 이론적으로 공부할 때에는 크게 장점을 느끼지 못할 수 있습니다. 하지만 실제로 프로그래밍하게 되면 얼마나 강력한지는 직접 느끼실수 있습니다.