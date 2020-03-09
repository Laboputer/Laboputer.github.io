---
layout: post
title:  "[C# 기초] 10. 람다식(Lambda expression)"
subtitle:   "lambda-expression"
categories: csharp
tags: language
---

> 함수형 언어에서 사용하는 람다식은 코드를 간결하게 만들어주는데 이를 C#에서도 사용할 수 있도록 지원해줍니다.

## 람다식 (Lambda-expression)
---

[함수형 프로그래밍](https://ko.wikipedia.org/wiki/%ED%95%A8%EC%88%98%ED%98%95_%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D)에서 사용하는 람다식이라는 것이 있습니다. 람다식은 익명 메소드(Anonymous Function)를 아주 간결하게 표현할 수 있습니다. C#에서도 이 람다식을 사용할 수 있습니다.

사용 방법:
```
(매개변수_목록)  =>  식
```

예제 코드:
```
delegate (int a, int b)
{
	return a+b;
}
```
```
(int a, int b) => a+b;
```
```
(a,b) => a+b;
```

위 세가지 방법은 모두 같은 기능을 가진 익명 메소드입니다. 첫번째 델리게이트를 람다식을 이용해 두번째 방식처럼 만들 수 있으며, 세번째 방식은 C#에서 형식 유추 (Type Inference) 기능을 제공하기 때문에 매개변수 형식을 제거할 수 있습니다.


문(Statement) 형식의 람다식의 선언 방식은 다음과 같습니다.

사용 방법:
```
(매개변수_목록) =>
{
	//코드
}
```

예제 코드:
```
class Program
{
    delegate int Method(int a, int b);

    static void Main(string[] args)
    {
        Method Add= (a, b) => a+b;
        Console.WriteLine(Add(3, 4)); // 7

        Method Minus = (a, b) =>
            {
                Console.WriteLine("{0} - {1} 의 결과는?", a, b);
                return a - b;
            };
        Console.WriteLine(Minus(5, 3)); // 5 - 3 의 결과는? 2
        
    }
}
```
위와 같이 익명 메소드를 만들기 위해 별개의 델리게이트를 선언하는 번거로운 일이 있어서 C# 은 Func 델리게이트와 Action 델리게이트를 지원합니다.
Func 델리게이트는 결과를 반환하는 메소드를 참조하기 위해, Action 델리게이트는 반환형식이 없는 메소드 참조를 위해서 사용합니다.

---
이 개념을 알지 못해도 개발하는데는 문제는 없겠지만 간단하게 사용방법만 익혀도 코드가 단순해집니다. 대부분 C# 프로그래머도 자주 사용하니 남들의 코드를 이해하기 위해서라도 사용하는 것이 좋습니다.