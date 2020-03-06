---
layout: post
title:  "[C# 기초] 8. 예외처리(Exception handling)"
subtitle:   "exception"
categories: csharp
tags: language
---

> 잘 짜여진 프로그램은 수많은 예외 상황에도 적절한 대처를 취해줍니다. 이런 예외 상황은 무엇이고 어떻게 처리하는지를 배웁니다.

## 예외 (Exception), 프로그램이 예외 상황을 만났을 때
---

프로그램을 유저가 실제로 사용하면 다양한 예외 상황들이 발생하게 됩니다. 만약 이 예외 상황에 대한 처리를 하지 않으면 프로그램은 강제종료 됩니다. 이런 예외 상황을 대처할 수 있도록 C#은 여러가지 예외클래스를 제공합니다.(DivideByZeroException, IndexOutOfRangeException, NullReferenceException 등)

먼저 예외 상황이 발생하면 어떻게 되는지 보겠습니다.

예외 상황(예시):
```
class Program
{
    static void Main(string[] args)
    {
        int[] Numbers = new int[5] { 1, 2, 3, 4, 5 };

        for(int i=0; i<6; i++)
        {
                Console.WriteLine(Numbers[i].ToString());
        }
    }
}
```
![](https://laboputer.github.io/assets/img/csharp/07-1.PNG)

위와 같이 잘못된 인덱스를 통해 접근하려하면 문제의 상세정보를 IndexOutOfRangeException의 객체에 담은 후 Main() 메소드로 던집니다. 여기서 예외처리를 하지 않았기 때문에 Main() 메소드는 다시 CLR에게 던집니다. CLR이 '인덱스가 배열 범위를 벗어났습니다.'는 문구를 띄우면서 프로그램이 강제종료 되는 모습을 보실 수 있습니다.

위의 메시지는 CLR이 예외처리를 하라고 프로그래머한테 메시지(이것도 일종의 예외처리겠지요.)를 보낸 것입니다. 즉 프로그래머가 예외처리를 하지 않았기 때문에 프로그램은 강제종료가 됩니다.

## 예외처리(Exception Handling), 프로그램 강제 종료는 금물!
---

예외처리는 if~else문으로 전부 처리할 수 있습니다. 하지만 예외에 대한 대처인지, 실제 문제처리하는 코드인지 가독성도 좋지 않을 뿐더러 예외가 여러가지일때 무슨 예외상황인지도 파악하기가 어렵습니다. 또 같은 예외가 코드 여러군데에서 발생하면 같은 예외도 일일이 다 처리해야합니다. 따라서 예외처리에는 try~catch 문을 이용합니다.

사용 방법:
```
try
{
	//실행코드
}
catch (예외 객체1)
{
	//예외1에 대한 처리
}
catch (예외 객체2)
{
	//예외2에 대한 처리
}
finally
{
	// 반드시 실행되는 코드
}
```

위에서 발생했던 예외 상황에 대한 try~catch를 적용해보겠습니다.

예제코드:
```
class Program
{
    static void Main(string[] args)
    {
        int[] Numbers = new int[5] { 1, 2, 3, 4, 5 };

        for(int i=0; i<6; i++)
        {
            try
            {
                Console.WriteLine(Numbers[i].ToString());
            }
            catch (IndexOutOfRangeException e)
            {
                Console.WriteLine("예외메시지 : {0}", e.Message);
                Console.WriteLine("예외가 발생한 곳(namespace): {0}", e.Source);
                Console.WriteLine("예외가 발생한 곳(method): {0}", e.TargetSite);
                Console.WriteLine("예외가 발생한 곳(line): {0}", e.StackTrace);
            }
        }

        Console.WriteLine("예외 상황이 끝난 후.."); // 프로그램은 강제종료 되지 않음.
    }
}
```
![](https://laboputer.github.io/assets/img/csharp/07-2.PNG)

예외클래스에는 예외에 대한 여러가지 정보가 담겨있습니다. 또 예외처리를 할 경우 프로그램이 종료되지 않고 다음 코드가 정상적으로 작동되는 것을 확인하실 수 있습니다.

try블록에서 자원해제와 같은 중요한 코드를 미처 실행하지 못한 상태로 예외를 발생시키게 된다면, 이는 곧 버그를 만드는 원인이 됩니다. 예를 들어 DB Connection을 닫는 코드를 실행하지 못할 경우에는 사용할 수 있는 커넥션이 점점 줄어 DB에 연결할수 없는 상태에 도달할 수 있습니다. 따라서 finally 절을 지원합니다. finally 절은 try절이 실행된다면 어떤 경우에라도 실행됩니다. 심지어 return 문이나 throw 문이 사용되도 finally 절은 반드시 실행됩니다.

## 사용자 정의 예외 클래스 만들기
---

C#에서는 모든 예외 객체는 System.Exception 클래스로부터 상속받아야 합니다. 즉 상속 관계로 인해 모든 예외클래스는 System.Exception 형식으로 간주하여 catch절 하나면 모든 예외를 다 받을 수도 있습니다. 하지만 예외마다 다른 대처를 해야한다면, 프로그래머는 예외상황에 따라 맞는 코드를 작성해야 합니다. C#에서는 100가지 넘는 예외 클래스를 제공하지만, 사용자가 직접 예외를 발생하고 싶은 경우에는 새로운 예외 클래스를 만들면 됩니다.

예를 들면 회원가입 할때 비밀번호는 8자리 이상을 입력해야 된다거나, 영문을 섞어야 할때 이를 지키지 않은 경우, 예외를 발생시킬 수도 있습니다.

```
class ShortPasswordException : Exception
{
    public ShortPasswordException(string message) : base(message)
    {

    }

    public string Range
    {
        get;
        set;
    }
}

class Program
{
    static void Main(string[] args)
    {
        do
        {
            Console.WriteLine("비밀번호를 입력하세요!");
            string Password = Console.ReadLine();
            try
            {
                if (Password.Length < 8)
                {
                    throw new ShortPasswordException("비밀번호가 너무 짧습니다")
                    {
                        Range = "8자리 이상"
                    };
                }
                else
                {
                    break;
                }
            }
            catch (ShortPasswordException e)
            {
                Console.WriteLine("예외메시지: {0}", e.Message);
                Console.WriteLine("예외발생한곳: {0}", e.StackTrace);
                Console.WriteLine("가능범위: {0}", e.Range);
            }             
        } while (true);
    }
}
```
![](https://laboputer.github.io/assets/img/csharp/07-3.PNG)

---
프로그램의 강제종료는 최악의 버그입니다. 실제 예상하지 못한 수많은 예외가 발생하지만 예외처리를 반드시 해야 합니다. 하지만 모든 경우를 예상하지 못하니 try~catch를 모든 코드에 남용하는 경우가 많습니다. 하지만 프로그램이 버그가 발생하는 것을 확인어렵기 때문에 try~catch는 신중하게 사용하여야 합니다.