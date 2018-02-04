---
layout: post
title:  "C# Language"
date:   2015-01-02 17:04:00
categories: Programming(C#)
tags: C#
---

* content
{:toc}

C# 프로그래밍 기초 문법
도서(뇌를 자극하는 C# 4.0 프로그래밍 - 박상현 저)의 내용을 정리합니다. 

## 1. .NET Framework

C# 이라고 하면  항상 .NET Framework 에 대한 얘기가 나오곤 합니다.

.NET Framework 는 Microsoft에서 개발한 웹 기반, 윈도우 기반 application을 개발할 수 있고, 실행시킬 수 있는 프레임워크이며, .NET 표준에 따르는 프로그래밍 언어로 개발된 프로그램의 실행환경이라고 할 수 있습니다.

.NET Framework의 구성을 그림으로 보면 다음과 같습니다.

![](./images/01.NET_framework/1-1.png)

위에서 하나씩 살펴보자면
여기서 .NET 표준을 따르는 (CLS; Common Language Specification)을 따르는 여러 언어를 제공합니다.
(대표적으로 C# 외에 VB.NET , Managed C++ , Jscript.NET , j # 등이 있다.)

Class Library 구성을 보면 다음과 같습니다.
* ASP.NET Web Forms Web Services : Web service와 Web apps를 개발 
* Windows Forms : Windows 프로그래밍
* ADO .NET and XML : XML 및 데이터베이스 활용
* Base Class Library : CLR 지원을 위한 핵심 클래스 라이브러리

### 1.1 Common Language Runtime(CLR)

.NET 언어로 작성된 프로그램을 실행하고 관리하는 실행환경이며 Java의 virtual machine에 해당한다고 볼 수 있습니다. CLR은 .NET에서 동작하는 프로그램을 적재하고, 프로그램의 동적 컴파일, 프로그램의 실행, 메모리관리 (Garbage Collection) ,프로그램의 예외처리, 언어 간의 상속 지원, COM과의 상호 운영성 지원 등을 가능케 합니다. 
.NET 언어라 하면 C#뿐만 아니라 위에서 나열하였던 VB.NET 등 .NET 표준을 따르는 여러언어들은 CLR을 통해 프로그램을 실행할 수 있게 됩니다.

![](/images/01.NET_framework/1-2png)

이것이 가능한 이유는 Intermediate Language (IL, 중간언어) 라는 기계어로 변환하기 쉬운 상태의 중간 단계의 언어로 .NET에서 번역되어 실행되기 위해 IL형태로 컴파일을 하는데, IL을 기계어로 바꾸는 번역기만 제공되면 어떤 플랫폼에서도 실행가능합니다. JIT (Just-In-Time) 컴파일러를 통해 IL을 동적으로 컴파일하는데 .NET 에서는 이와같이 프로그램을 2번 컴파일 합니다. Assembly는 이 때 IL로 컴파일된 결과 파일들을 패키징 한 것을 말합니다.(.exe는 여러개의 class와 Main프로그램을 포함하고, dll은 class만을 포함한다고 생각하면 된다.)

또 .NET 언어가 지켜야 하는 스펙으로 Common Language Specifications (CLS) 로 다른 .NET 언어로 작성된 것도 호환되어 동작 가능하게 합니다. 이 과정에서 Common Type System (CTS) 라는 .NET 언어마다 Data type이 다를 수 있는데,이를 언어나 시스템 환경에 관계없이 동일한 Data type을 유지하기 위한 규약이 사용되기도 합니다.

쉽게 요약하면, C# 컴파일러는 우리가 작성한 코드를 IL로 작성된 실행파일을 만들게 되고, 이 파일을 실행시키면 CLR이 IL을 읽어 다시 OS에 이해할수 있는 코드로 컴파일하여 실행시킵니다. 이렇게 서로 다른 언어가 만나는 지점이 IL언어이고, CLR이 다시 자신이 설치되어 있는 플랫폼에 최적화시켜 컴파일한 후 실행하게 됩니다.

이렇게 플랫폼에 최적화된 코드를 만들어 내는 장점이 있으나, 단점은 실행시에 이루어지는 컴파일의 비용부담이 되겠습니다.



## 2. 데이터 다루기

**데이터 형식 (Data Type)**

C#에서 제공하는 데이터 형식은 숫자, 텍스트 뿐만 아니라, 이미지, 소리까지 다룰 수 있는 데이터 형식을 제공합니다. 데이터 형식은 기본 데이터형식(Primitive Type) 과 복합 데이터형식 (Complex Data Type) 으로 나눌 수 있습니다.
 * 기본 데이터형식은 흔히 생각하는 int , double 과 같은 개념
 * 복합 데이터형식은 기본데이터 형식을 기본으로 하는 구조체나 클래스 등

기본 데이터형식과 복합 데이터형식을 다시 값 형식과 참조 형식으로 분류할 수 있습니다.


**값 형식 (Value Types), 참조 형식 (Reference Types)**

값 형식은 메모리 영역 중 스택(Stack) 영역 에 저장되며, 선언 되었던 코드 블록이 끝나면 스택영역에서 데이터가 제거됩니다.
스택에 쌓인 데이터들이 코드 블록이 끝나는 순간 제거 된다는 것은 값 형식의 장점이자 단점이 됩니다.

코드 블록이 끝나는 순간 데이터가 제거되기 때문에 메모리영역이 전부 회수되지만, 코드 블록안에서만 사용가능합니다.

참조 형식은 메모리 영역 중 힙(Heap) 영역에 데이터를 저장하는데, 힙 영역은 코드 블록과 상관 없이 데이터는 사라지지 않습니다. 참조 형식의 변수는 힙과 스택을 동시에 이용합니다. 힙 영역에는 데이터의 값을 저장하고, 스택 영역에는 이 데이터의 주소를 저장합니다. 스택 영역은 코드 블록이 끝나는 순간 사라지지만 ( 데이터의 주소), 힙 영역에 데이터의 값은 사라지지 않습니다. 
C# 에서는 CLR의 가비지 컬렉터 (Garbage Collector) 가 힙에 사용하지 않는 객체를 자동으로 제거해줍니다.

**object 형식**

 C#은 모든 데이터 형식을 다룰 수 있는 object형식이 존재합니다. 모든 데이터 형식 ( 기본 데이터 형식 뿐만 아니라 모든 복합 데이터 형식, 프로그래머가 만드는 데이터형식 까지도) 자동으로 object 형식으로부터 상속받게 하였기 때문입니다.

정수 형식은 short 와 int 의 처리가 다르고, 부동 소수점 형식은 float 와 double 가 처리가 다른데도 가능한 이유는 박싱(boxing) 과 언박싱(unboxing) 이라는 메커니즘이 있기 때문입니다.

**박싱(boxing), 언박싱(unboxing)**

object 형식은 참조 형식이기 때문에 힙에 데이터를 할당합니다.
하지만 object 형식에 값 형식의 데이터 (int와 같은)를 할당하면 object 형식은 박싱을 수행해서 데이터를 힙에 할당합니다. 이 object 형식을 다시 값 형식의 변수에 할당하면, object 형식의 데이터를 언박싱하여 데이터 값을 값 형식의 변수에 할당합니다.

예를 들어, 다음과 같은 코드가 있습니다.
```
object a= 10;
int b = (int) a;
```
이럴 경우 10을 박싱하여 힙에 저장하고, a는 박싱한 10의 주소를 갖습니다. 다시 b에 저장하고자 할때 a는 언박싱 되어 10이 b에 저장됩니다.

**열거 형식(Enumerator)**

종류는 같지만, 다른 값을 갖는 상수들을 선언해야 할때 유용합니다.

예를 들면, 메시지 박스를 띄웠을 때 사용자로 부터 받는 응답이 Yes, No, Cancel 과 같을 때 상수로 선언하여 컨트롤하면 쉽습니다.
```
enum 열거형식명 : 기반자료형 { 상수1, 상수2, .... }
```
기반 자료형은 정수 계열(byte, short, int, long 등)만 사용할 수 있습니다. 생략할 경우 컴파일러가 int 형으로 사용합니다. 

enum 의 여러가지 사용방법은 다음과 같습니다.

```
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
 
namespace CsharpStudy
{
    class Program
    {
        enum DialogResult { YES=1 , NO , CANCEL }
 
        static void Main(string[] args)
        {
            Console.WriteLine(DialogResult.YES);
            Console.WriteLine(DialogResult.NO);
            Console.WriteLine(DialogResult.CANCEL);
            Console.WriteLine();
            Console.WriteLine((int)DialogResult.YES);
            Console.WriteLine((int)DialogResult.NO);
            Console.WriteLine((int)DialogResult.CANCEL);
            Console.WriteLine();
            Console.WriteLine( (DialogResult) 1);
            Console.WriteLine( (DialogResult) 2);
            Console.WriteLine( (DialogResult) 3);
        }
    }
}
```
[Image]

**Nullable 형식**

C#에서는 메모리 공간에 어떤 값이든 넣도록 강제합니다. 하지만 어떤 값도 가지지 않는 변수가 필요할 때가 있습니다. 0 이 아니라 아예 비어있는 변수, 즉 null 한 변수를 말합니다.
```
데이터형식 ? 변수이름;
```

```
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
 
namespace CsharpStudy
{
    class Program
    {
 
        static void Main(string[] args)
        {
            int? a = null;
            double? b = null;
            Console.WriteLine(a.HasValue);
            Console.WriteLine(a == b);
            a = 3;
            Console.WriteLine();
            Console.WriteLine(a.HasValue);
            Console.WriteLine(a != null);
            Console.WriteLine(a.Value);
        }
    }
}
```

[Image]

**var 형식**
C#은 변수나 상수에 대해 깐깐하게 형식 검사를 하는 강력한 형식의 언어 (Strong Typed Language) 입니다.

이는 프로그래머의 실수를 줄여줍니다. 하지만 int, uint, long, ulong 등 수많은 형식을 외워야 하기 때문에 단점도 존재합니다. 이를 극복하기 위해 var 키워드를 통해 컴파일러가 해당 변수의 형식을 알아서 지정해줍니다.
```C
var a= 3;  // a는 int형식
var b= "Hello"; // b는 string 형식
```
 

var 형식은 지역 변수로만 사용할 수 있습니다. 그리고 클래스의 필드를 선언할 때는 반드시 명시적인 형식을 선언해야 합니다.

클래스의 필드는 보통 생성자에서 초기화하게 되는데 var키워드로 선언하면 무슨 형식인지 컴파일러가 알 수 없기 때문입니다.

참고로 C#에서는 전역변수를 지원하지 않습니다. 코드의 가독성을 해치고 오류를 낳는 원흉으로 지적되었기 때문에 전역변수를 지원하지 않게 되었습니다.

 
다음과 같이 선언하면 var 와 object 가 헷갈릴 수 있으나 전혀 다른 개념입니다. 
```
object a= 10;
var b = 10;
```
위 코드가 컴파일되어 실행하면  object 형식에서 CLR은 20을 박싱해서 힙에 넣고 a가 힙을 가리키게 됩니다.

var 형식에서 컴파일 시점에 컴파일러가 적합한 데이터형식을 파악하여 int b=10; 으로 되어 CLR은 int b=10; 으로 스택에 10을 올립니다.

## 3. Method (메소드)

**메소드(Method)**

 객체지향 프로그래밍 언어에서 사용하는 용어로 C와 C++에서는 함수(Function) 이라 불렸고, 파스칼에서는 프로시져(Procedure)라고 불렸습니다. 또는 서브루틴(Subroutine), 서브프로그램(Subprogram)이라고 부르기도 합니다.
메소드를 사용할때 매개변수 전달방식에 대해서 설명하고자 합니다.

 
**참조에 의한 매개 변수 전달 (Call by reference)**
C에서는 포인터를 이용하여 직접 데이터의 주소를 매개변수로 전달하여 함수에서 값을 컨트롤하였습니다.
C#에서는 프로그래머가 메모리를 참조한다는 것 자체가 위험하다고 판단하였기 때문에 포인터를 사용할 수 없게 했습니다. (하지만 설정으로 포인터를 사용가능하게 만들 수는 있습니다.)

Call by reference 를 이용하여 두개의 데이터의 값을 바꾸는 Swap 메소드 를 만들어 보겠습니다.

```
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
  
namespace CsharpStudy
{
    class Program
    {
        static void Swap(ref int x, ref int y)
        {
            int temp = x;
            x = y;
            y = temp;
        }
       
        static void Main(string[] args)
        {
            int x = 3;
            int y = 5;
            Swap(ref x, ref y);
            Console.WriteLine("{0} , {1}", x, y);
        }
    }
}
```

[Image]
주의하실점은 매개변수를 전달할때, 함수에서 매개변수를 받을 때 ref 키워드를 붙여서 명확한 표현을 해주셔야 합니다.

그리고, C#에서는 전부 클래스 단위로 움직입니다. Main 메소드도 마찬가지로 클래스 안에 있습니다만, static 키워드로 정적으로 존재하기 때문에 Main에서 Swap메소드를 사용하기 위해서 Swap메소드도 static으로 선언되어야 합니다.

**출력 전용 매개변수**
메소드가 두개 이상의 반환 값을 갖고 싶을 때 사용합니다.
예를 들어 어떤 두 숫자를 가지고 몫과 나머지를 동시에 반환하고 싶을 때는 다음과 같이 함수를 만들면 됩니다.
```
void Divide ( int a , int b, ref int quotient, ref int remainder)
{
	quotient = a/b;
	remainder = a%b;
}
```
하지만 C#은 out 키워드를 이용한 출력전용 매개변수를 제공해 더욱 안전한 방법으로 사용할 수 있습니다.

```
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
  
namespace CsharpStudy
{
    class Program
    {
        static void Divide(int x, int y, out int quotient, out int remainder)
        {
            quotient = x / y;
            remainder = x % y;
        }
 
        static void Main(string[] args)
        {
            int x = 5;
            int y = 3;
            int quo, rem;
            Divide(x, y, out quo, out rem);
            Console.WriteLine("quo: {0} , rem: {1}", quo, rem);
        }
    }
}

```
[image]

out 키워드를 사용할 경우

1. 메소드가 out 변수에 값을 저장하지 않을 경우 컴파일러가 에러를 발생합니다.
2. 호출된 메소드에서는 입력된 out 매개변수를 읽을 수 없고, 오직 쓰기만 가능합니다.(다른 용도로 사용되는 것을 금지합니다)

컴파일러가 에러를 발생한다는 것은 긍정적인 일입니다. 그 이유는 프로그래머에게 쉽게 에러를 고칠 수 있게 하지만, 런타임 오류는 프로그래머의 논리력으로 추적하여 고쳐야 하기 때문입니다.

**메소드 오버로딩(Method Overloading)**
메소드 오버로딩은 하나의 메소드 이름에 여러가지 구현을 overloading(과적) 하는 것입니다.

오버로딩을 할 경우 비슷한 기능을 하는 메소드를 매개 변수의 자료형이나 개수 등을 보고 여러가지 메소드 중에서 컴파일을 할때 컴파일러가 실행할 메소드를 선택하기 때문에, 프로그램의 성능 저하도 없을 뿐더러 한가지 메소드 이름으로 비슷한 기능의 메소드를 전부 묶을 수가 있습니다.

프로그래밍을 하다보면 클래스의 이름, 메소드의 이름 등 이름만 봐도 다른사람이 직관적으로 바로 알 수 있게 정하는 것이 아주 중요합니다. 이 때 유용하게 쓰입니다.


**가변길이 매개변수**
매개 변수의 수만 다른 같은 기능을 하는 메소드를 오버로딩 하고 싶을 때 사용할 수 있습니다.
메소드 오버로딩을 이용할 수도 있겠지만, 가변길이 매개변수는 수만 다른 경우 갯수만큼 메소드를 오버로딩 하는 것보다 한번에 형식은 같지만 변수의 개수가 정해져있는 것이 아닐 경우 가변길이 매개변수가 유용하게 쓰일 수 있습니다.

예를 들면 모든 수를 합하는 Sum 메소드를 가변길이 매개변수를 이용하여 만들어 보겠습니다.
```
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
  
namespace CsharpStudy
{
    class Program
    {
        static int Sum(params int[] args)
        {
            int sum = 0;
            foreach(int num in args)
            {
                sum += num;
            }
            return sum;
        }
 
        static void Main(string[] args)
        {
            int result1 = Sum(2, 3, 5);
            int result2 = Sum(2, 3, 5, 7);
            int result3 = Sum(2, 3, 5, 7, 9);
            Console.WriteLine("result1 : {0} result2: {1} result3: {2}", result1, result2, result3);
        }
    }
}
```
[image]

주의할 점은 순서에 상관없이 매개변수 이름에 할당하기 때문에 하나라도 사용할 경우 전부 명명된 매개변수로 값을 할당해줘야야 합니다.

**선택적 매개변수**
매개 변수에 디폴트 값을 넣어 매개변수를 전달하지 않을 경우 디폴트 값이 자동으로 할당됩니다.
명명된 매개변수와 함께 이용할 경우 매개변수가 많은 메소드에서 넣고 싶은 매개변수만 원하는 데이터를 쉽게 넣을 수 있기 때문에 유용하게 쓰입니다.

주의할 점은 메소드 오버로딩과 선택적 매개변수를 동시에 사용할 때의 모호함이 있으므로 같이 사용하면 안됩니다.
```
void Method(string str1="", string str2="");
void Method(string str1);
```
위와 같은 두 가지 메소드가 있을 경우
Method("abc"); 로 메소드 호출 할때 여러분이 컴파일러라면 어떤 함수를 호출해야 할지 판단이 되나요?

