---
layout: post
title:  "[C# 기초] 7) 컬렉션(Collection)"
subtitle:   "collection"
categories: csharp
tags: language
---

> 배열, 리스트, 해시테이블 등과 같이 비슷한 성격을 가진 데이터를 모아놓는 자료구조를 컬렉션이라고 합니다.

## 컬렉션(Collection)
---

컬렉션은, 같은 성격을 띠는 데이터의 모음을 담는 자료구조를 말합니다. 같은 성격의 데이터의 묶음들을 배열로 사용해보신적이 있으실 겁니다. C#은 컬렉션이라는 개념으로 배열뿐만 아니라, ArrayList, Queue, Stack, Hashtable 등 여러가지 컬렉션 클래스를 제공합니다.

예를 들어, Hashtable은 Key와 Value가 쌍으로 이루어진 데이터를 다룰때 사용합니다. 자료구조에서도 배우겠지만, 해싱으로 아주 빠른 탐색속도를 자랑합니다.

예제 코드:
```
using System.Collections;

class Program
{
    static void Main(string[] args)
    {
        Hashtable Ht = new Hashtable();
        Ht["하나"] = "One";
        Ht["둘"] = "Two";
        Ht["셋"] = "Three";
        
        Console.WriteLine("요소 직접 접근: {0}", Ht["셋"]);
            
        Console.WriteLine("==Hashtable의 모든 데이터 접근==");
        foreach(var value in Ht.Values)
        {
            Console.WriteLine(value);
        } // Two, One, Trre

        Console.WriteLine("==Hashtable의 모든 Key 접근==");
        foreach (var key in Ht.Keys)
        {
            Console.WriteLine(key);
        } // 둘, 하나, 셋
    }
}
```

## 인덱서(Indexer)
---

자료구조를 사용해보면 'arr[i]'와 같이 인덱스로 접근하는 방식이 편리합니다. 인덱서는 새로 생성한 클래스의 필드로 존재하는 컬렉션에 대해 클래스의 인덱스로 접근할 수도 있습니다.

사용방법:
```
class 클래스이름
{
	한정사 인덱서형식 this[형식 index]
    {
		get
		{
			// index를 이용하여 내부 데이터 반환
		}
		set
		{
		    // index를 이용하여 내부 데이터 저장
		}
	}
}
```

예제 코드:
```
using System.Collections;

class MyList
{
    private int[] array;

    public MyList()
    {
        this.array = new int[3];
    }

    public int this[int index]
    {
        get
        {
            return array[index];
        }
        set
        {
            if(index >= array.Length)
            {
                Array.Resize<int>(ref array, index + 1);
                Console.WriteLine("array resized : {0}", array.Length);
            }
            array[index] = value;
        }
    }

    public int Length
    {
        get
        {
            return array.Length;
        }
    }

}

class Program
{
    static void Main(string[] args)
    {
        MyList mylist = new MyList();

        Console.WriteLine("==데이터 저장==");
        for(int i=0; i<5; i++)
        {
            mylist[i] = i;
        } // array resize : 4 , 5

        Console.WriteLine("==데이터 출력==");
        for (int i = 0; i < mylist.Length; i++) // foreach를 사용할 수 없습니다.
        {
            Console.WriteLine(mylist[i]);
        }    // 0 , 1, 2, 3, 4
    }
}
```

위와 Mylist라는 객체에 대한 데이터를 인덱스로 접근할 수 있게 만들었습니다. 하지만 조금만 생각해보면 foreach문은 불가능하다는 사실을 아실겁니다. Mylist라는 객체만 보고 요소하나하나를 어떻게 판단할 것이며, 어떻게 순회할지에 대한 약속이 전혀 없기 때문입니다.

그렇다면 foreach문이 가능한 객체는 어떻게 만들 수 있을까요?

## IEnumerable, IEnumrator: foreach문이 가능한 객체만들기
---

foreach문이 가능하기 위해서는 IEnumerable, IEnumerator를 상속받아 구현하면 됩니다. 즉, IEnumerable, IEnumerator 를 구현하면서 약속을 정해주어야하는 것입니다.

### IEnumerable

|            메소드           |              설명              |
|:---------------------------:|:------------------------------:|
| IEnumerator GetEnumerator() | IEnumerator 형식의 객체를 반환 |

### IEnumerator

|          메소드         |                                                                 설명                                                                 |
|:-----------------------:|:------------------------------------------------------------------------------------------------------------------------------------:|
| boolean MoveNext()      | 다음 요소로 이동합니다. 컬렉션의 끝을 지난 경우에는 false, 이동이 성공한 경우에는true를 반환합니다.                                  |
| void Reset()            | 컬렉션의 첫 번째 위치의 '앞'으로 이동합니다. 첫번째 위치가 0번째라면 -1번으로 이동합니다.  MoveNext()를 호출한 다음에 이루어집니다.  |
| Object Current { get; } | 컬렉션의 현재 요소를 반환합니다.

예제 코드:
```
using System.Collections;

class MyList : IEnumerable , IEnumerator
{
    private int[] array;
    int position = -1;
    public MyList()
    {
        this.array = new int[3];
    }

    public int this[int index]
    {
        get
        {
            return array[index];
        }
        set
        {
            if(index >= array.Length)
            {
                Array.Resize<int>(ref array, index + 1); 
                Console.WriteLine("array resized : {0}", array.Length);
            }
            array[index] = value;
        }
    }

    public object Current
    {
        get
        {
            return array[position];
        }
    }

    public void Reset()
    {
        position = -1;
    }

    public bool MoveNext()
    {
        if( position == array.Length -1)
        {
            Reset();
            return false;
        }

        position++;
        return (position < array.Length);
    }
    public IEnumerator GetEnumerator()
    {
        for(int i=0; i <array.Length; i++)
        {
            yield return (array[i]);
        }
    }
}

class Program
{
    static void Main(string[] args)
    {
        MyList mylist = new MyList();

        Console.WriteLine("==데이터 저장==");
        for(int i=0; i<5; i++)
        {
            mylist[i] = i;
        }

        Console.WriteLine("==데이터 출력==");
        foreach(var item in mylist) // foreach가 가능해졌습니다.
        {
            Console.WriteLine(item);
        }    
    }
}
```

---
상황에 따라 배열과 리스트 같은 다양한 자료구조가 필요한데 이 컬렉션을 통해 많은 자료구조를 사용할 수 있을 뿐만 아니라 각각의 자료구조에 대해 변환도 쉽게 할 수 있습니다.