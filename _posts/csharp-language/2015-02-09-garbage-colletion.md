---
layout: post
title:  "[C# 기초] 14. 가비지 컬렉션(Garbage collection) - 마지막"
subtitle:   "garbage-collection"
categories: csharp
tags: language
---

> `[C# 기초]` 시리즈의 마지막 포스팅으로 할당한 메모리를 자동으로 회수해주는 가비지 컬렉터에 대해 배웁니다. 

## 가비지 컬렉션(Garbage collection)
---

C 또는 C++ 언어로 프로그래밍을 하다보면 malloc(), new를 통해 힙영역에 메모리를 할당받아 사용하고 메모리를 다시 반환해주어야 하는 골치아픈일이 많았습니다. 하지만 C#에서는 가비지 컬렉터(Garbage Collector)가 자동으로 메모리를 해제해줍니다.

물론 C/C++ 프로그래밍에서 정확하고 안정적으로 코드를 짤 수 있다면 메모리를 효율적으로 사용할 때만 사용하고 버리고 싶을 때는 제거할 수 있겠지만, 사람인 이상 실수를 할 수 밖에 없습니다. 더군다나 프로그램이 커질수록 혼자만 코딩을 하는게 아니기 때문에 메모리 관리는 아주 큰 이슈입니다. 또, C/C++ 기반의 프로그램을 실행하는 C-runtime 은 힙에 객체를 할당하기 위해 비싼 비용을 치르는 문제(메모리를 할당하기 위해 순차적으로 탐색하면서 객체를 담을만한 메모리 블록을 찾고, 메모리를 쪼개고, 메모리 블록의 리스트를 재조정하는; 탐색,분할,재조정의 오버헤드)를 가지고 있습니다.

C#에서는 가비지컬렉터가 사용하지 않는 객체를 알아서 판단하여 메모리를 회수합니다. 하지만, 가비지 컬렉터도 소프트웨어이기 때문에 CPU와 메모리를 사용합니다. 따라서 가바지 컬럭테가 최소한으로 자원을 사용하게 하는 것도 프로그램의 성능을 높이는 방법입니다. 

그래서 가비지 컬렉션의 메커니즘을 이해하고, 이를 바탕으로 어떻게 코딩지침을 세울 것인가를 정리하고자 합니다.

## 가비지 컬렉터는 어떻게 메모리를 회수할까?
---

C#으로 작성한 프로그램을 실행하면 CLR은 프로그램을 위한 일정 크기의 메모리를 확보합니다. C-runtime 처럼 메모리를 쪼개는 일을 하지 않고 그냥 일정 메모리 공간을 확보해서 하나의 관리되는 힙(Managed Heap)을 마련합니다. 객체를 할당하게 되면 메모리에 순서대로 할당하게 됩니다. (CLR은 객체가 위치할 메모리를 할당하기 위해 공간을 쪼개만든 리스트를 탐색하는 시간도 소요하지 않고, 공간을 나눈 뒤에 리스트를 재조정하는 작업도 하지 않습니다)

참조 형식의 객체가 할당될 때는 스택 영역에는 힙의 메모리 주소를, 힙 영역에 실제 값이 할당된다고 했습니다.
그럼 객체가 할당된 코드블록이 끝나면 스택 영역의 메모리가 회수되고, 힙 영역에 값이 쓰레기가 됩니다. 여기서 회수된 스택의 객체를 루트(Root) 라고 부릅니다.
.NET 응용 프로그램이 실행되면 JIT 컴파일러가 이 루트들을 목록으로 만들고 CLR은 이 루트 목록을 관리하며 상태를 갱신하게 됩니다.

가비지 컬렉터는 힙영역의 임계치에 가까워지면

1. 모든 객체가 쓰레기라고 가정합니다.( 루트 목록 내의 어떤 루트도 메모리를 가리키지 않는다고 가정)
2. 루트 목록을 순회하면서 참조하고 있는 힙 객체와의 관계 여부를 조사합니다. 즉 어떤 루트와도 관계가 없는 힙의 객체들
   은 쓰레기로 간주됩니다.
3. 쓰레기가 차지하고 있던 메모리가 회수되면, 인접 객체들을 이동시켜 차곡차곡 채워서 정리합니다.

CLR의 메모리도 구역을 나누어 메모리에서 빨리 해제될 객체와 오래 있을 것 같은 객체를 따로 담아 관리합니다.

구체적으로 CLR은 메모리를 0세대, 1세대, 2세대 3가지로 분리하고, 0세대는 빨리 사라질 객체, 2세대는 오래 남아있을 것 같은 객체들을 위치시킵니다.

정리하면

0. 응용 프로그램을 실행하면 0세대부터 할당된 객체들이 차오르기 시작합니다.
1. 0세대 가비지 컬렉션 임계치에 도달하면 0세대에 대해 가비지 컬렉션을 수행합니다. 여기서 살아남은 객체는 1세대로 옮겨집니다. 
2. 1번과정을 계속 반복하다보면, 1세대 가비지 컬렉션이 임계치에 도달하게 되고, 1세대에 대해 가비지 컬렉션을 수행합니다. 여기서 살아남은 객체는 다시 2세대로 옮겨집니다.
3. 2번과정도 지속되다보면 2세대 가비지 컬렉션이 임계치에 도달합니다.
   2세대 가비지컬렉션이 임계치에 도달하면 0,1,2 세대 전체 가비지컬렉션 (Full Garbage Collection) 을 수행합니다. 

프로그램이 오래 살아남는 개체들을 마구 생성하면, 2세대 힙이 가득차게 될 것입니다. 2세대 힙이 가득차게 되면 CLR은 응용프로그램의 실행을 멈추고 전체 가비지 컬렉션을 수행하면서 메모리를 확보하려 하기 때문에 응용 프로그램이 차지하는 메모리가 많을 수록 프로그램이 정지하는 시간도 그만큼 늘어나게 됩니다.

## 코딩지침
---

가비지 컬렉션의 메커니즘을 바탕으로 효율적인 코드를 작성하기 위한 방법이 있습니다.

* 객체를 너무 많이 할당하지 마세요.
* 너무 큰 객체 할당을 피하세요.
* 너무 복잡한 참조 관계는 만들지 마세요.
* 루트를 너무 많이 만들지 마세요.
* 객체를 너무 많이 할당하지마세요.

### 객체를 너무 많이 할당하지 마세요.

객체 할당 코드를 작성할 때 꼭 필요한 객체인지 필요 이상으로 많은 객체를 생성하지 않는지 고려하라는 말입니다.

### 너무 큰 객체 할당을 피하세요.

CLR은 85KB 이상의 대형 객체를 할당하기 위한 대형 객체 힙 (Large Object Heap ; LOH) 을 따로 유지하고 있습니다.
대형 객체가 0세대에 할당하면 가비지 컬렉션을 보다 자주 수행하기 때문입니다.
대형 객체 힙은 객체의 크기를 계산한 뒤 여유공간이 있는지 힙을 탐색하여 할당합니다. 또한, 대형 객체 힙은 해제된 공간을 인접 객체가 채우는 것이 아니라 그대로 둡니다. 대형 객체를 복사하여 옮기는 비용이 너무 비싸기 때문입니다. 이로 인해 큰 공간 사이사이가 메모리를 낭비하게 됩니다. (C-runtime 방식의 문제점과 비슷합니다)

### 너무 복잡한 참조관계는 만들지 마세요.

참조 관계가 많은 객체는 가비지 컬렉션 후에 살아남아 있을 때가 문제입니다. 살아 남은 객체는 세대를 옮기기 위해 메모리 복사를 진행하는데, 참조 관계가 복잡할 경우 객체를 구성하고 있는 각 필드 객체간의 참조관계를 조사하여 메모리 주소를 전부 수정해야 되기 때문에 탐색과 수정의 오버헤드가 발생합니다. 또한 A객체가 2세대인데 A 객체안에 B객체를 이제막 생성하여 0세대로 되었다면, A의 인스턴스는 2세대에 있고 B 필드를 참조하는 메모리는 0세대에 위치하게 됩니다. 이때 0세대 가비지 컬렉션이 수행된다면 B필드가 수거될 수 있습니다. 하지만 CLR은 쓰기 장벽(Write barrier)이라는 장치를 통해서 B필드가 루트를 갖고 있는 것으로 간주하게 해서 수거 되지 못하게 합니다. 이 쓰기 장벽을 생성하는 데 오버헤드가 크다는 것이 문제가 됩니다.

### 루트를 너무 많이 만들지 마세요.

가비지 컬렉터는 루트 목록을 돌면서 쓰레기를 수거하기 때문에 적을 수록 성능에 유리합니다.

---
마지막으로 C#의 가비지 컬렉터에 대해 정리하였습니다. 저도 C++ 언어를 배우면서 [메모리 누수](https://ko.wikipedia.org/wiki/%EB%A9%94%EB%AA%A8%EB%A6%AC_%EB%88%84%EC%88%98) 때문에 작은 기능을 하는 프로그램도 에러로 인해 터져버렸던 기억이 있습니다. 메모리에 대해 신경쓰지 않는 것은 매우 큰 장점이지만 C# 에서도 메모리 누수가 발생하지 않는 것은 아닙니다. 이 가비지 컬렉터에 대해 이해하고 있어야 메모리가 확보되지 않는 현상이 발생했을 때 해결할 수 있을 것입니다.  