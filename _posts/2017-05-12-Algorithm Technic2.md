---
layout: post
title:  "Algorithm Technic2"
date:   2017-05-12 11:46:00
categories: ProblemSolving
tags: ProblemSolving
---

* content
{:toc}

## 1. 비트마스크 연산
비트 연산을 이용하여 효율적으로 계산할 수 있는 방법입니다.

1. AND 연산 :  (a & b)
2. OR 연산 : (a l b)
3. XOR 연산 : (a ^ b)
4. NOT 연산 : (~a)

a를 왼쪽으로 b비트 시프트 : a << b

비트마스크를 이용한 집합구현

int set; , p ( 0<= p <=29)

![](https://raw.githubusercontent.com/laboputer/laboputer.github.io/master/images/Problem_Solving/01.PNG)

## 2. Bit Manipulation
특정한 수를 Bit로 표현했을때 1의 개수를 구하는 방법입니다.

1의 개수를 구하는 방법은 3가지 단계로 최적화시킬 수 있습니다.
1번 방법은 단순히 완전탐색하는 방법이고, 2번은 비트열의 특징을 이용하여 x & (x-1)을 할 경우 마지막의 1비트가 사라지는 현상을 이용하여 구합니다.
3번은 분할정복을 이용하는데 자세한 내용은 [Wiki](https://en.wikipedia.org/wiki/Hamming_weight)

Algorithm :
1. 각 비트 열에서 1이 존재하는 가를 봅니다. O(N) (비트열의 개수만큼의 복잡도)
2. x & (x-1) 이 0이 될때까지 반복 횟수를 카운팅합니다. O(N) (비트열의 1 개수만큼의 복잡도)
3. Hamming weight 방법으로 구합니다. O(1) (12번의 연산)

Code:
```
int BitCount1(int x)
{
	int cnt = 0;
	for (int i = 0; (1 << i) <= x; i++)
		if (x & (1 << i)) cnt++;
	return cnt;
}

int BitCount2(int x)
{
	int cnt = 0;
	while (x)
	{
		x = (x & (x - 1));
		cnt++;
	}
	return cnt;
}

int BitCount3(int x)
{
	x = x - ((x >> 1) & 0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
```


## 3. Grid Compression (좌표압축)

좌표평면 상에서 주어진 N개의 점보다 구간의 범위가 훨씬 큰 상황에서 사용할 수 있는 테크닉

좌표평면 상에서 구간의 범위가 매우 길면 Seqment Tree로 구간 정보를 구할 때 쓰는 자료구조들을 메모리문제로 사용할 수 없으므로 구간의 범위가 큰 경우 N개가 한정되어 있을때 좌표를 압축하여 사용하는 좌표만 인덱스를 붙여 사용할 수 있다.

어떻게 보면 해쉬와도 유사할 수 있다.

Algorithm:

1. N개의 데이터를 정렬한다.
2. (데이터 중복이 있을 경우) 중복된 데이터를 제거한 X개의 데이터로 바꾼다(X<N)
3. Binary Search를 이용하여 실제 데이터가 몇번째인지 찾는다.


```
for (int i = 0; i < N; i++) scanf("%d", &a[i]), b[i] = a[i];
int pre = b[0]; int cnt = 1;
for (int i = 1; i < N; i++) if (pre != b[i]) c[cnt++] = pre = b[i];
SORT(0, cnt - 1);

```

***

연습문제(기초)

1. https://www.acmicpc.net/problem/2517

연습문제(응용)

1. https://www.acmicpc.net/problem/10256


## 4. Hashing (해싱)

임의의 크기를 가진 데이터를 고정된 크기의 데이터로 변환하는 방법

데이터를 인덱스로 접근하면 빠른 시간에 데이터를 찾아낼 수 있는데, 이를 이용하기 위하여 숫자를 포함한(넓은 범위, 좌표압축처럼)문자열 등을 하나의 인덱스로 대칭시켜 해당 메모리에 저장한다.

**Hash Function** : Key값을 임의의 인덱스로 변경하는 것. 즉 k -> H(k)

Hash Function은 특정 수로 변환하지만 그 수로 다시 key값을 찾는 것은 불가능하다.

또, 다른 Key값이 같은 해싱값을 만들어 낼 수 있는데, 이를 **Collusion(충돌)**이라고 한다.


좋은 해싱은 적은 메모리로 Collusion을 최소화하는 Hash Function을 만들어 내는 것이다.

Division method:

Key를 어떤 수 k로 나눈 나머지를 이용한다. (여기서 k는 테이블 크기로 정한다)
여기서 k값에 따라 해싱의 성능이 크게 결정된다.

k의 크기는 전체 Key의 수의 3배 정도가 적당하다고 한다. k는 소수여야 나눴을때 나머지가 골고루 퍼질 확률이 높고, k는 2의 지수승에 가까우면 좋다고 한다.

K는 테이블의 3배, 2지수승에 가까운 소수


```
#define _K 93419
	int hashing(long long x)
	{
		long long idx = 0;
		while (x)
		{
			long long v = x % 10;
			idx = (idx << 5) - idx + v;
			idx %= _K;
			x /= 10;
		}
		return idx;
	}
```

Collusion:

충돌이 일어났을 때 방법은 리스트로 체이닝 하는 방법, 충돌나면 다음 인덱스에 저장하는 방법, 더블 해싱하는 방법 등 여러가지가 있다.

만약에 원본 데이터가 클 때에는 충돌일 때 원본 데이터와 같은지 비교연산이 오래 걸리 경우에는 Key값을 여러개 구하여 Value로 저장해놓으면 원본 확인하는데 빠른 시간안에 해결할 수 있다.


자세한 내용은 [Geeksforgeeks 참조](https://www.geeksforgeeks.org/hashing-set-2-separate-chaining/)

리스트로 체이닝 하는 방법 예:

```
struct node
{
	long long key;
	int value;
	node* next = NULL;
};

struct list
{
	int sz = 0;
	node* head = new node();
    
	void push(long long x)
	{
		node* nn = new node();
		nn->key = x;
		nn->value = 1;
		nn->next = head;
		head = nn;
		sz++;
	}
};

struct Hash
{
	list map[HASH];
	void push(long long x) { map[hashing(x)].push(x); }
	int hashing(long long x)
	{
		long long idx = 0;
		while (x)
		{
			long long v = x % 10;
			idx = (idx << 5) - idx + v;
			idx %= HASH;
			x /= 10;
		}
		return idx;
	}
};

```
리스트를 동적할당 대신 배열로 구현하는 방법

```
#define MAXN 100005
#define H1 917121

struct Node
{
	long long key;
	int value;
	int next;

	Node() {}
	Node(long long k, int v, int n) { key = k, value = v, next = n; }
};

Node Shared[MAXN];
int Ccnt = 0;

struct List
{
	int head = -1;
	int sz = 0;

	void Push(long long x)
	{
		Shared[Ccnt] = { x,1 ,head };
		head = Ccnt++;
		sz++;
	}
};

struct Map
{
	List Table[H1];

	Map()
	{
		Ccnt = 0;
		for (int i = 0; i < H1; i++) Table[i].sz = 0;
	}

	int Hashing(long long x)
	{
		long long hash = 0;
		while (x)
		{
			hash = (hash << 5) - hash + x % 10;
			hash %= H1;
			x /= 10;
		}
		return hash;
	}
    
	void Push(long long x)
	{
		int idx = Hashing(x);
		Table[idx].Push(x);
	}

	bool Exist(long long x)
	{
		int idx = Hashing(x);
		int cur = Table[idx].head;
		for (int i = 0; i < Table[idx].sz; i++)
		{
			if (Shared[cur].key == x) return true;
			cur = Shared[cur].next;
		}
		return false;
	}
};
```

***

연습문제(기초)

1. https://www.acmicpc.net/problem/1920

연습문제(응용)

1. https://www.acmicpc.net/problem/10256