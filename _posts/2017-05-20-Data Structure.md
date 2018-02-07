---
layout: post
title:  "Data Structure"
date:   2017-05-20 21:46:00
categories: ProblemSolving
tags: ProblemSolving
---

* content
{:toc}

## Stack (스택)

한쪽 끝에서 삽입과 삭제가 일어나는 자료구조 (Last In First Out)

배열로 Push, Pop, Top 연산을 구현하면 된다.

```
struct STACK
{
	int arr[MAXN];
	int sz = 0;

	void Push(int x) { arr[sz++] = x; }

	int Pop()
	{
		if (sz == 0) return -1;
		else return arr[--sz];
	}

	int Top()
	{
		if (sz == 0) return -1;
		else return arr[sz - 1];
	}
};
```

<hr>

연습문제(기초)
1. https://www.acmicpc.net/problem/10828

연습문제(응용)
1. https://www.acmicpc.net/problem/9012
2. https://www.acmicpc.net/problem/2841


## Queue (큐)
한쪽 끝에서는 삽입, 다른 한쪽 끝에서는 삭제가 일어나는 자료구조 (FIFO)

Head, Tail이 같이 늘어나는 구조로 구현하면 메모리를 비효율적으로 사용하게 된다.
따라서 원형큐로 Push, Pop, Front 연산을 구현한다.

```
struct Queue
{
	Data arr[MAXN];
	int sz = 0;
	int front = 0;
	int rear = 0;

	bool Full() { return (sz == MAXN - 1) ? 1 : 0; }
	bool Empty() { return !sz; }

	Data Top()
	{
		return arr[rear];
	}

	void Push(Data x)
	{
		if (Full()) return;
		arr[front] = x;
		front++, sz++;
		front %= MAXN;
	}

	void Pop()
	{
		if (Empty()) return;
		rear++, sz--;
		rear %= MAXN;
	}
};
```


연습문제(기초)
1. https://www.acmicpc.net/problem/10845

연습문제(응용)

## Priority Queue (우선순위 큐)

큐와 삽입은 똑같지만 삭제할 경우 가장 우선순위가 높은 데이터가 나온다.
구현은 Heap 이라는 자료구조를 이용하여 구현할 수 있다.

Heap은 완전이진트리 이며 배열로 구현하기 쉽다.

Heap의 특징
 1. root(index : 1)의 우선순위가 가장 크다.
 2. 부모노드는 자식노드보다 우선순위가 높다.

Heap은 우선순위가 가장 큰 데이터는 O(1)만에 뽑을 수 있고, 삽입과 삭제연산이 O(logN)에 가능하다. 하지만 root의 우선순위가 가장 높다는 사실 빼고는 나머지 데이터에 대한 아무런 정보를 얻을 수 없다.

Algorithm:
1. root는 1번 인덱스
2. node의 왼쪽 자식은 node*2, 오른쪽 자식은 node*2+1 인덱스
3. 삽입연산 : 가장 마지막 노드에 삽입한 후 해당 노드의 부모의 우선순위가 높을 때까지 스왑
4. 삭제연산 : 루트와 마지막 인덱스를 스왑한다.
   while (1)
   4-1 자식이 없을 경우 : 완료
   4-2 왼쪽 자식만 있을 경우 : 왼쪽 자식이 해당 노드보다 우선순위가 높으면 스왑
   4-3 자식이 둘다 있을 경우 : 둘 중 높은 우선순위를 가진 노드가 해당 노드보다 높으면 스왑
   
Time Complexity : 삽입: O(logN), 삭제:O(logN)

```
struct PriorityQueue
{
	Data arr[MAXN];
	int sz = 0;

	void Swap(int i, int j) { Data t = arr[i]; arr[i] = arr[j]; arr[j] = t; }

	void Push(Data x)
	{
		sz++;
		arr[sz] = x;
		int cur = sz, p = cur >> 1;
		while (cur != 1 && arr[cur].prior > arr[p].prior) 
          Swap(cur, p), cur >>= 1, p >>= 1;
	}

	void Pop()
	{
		Swap(1, sz);
		sz--;
		int cur = 1, c = cur << 1;
		while (c <= sz)
		{
			if (c + 1 <= sz && arr[c].prior < arr[c + 1].prior && arr[cur].prior < arr[c + 1].prior) 
              Swap(cur, c + 1), cur = c + 1, c = cur << 1;
			else if (c <= sz && arr[cur].prior < arr[c].prior) 
              Swap(cur, c), cur = c, c = cur << 1;
			else break;
		}
	}

	Data Top() { return arr[1]; }
};
```

연습문제(기초)
1. https://www.acmicpc.net/problem/1966

연습문제(응용)
