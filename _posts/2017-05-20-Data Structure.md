---
layout: post
title:  "Data Structure"
date:   2017-05-20 21:46:00
categories: ProblemSolving
tags: ProblemSolving
---

* content
{:toc}

## 1. Stack (스택)

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


## 2. Queue (큐)
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

## 3. Priority Queue (우선순위 큐)

우선순위 큐는 우선 순위가 가장 높은 데이터가 가장 먼저 나올 수 있도록 고안된 자료구조

큐와 삽입은 똑같지만 삭제할 경우 가장 우선순위가 높은 데이터가 나온다.
구현은 Heap 이라는 자료구조를 이용하여 구현할 수 있다.

Heap은 완전이진트리 이며 배열로 구현하기 쉽다.

Heap의 특징
 1. 완전 이진트리
 2. 부모노드는 자식노드보다 우선순위가 높다.

Heap은 우선순위가 가장 큰 데이터는 O(1)만에 뽑을 수 있고, 삽입과 삭제연산이 O(logN)에 가능하다. 이를 이용한 Heap Sort도 가능하다. 삽입을 N번 하고 삭제를 N번하면 정렬된 데이터가 나오게 된다.

Heap은 균형잡힌 이진트리보다 좀 더 엄격한 구조로 H를 가장 작게 유지 할 수 있으며, 균형 잡힌 이진트리는 루트 밑의 데이터도 의미가 있다. (중위순회하면 정렬된다.)
하지만 root의 우선순위가 가장 높다는 사실 빼고는 나머지 데이터에 대한 아무런 정보를 얻을 수 없다.

Algorithm:
1. root는 1번 인덱스, node의 왼쪽 자식은 node*2, 오른쪽 자식은 node*2+1 인덱스
2. 삽입연산 : while(cur부터 root까지) 현재위치 cur가 부모pnt보다 우선순위가 높으면 스왑
3. 삭제연산 : 루트와 마지막 인덱스를 스왑한다.
   while (루트부터 자식이 없을 때 까지)
   4-1 왼쪽 자식만 있을 경우 : 왼쪽 자식이 해당 노드보다 우선순위가 높으면 스왑
   4-2 자식이 둘다 있을 경우 : 둘 중 높은 우선순위를 가진 노드가 해당 노드보다 높으면 스왑
   
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
2. https://www.acmicpc.net/problem/11279
3. https://www.acmicpc.net/problem/11286

연습문제(응용)
1. https://www.acmicpc.net/problem/1655


## 4. Union-Find (Disjoint-set)

서로소 집합을 표현할 수 있는 자료구조.

서로 배타적인 관계 집합을 나타낼 수 있으며, 어떤 두 노드가 주어지면 같은 집합인지 다른 집합인지 표현이 가능하다.


Algorithm:
1. 최초에 각 노드를 자기 자신을 가리키게 초기화한다. (모든 노드는 독립적)
2. Find 연산 : x가 자기 자신과 같아질 때 까지 재귀적으로 구할 수 있다.
3. Union 연산 : Find(x)와 Find(y)가 다르면 하나로 합쳐준다.
   
Time Complexity : 정확하게 표현하기 어렵고 연산당 O(logN) 정도로 생각해도 무방하다.

```
struct UF
{
	int set[MAXN];
	UF() { for (int i = 0; i < MAXN; i++) set[i] = i; }

	int Find(int x) { 
		return set[x] = (set[x] == x ? x : Find(set[x])); 
	}
	void Union(int a, int b) { 
		set[Find(a)] = Find(b); 
	}
};

```

연습문제(기초)
1. https://www.acmicpc.net/problem/1717

연습문제(응용)

## 5. Graph (그래프 기초)
그래프는 객체들의 상호관계를 표현하기 위해 고안된 자료구조이다.

그래프는 현실 세계의 사물이나 추상적인 개념 간의 연결 관계를 모델링하여 표현한다.
알고리즘 문제에서는 그래프로 모델링하여 효율적으로 계산할 수 있는 방법으로 접근한다.

**그래프의 정의**

그래프는 G(V,E)는 어떤 자료나 개념을 표현하는 정점(vertex)들의 집합 V와 이들을 연결하는 간선(edge)들의 집합 E로 구성된 자료구조

**그래프의 표현**

그래프는 다음과 같은 3가지 방법으로 표현할 수 있다.

1. 인접 행렬
2. 인접 리스트
3. 간선들의 집합

인접 행렬은 G[MAXN][MAXN] 메모리를 잡아놓은 상태에서 각 정점마다 연결된 객체를 표현한다.
장점은 노드간 연결상태나 Edge를 보는 연산이 O(1)로 매우 빠르지만, 단점은 메모리를 N^2만큼 잡아먹기 때문에 Edge가 많이 없는 경우에는 매우 비효율적이다.

인접 리스트는 List[MAXN]를 잡아 놓은 상태에서 각 정점마다 연결된 객체를 리스트로 연결한다.
장점은 메모리를 효율적으로 사용하며 연결된 Edge들만 접근할 수 있으나, Edge가 많아질 경우 인접 행렬보다 오히려 많은 메모리가 필요하다.

간선들의 집합은 그 자체로 그래프로 볼 수 있는데 보통은 1.2.로 표현되겠지만 문제상황에 따라 간선들의 집합만으로도 구하기도 한다.

**그래프의 순회**

그래프의 각 노드들을 완전탐색하는 것.

방법은 2가지가 있다.
1. DFS (Depth First Search, 깊이우선 탐색)
2. BFS (Breadth First Search, 너비우선 탐색)

DFS는 스택을 이용하여 현재 노드의 방문하지 않은 첫번째 노드를 계속적으로 방문한다. 함수 호출도 스택 구조를 띄기 때문에 DFS를 구현하면 재귀적으로 짧은 코드로 표현할 수 있다.


BFS는 큐를 이용하여 현재 노드에서 방문하지 않은 노드들을 모두 방문한 후에 다음 단계로 넘어간다. BFS를 구현하기 위해서 별도의 큐를 구현해야 한다.

DFS Code:
```
void DFS(int n)
{
	check1[n] = true;
	path1.push_back(n);
	for (int i = 0; i < G[n].size(); i++)
	{
		if (!check1[G[n][i]])
			DFS(G[n][i]);
	}
}
```

BFS Code:
```
void BFS(int n)
{
	queue<int> q;
	q.push(n);
	check2[n] = true;
	path2.push_back(n);
	while (!q.empty())
	{
		int x = q.front();
		q.pop();

		for (int i = 0; i < G[x].size(); i++)
		{
			if (!check2[G[x][i]])
			{
				check2[G[x][i]] = true;
				path2.push_back(G[x][i]);
				q.push(G[x][i]);
			}
		}
	}
}
```

***

연습문제(기초)

1. https://www.acmicpc.net/problem/1260

연습문제(응용)

## 6. Tree (트리 기초)
트리는 계층 구조를 표현하기 위한 자료구조이다.

트리는 그래프에서 사이클이 없는 관계로도 볼 수 있다. 트리의 각 노드의 부모는 하나이다. 그래서 계층구조로 표현할 수 있다.

트리가 유용한 이유는 재귀적 특성을 가지고 있기 때문이기도 한다. 재귀호출을 이용하여 쉽게 문제를 풀 수 있는 경우가 많다.  그래프와 마찬가지로 트리도 순회가 있는데, 일반적인 경우는 그래프와 비슷한 DFS, BFS로 순회할 수 있지만 이진트리에서는 전위순회(Preorder), 중위순회(Inorder), 후위 순회(Postorder)로 표현 하기도 한다.

트리가 자주 사용되는 자료구조는 이진 검색트리(BST)가 있는데, 이진검색트리가 선형적인 구조면 검색연산이 O(N)이기 때문에, 높이를 맞춰 O(logN)이 되기 위해 균형잡힌 이진트리를 구현하게 된다. 이 균형잡힌 이진트리는 AVL, Red-Black 등이 있다.