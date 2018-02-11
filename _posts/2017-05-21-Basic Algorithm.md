---
layout: post
title:  "Basic Algorithm"
date:   2017-05-21 21:46:00
categories: ProblemSolving
tags: ProblemSolving
---

* content
{:toc}

## 1. Sorting (정렬)

Data를 정렬할 때 사용하는 알고리즘 정리

**Bubble Sort**

설명이 필요 없음.

Time Complexity: O(N^2)

```
for (int i=0; i<N; i++)
 for (int j=i+1; j<N; j++)
  if(arr[i] > arr[j]) SWAP(i,j);
```

**Merge Sort**

분할정복 기법을 이용한 정렬 방법.
최악의 경우에도 NlogN의 시간복잡도를 가지지만, Merge할 때의 N개의 메모리가 더 필요하고,
상수배 연산이 QuickSort에 비해 크다.

Time Complexity: O(NlogN)

Algorithm:
1. 리스트 길이가 0 또는 1이면 정렬되어 있다고 생각한다.
2. 정렬되지 않은 리스트를 반으로 나눈다.
3. 각 부분 리스트를 재귀적으로 MergeSort한다.
4. 각각 정렬되어 있는 List1, List2를 각 앞의수로만 비교하여 List1, List2를 합친다.(정렬한다.)

```
void MergeSort(int l, int r)
{
	if (l >= r) return;
	int m = (l + r) >> 1;

	MergeSort(l, m);
	MergeSort(m + 1, r);

	int p1 = l, p2 = m + 1, p3 = l;
	while (p1 <= m && p2 <= r) tmp[p3++] = (arr[p1] < arr[p2]) ? arr[p1++] : arr[p2++];
	while (p1 <= m) tmp[p3++] = arr[p1++];
	while (p2 <= r) tmp[p3++] = arr[p2++];

	for (int i = l; i <= r; i++) arr[i] = tmp[i];
}
```

**Quick Sort**

최악의 경우 BubbleSort와 같지만 평균적으로 O(NlogN)

Pivot을 중앙에 가까운 값을 어떻게 선택하느냐가 성능을 결정한다. Randomized QuickSort, Median Of 3 등을 이용할 경우 최악의 경우가 확률적으로 거의 등장하지 않는다.
다른 NlogN 정렬 알고리즘에 비하여 별개의 메모리가 필요하지 않으며 상수계수로 인하여 매우 빠르다.

Time Complexity: Worst. O(N^2) , Average. O(NlogN)

Algorithm: 분할정복을 이용한다.
1. 리스트 중 하나의 원소를 피벗으로 정한다.
2. 피벗 앞에는 피벗보다 작은 값을, 피벗 뒤에는 피벗보다 큰 값을 배치시킨 후 리스트를 둘로 나눈다. (분할, 이때 피벗은 정렬된 후의 자리가 된다.)
3. 분할된 두개의 작은 리스트에 대하여 Recursion으로 돌린다.

분할하는 방법도 Lomuto partition, Hoare partition 등이 있는데 Hoare partition으로 정리해본다. [자세한 내용은 WIKI](https://en.wikipedia.org/wiki/Quicksort)

```
void Qsort(int l, int r)
{
	if (l >= r) return;
	int m = (l + r) >> 1;
	SWAP(m, r);

	int ll = l, rr = r - 1;
	while (ll <= rr)
	{
		while (ll <= rr && arr[r] <= arr[rr]) rr--;
		while (ll <= rr && arr[r] >= arr[ll]) ll++;
		if (ll < rr) SWAP(ll, rr);
	}
	SWAP(ll, r);
	Qsort(l, ll - 1);
	Qsort(ll + 1, r);
}
```


**Counting Sort**

모든 경우에 대한 정렬의 시간복잡도는 O(NlogN)이 가장 빠른 것으로 증명되었다.

하지만 특정한 조건 내에서는 더 빠른 시간복잡도를 가질 수 있는데, 대표적으로 Counting Sort가 있다.

Counting Sort은 다음과 같은 조건을 만족할때 성능이 좋다.
1. 데이터가 일정 범위내 있을 것 (그 범위가 작을 것)
2. 데이터가 정수일 것

데이터가 일정 범위 내 (1<= K <= 10000) 데이터가 N개(만개 이상의 큰 수) 들어온다면 비둘기 원리에 의해 N개의 데이터가 중복되는 경우가 생긴다.
즉 K범위 배열에 카운팅하여 각 인덱스의 개수만큼 결과를 뿌리면 된다.

하지만 K가 커지면 메모리 자체가 너무 많이 필요하고, N보다 크면 결국 NlogN보다 느리다.

Time Complexity : O(N+K) K가 작을 경우 O(N)이다.

```
	int cnt[10005];
	scanf("%d", &N);
	for (int i = 0, x; i < N; i++) scanf("%d", &x), cnt[x]++;
	for (int i = 0; i <= 10000; i++)
		while (cnt[i]--) printf("%d\n", i);
```

***
연습문제(기초)
1. https://www.acmicpc.net/problem/2751
2. https://www.acmicpc.net/problem/10989
3. https://www.acmicpc.net/problem/11931

연습문제(응용)


## 2. Divide and Conquer (분할정복)
분할정복은 말그대로 주어진 문제를 (1) 분할, (2) 해결하는 방법론입니다.

Algorithm:
1. 문제를 더 작은 문제로 분할하는 과정 (Divide)
2. 각 문제에 대해 구한 답을 원래 문제에 대한 답으로 병합하는 과정(Merge)
3. 더이상 답을 분할하지 않고 곧장 풀수 있는 매우 작은 문제(Base)

분할정복을 적용하여 문제를 해결할 때는 다음과 같은 특징이 있어야 합니다.
1. 문제를 둘 이상의 부분 문제로 나누는 방법이 있어야 합니다.
2. 부분 문제의 답으로 더 큰 문제의 답을 계산하는 효율적인 방법이 있어야 합니다.

즉, 부분 문제로 분할하여 구한 답을 합칠때 효율적으로 계산할 수 있는 문제에서만 적용 가능 합니다.

***
연습문제(기초)
1. https://www.acmicpc.net/problem/2104

연습문제(응용)
1. https://www.acmicpc.net/problem/1725

## 3. Dynamic Programming (동적계획법)
복잡한 문제를 간단히 여러 개의 문제로 나누어 푸는 방법론입니다.

분할 정복과 같은 접근 방식을 이용합니다. 처음 주어진 문제를 더 작은 문제들로 나눈 뒤 계산하고 이를 이용하여 더 큰 문제를 해결합니다.

분할정복과 동적계획법의 차이점은 부분 문제를 나누는 방식에 있습니다.

동적계획법에서의 어떤 부분 문제는 2번 이상을 계산하는 경우가 발생하게 되는데
이때 이를 **중복되는 부분문제(Overlapping subproblems)**라고 합니다.

동적계획법에서는 **중복되는 부분문제(Overlapping subproblems)**의 계산결과를 캐시에 저장하는 것으로 한 번 계산한 부분문제는 메모리에 저장해둬 속도를 향상시키는 알고리즘입니다.

동적계획법 알고리즘을 이용하는 문제는 보통 최적화 문제입니다. 즉 가장 좋은 최적해를 찾아내는 문제입니다.

동적계획법 알고리즘을 적용하기 위해서는 두가지 조건이 필요합니다.

1. 최적 부분 구조 (Optimal substructure) :
   큰 문제의 최적해는 반드시 부분 문제의 최적해를 이용해 풀 수 있는 문제

2. 중복되는 부분 문제 (Overlapping subproblems) :
   부분 문제가 여러번 등장하는 문제

예) 피보나치 수열 구하기, 이항계수 구하기, LIS, LCS 등

***
연습문제(기초)
1. https://www.acmicpc.net/problem/1463

연습문제(응용)

## 4. Topological Sort (위상정렬)
위상정렬은 의존성이 있는 작업들이 주어질 때 어떤 순서로 나열해야 하는지를 나타냅니다.

위상 정렬이 가능한 조건은 다음과 같습니다.
1. DAG(Directed Acyclic Graph)

즉, 유향(Directed)이면서 싸이클이 없으면(Acyclic) 위상정렬이 가능합니다.

Algorithm:
1. 각 정점 V에 대하여 필요한 선행작업의 수를 저장한 배열을 만듭니다.
2. 각 정점 V가 필요한 작업에 대한 방향 간선을 리스트로 연결합니다.
3. 선행작업이 없는 수에 대하여 큐에 넣습니다.
4. 큐가 빌때 까지 작업하나를 방문하고 그 작업이 필요한 곳의 값을 -1 해주고,
   값이 0이 된 경우에는 해당 작업도 큐에 넣습니다.

특징:
1. 큐가 빌때까지 V개 만큼 방문하지 못하면 위상 정렬이 불가능하다. (싸이클이 존재함)
2. 큐 루프를 돌면서 큐에 작업이 2개 이상 있을 경우 정렬 결과가 여러개 나올 수 있음.

각 정점 V를 한번씩만 방문하고 각 정점V에 대하여 연결간선만 세므로,

Time Complexity : O(V+E)

***
연습문제(기초)
1. https://www.acmicpc.net/problem/2252
2. https://www.acmicpc.net/problem/1516

연습문제(응용)

## 5. Grid Compression (좌표압축)

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

## 6. Hashing (해싱)

임의의 크기를 가진 데이터를 고정된 크기의 데이터로 변환하는 방법

데이터를 인덱스로 접근하면 빠른 시간에 데이터를 찾아낼 수 있는데, 이를 이용하기 위하여
숫자를 포함한(넓은 범위, 좌표압축처럼)문자열 등을 하나의 인덱스로 대칭시켜
해당 메모리에 저장한다.

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

***
연습문제(기초)
1. https://www.acmicpc.net/problem/1920

연습문제(응용)