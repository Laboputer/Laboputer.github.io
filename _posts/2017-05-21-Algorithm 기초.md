---
layout: post
title:  "Algorithm 기초"
date:   2017-05-21 21:46:00
categories: ProblemSolving
tags: ProblemSolving
---

* content
{:toc}

## 1. Sorting (정렬)
Data를 정렬할 때 사용하는 알고리즘 정리

### Bubble Sort

설명이 필요 없음.

Time Complexity: O(N^2)

```
for (int i=0; i<N; i++)
 for (int j=i+1; j<N; j++)
  if(arr[i] > arr[j]) SWAP(i,j);
```



### Merge Sort

분할정복 기법을 이용한 정렬 방법.

최악의 경우에도 NlogN의 시간복잡도를 가지지만, Merge할 때의 N개의 메모리가 더 필요하고, 상수배 연산이 QuickSort에 비해 크다.

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

### Quick Sort

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

### Counting Sort

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

## 2. Topological Sort (위상정렬)

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
3. https://www.acmicpc.net/problem/1766

연습문제(응용)
1. https://www.acmicpc.net/problem/2056
2. https://www.acmicpc.net/problem/1948

## 3. Divide and Conquer (분할정복)

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

## 4. Dynamic Programming (동적계획법)

복잡한 문제를 간단히 여러 개의 문제로 나누어 푸는 방법론입니다.

동적계획법 알고리즘을 이용하는 문제는 보통 최적화 문제입니다. 즉 가장 좋은 최적해를 찾아내는 문제입니다. 완전탐색을 하면서 중복되는 계산을 메모리에 저장하는 방법이라고 볼 수 있습니다.

분할 정복과 같은 접근 방식을 이용합니다. 처음 주어진 문제를 더 작은 문제들로 나눈 뒤 계산하고 이를 이용하여 더 큰 문제를 해결합니다.

분할정복과 동적계획법의 차이점은 부분 문제를 나누는 방식에 있습니다.

동적계획법에서의 어떤 부분 문제는 2번 이상을 계산하는 경우가 발생하게 되는데 이때 이를 **중복되는 부분문제(Overlapping subproblems)**라고 합니다.

동적계획법에서는 **중복되는 부분문제(Overlapping subproblems)**의 계산결과를 캐시에 저장하는 것으로 한 번 계산한 부분문제는 메모리에 저장해둬 속도를 향상시키는 알고리즘입니다.

DP문제들을 접근할 때에는 큰 문제가 어떤 부분문제를 발생시키는가, 그리고 다른 방식(정의를 수정해서)으로 부분문제를 발생시킬 때 최소화시킬 수 있는 방안을 찾으면서 Table을 정의하고 그에 맞는 점화식을 이용하여 푼다.


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
1. https://www.acmicpc.net/problem/4095

### 4.1 Recursive DP vs Iterative DP
DP를 구현함에 있어서 Recursive DP (재귀적 DP)는 가독성도 좋고, 점화식을 그대로 옮기면(의존 관계나 순서를 고려할 필요 없이)되기 때문에 쉽습니다. 또 재귀적으로 구하면서 일부의 답만을 구하고도 최적값을 알수도 있습니다. 하지만 재귀적으로 구현하게 되면 스택 메모리를 예상하기가 어렵고, 우선 상수시간이 많이 소요됩니다.

점화식이 순차적(연속적)으로 이루어진 경우 Iterative DP로 for문만으로도 구할 수 있는 경우가 있는데,
이때 우선 시간복잡도는 동일하지만 상수시간이 빠를 수 밖에 없습니다. 이 외에도, 순차적으로 구하기 때문에 Sliding Window(슬라이딩 윈도우) 기법을 적용하여 필요한 메모리만 가지고도 구할 수 있습니다.
추가적으로 한정된 경우에 사용할 수 있는 방법으로 Linear Transform(선형 변환)을 이용하여 행렬식을 이용하여 시간복잡도 자체를 줄일 수 있는 경우도 간혹 있습니다. 예를 들면 피보나치 수열을 행렬식을 이용하여 O(logN)만에 구할 수 있습니다.

간단하게 정리하면,

Recursive DP
1. 장점: 좀 더 직관적인 코드를 짤 수 있다.
2. 장점: 부분 문제 간의 의존관계나 계산 순서에 대해 고민할 필요가 없다.
3. 장점: 전체 부분 문제중 일부의 답만 필요할 경우에는 더 빠르게 동작한다.
4. 단점: 슬라이딩 윈도 기법을 쓸 수 없다.
5. 단점: 스택 오버플로를 조심해야 한다.

Iterative DP
1. 장점: 구현이 대개 더 짧다.
2. 장점: 재귀 호출에 필요한 부하가 없기 때문에 조금 더 빠르게 동작한다.
3. 장점: 슬라이딩 윈도 기법을 쓸 수 있다.
4. 단점: 구현이 좀 더 비직관적이다.
5. 단점: 부분 문제 간의 의존 관계를 고려하여 계산되는 순서를 고민해야 한다.


## 5. Greedy (탐욕적 방법)
최적해를 구할 때 쓰이는 근사 알고리즘 방법론입니다.

Greedy는 근사 알고리즘 방법론이기 때문에 많은 경우에 최적해를 찾을 수 없습니다. 다시 말하면 주어진 상황에서 항상 최적의 선택을 하는 알고리즘인데, 지역적 최적해를 선택해가며 만든 최종해는 최적해라는 보장이 없습니다. 하지만 특정한 상황에서는 탐욕적 알고리즘이 최적해를 구해낼 수 있습니다.

1. Greedy Choice Property (지역적 최적의 선택이 이후 선택에 영향을 주지 않는다)
2. Optimal Substructure (문제에 대한 최적해는 부분문제의 최적해를 포함한다)

위 두 조건이 성립하는 경우에는 Greedy 알고리즘을 적용할 수 있습니다. 따라서 Greedy 알고리즘으로 PS를 접근할 때에는 항상 최적해를 가져온다는 알고리즘 정당성 증명이 가장 중요합니다.

동적계획법으로도 이 Greedy 알고리즘과 동일한 최적해를 구할 수 있지만, 많은 경우에 메모리 부담도 없고 모든 경우를 계산해보는 것보다 지역적 최적선택을 하는 것이 효율적이기 때문에 Greedy 알고리즘을 적용합니다.


***
연습문제(기초)

연습문제(응용)
