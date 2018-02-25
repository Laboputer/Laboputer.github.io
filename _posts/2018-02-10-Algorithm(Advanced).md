---
layout: post
title:  "Algorithm(Advanced)"
date:   2017-05-21 21:46:00
categories: ProblemSolving
tags: ProblemSolving
---

* content
{:toc}

## 1. Knapsack

배낭 문제로 알려진 문제로, 일정 가치와 무게가 있는 짐을 넣을때 가치의 합이 최대가 되도록 짐을 고르는 방법을 찾는 문제이다.

같은 문제여도 여러가지 조건에 따라 다양한 알고리즘이 사용되어 공부하기 좋기 때문에 정리해본다.

**Fractional Knapsack**

물건을 쪼갤 수 있을 때 배낭 문제.

Greedy 알고리즘을 적용해도 최적해를 구할 수 있다.

Algorithm :
 무게 대비 가격 효율이 가장 좋은 것을 최대한 쪼개담는다.


** 01 Knapsack **

물건을 쪼갤 수 없을 때 배낭 문제.

Greedy 알고리즘이 최적해를 보장하지 않아서 다른 알고리즘으로 구해야 한다.

이 문제는 제한이 없는 조건에서는 NP-Complete 문제로 완전탐색을 해서 구할 수 있다. Time Complexity : O(2^N)

물건의 개수(N), 배낭의 크기(T)라고 할 때, N이나 T에 특정한 조건이 있을 경우에는 다항시간안에 구할 수 있는데, 조건마다 효율적인 알고리즘이 있어서 문제를 나누어볼 수 있다.

T의 크기가 작은 경우 Dynamic Programming(동적계획)으로 접근할 수 있다.

다음과 같은 점화식이 성립한다.
 1. D[i][j] : j크기 안에서 i번째 물건까지 고려했을 때 최대값 으로 정의한다.
 2. D[i][j] = MAX(D[i-1][j] , D[i-1][T-W[i]]) (단, T-W[i] < 0인 경우에는 D[i][j] = D[i-1][j])


재귀적으로 구할 경우 함수호출 오버헤드 및 스택 오버플로우 우려가 있지만 모든 D[i][j]의 경우를 탐색하지 않으므로 좀 더 빠를 수도 있다. 하지만 반복적으로 구할 경우 일반적으로 빠르지만 모든 D[i][j]을 구하므로 느린 경우가 생긴다.

또 이번 문제에서 반복적DP로 구할 경우에는 특징을 보면 알겠지만 이전값을 저장할 필요가 없다. 다시말하면 메모리를 [N][T]를 잡을 필요없이 [T]로도 구할 수가 있게 된다. 단, 제대로된 값을 이용해야 되기 때문에 반복문의 순서를 잘 고려해야 한다.

Time Complexity : O(N*T)


재귀적 방법:
```
#include <stdio.h>
#define MAX(a,b) ((a) < (b)) ? (b) : (a)
#define MAXN 105
#define MAXT 10005

int N, T;
int W[MAXN], V[MAXN];

int d[MAXN][MAXT];
int F(int n, int t)
{
	if (n < 1) return 0;
	int& ref = d[n][t];
	if (ref != -1) return ref;
	if (t - W[n - 1] < 0) return ref=F(n - 1, t);
	else return ref=MAX(F(n - 1, t), F(n - 1, t - W[n - 1]) + V[n - 1]);
}

int main()
{
	scanf("%d%d", &N, &T);
	for (int i = 0; i < N; i++) scanf("%d%d", &W[i], &V[i]);

	for (int i = 0; i <= N; i++) for (int j = 0; j <= T; j++) d[i][j] = -1;
	printf("%d\n", F(N, T));
	return 0;
}
```

반복적 방법:
```
#include <stdio.h>
#define MAX(a,b) ((a) < (b)) ? (b) : (a)
#define MAXN 105
#define MAXT 10005

int N, T;
int W[MAXN], V[MAXN];
int d[MAXN][MAXT];

int main()
{
	scanf("%d%d", &N, &T);
	for (int i = 1; i <= N; i++) scanf("%d%d", &W[i], &V[i]);

	for (int i = 1; i <= N; i++) for (int j = 0; j <= T; j++)
	{
		if (j - W[i] >= 0) d[i][j] = MAX(d[i - 1][j], d[i - 1][j - W[i]] + V[i]);
		else d[i][j] = d[i - 1][j];
	}
	printf("%d\n", d[N][T]);

	return 0;
}
```


공간복잡도를 최소화시킨 반복적DP:

```
#include <stdio.h>
#define MAX(a,b) ((a) < (b)) ? (b) : (a)
#define MAXN 105
#define MAXT 10005

int N, T;
int W[MAXN], V[MAXN];

int d[MAXT];

int main()
{
	scanf("%d%d", &N, &T);
	for (int i = 1; i <= N; i++) scanf("%d%d", &W[i], &V[i]);

	for (int i = 1; i <= N; i++) for (int j = T; j >=W[i]; j--)
			d[j] = MAX(d[j], d[j - W[i]] + V[i]);

	printf("%d\n", d[T]);

	return 0;
}
```

**Bound Knapsack**

01Knapsack에서 각 물건 당 수량이 여러개인 경우이다.

각기 다른 물건으로 N의 개수를 늘려 구할 경우 위와 같이 일반적인 방법으로도 구할 수 있다. 하지만, 개수만큼 메모리도 큰 폭으로 증가하기 때문에 DP중에 개수만큼 반복을 더 하여 해결할 수 있다.

Time Complexity : O(NTK)

위의 반복문 코드에서 이렇게만 바꾸면 된다.
```
int W[MAXN], V[MAXN], K[MAXN];

	for (int i = 1; i <= N; i++) for (int j = 0; j <= T; j++)
	{
		for (int k = 0; k < K[i]; k++)
		{
			if (j - W[i] * k < 0) d[i][j] = d[i - 1][j];
			else d[i][j] = d[i - 1][j - W[i] * k] + V[i] * k;
		}
	}
```

**Unbounded Knapsack**

01Knapsack에서 각 물건의 개수가 제한이 없는 경우.

Bound Kanpsack에서 K가 커지면 복잡도가 무한히 커지므로 N번째 물건까지 한번씩 사용해보면서 최대값을 계속 저장하는 방법을 이용한다.

Time Complexity : O(N*T)

Algorithm:
1. D[T] : T 공간에서 최대값 으로 정의한다.
2. (1<=j<=T, 1<=i<=N) D[j] = MAX(D[j], D[j-W[i]] + V[i]으로 구한다.

Code:
```
#include <stdio.h>
#define MAX(a,b) ((a) < (b)) ? (b) : (a)
#define MAXN 105
#define MAXT 10005

int N, T;
int W[MAXN], V[MAXN];
int d[MAXT];
int main()
{
	scanf("%d%d", &N, &T);
	for (int i = 1; i <= N; i++) scanf("%d%d", &W[i], &V[i]);

	for (int j = 1; j <= T; j++)
	{
		for (int i = 1; i <= N; i++)
		{
			//이때 정렬되어 있을 경우 -> if (j - W[i] < 0) break;
			if (d[j] < d[j - W[i]] + V[i]) d[j] = d[j - W[i]] + V[i];
		}
	}
	printf("%d\n", d[T]);
	return 0;
}
```

**Meet in the middle**

정확히 데이터를 둘로 나누어서 각각의 데이터를 완전탐색한 후 해결하는 알고리즘.

분할정복이 아니고 단 한번 데이터를 둘로만 나눈 후 양쪽 결과를 이용하여 구하는 것입니다.
자세한 내용은 [여기를 참조](https://www.geeksforgeeks.org/meet-in-the-middle/)

01Knapsack에서 T값이 매우커서 메모리를 잡을 수 없는 경우 (동적계획이 불가능한 경우)

이때에는 완전검색O(2^N)으로 구해야 하는데요. N의 값이 30~50인 경우에는 구할 수 있는 방법이 있습니다.

Time Complexity : O( 2^(N/2) * N)

Algorithm:
1. N개의 물건을 둘로 쪼갠다 (A[]=N/2, B[]=N/2)
2. A[]을 이용하여 완전 검색하여 결과를 리스트에 저장한다. (다른 B[]도 마찬가지로 구한다)
3. List A, List B를 각각 사용한 가방 크기를 기준으로 정렬한다.
4. List B에서 [0, i]까지 최대 가치를 구한다.
5. MAX(A[i].v + max[0, T-A[i].t].v in list B)) (1<=i<=N/2)

Knapsack뿐만 아니라 비슷한 류 O(2^N) 문제나 냅색에서 특정 조건에 따라 리스트 데이터끼리 합칠 때 효율적으로 구할 수 있는 방법이 있을 경우 적용할 수 있습니다.


연습문제(기초)

1. https://www.acmicpc.net/problem/14728
2. https://www.acmicpc.net/problem/1699
3. https://www.acmicpc.net/problem/1450

연습문제(응용)





