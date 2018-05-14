---
layout: post
title:  "Algorithm 중급"
date:   2018-02-10 21:46:00
categories: ProblemSolving
tags: ProblemSolving
---

* content
{:toc}

## 1. Knapsack

배낭 문제로 알려진 문제로, 일정 가치와 무게가 있는 짐을 넣을때 가치의 합이 최대가 되도록 짐을 고르는 방법을 찾는 문제이다.

같은 문제여도 여러가지 조건에 따라 다양한 알고리즘이 사용되어 공부하기 좋기 때문에 정리해본다.

### Fractional Knapsack

물건을 쪼갤 수 있을 때 배낭 문제.

Greedy 알고리즘을 적용해도 최적해를 구할 수 있다.

Algorithm :
 무게 대비 가격 효율이 가장 좋은 것을 최대한 쪼개담는다.


### 01 Knapsack

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

### Bound Knapsack

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

### Unbounded Knapsack

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

### Meet in the middle

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

***
연습문제(기초)

1. https://www.acmicpc.net/problem/14728
2. https://www.acmicpc.net/problem/1699
3. https://www.acmicpc.net/problem/1450
4. https://www.acmicpc.net/problem/1208

연습문제(응용)

## 2. Shortest Path (최단경로)
그래프에서 주어진 두 정점을 연결하는 가장 짧은 경로의 길이.

다양한 그래프의 종류와 특성에 따라 최적화된 많은 최단경로 알고리즘이 존재합니다.

정점 간에 가중치가 없는 경우 최단 경로는 BFS로 O(N) 복잡도로 구할 수 있지만, 가중치가 있는 경우엔 다릅니다. 이 경우에 대하여 다뤄보고자 합니다.

먼저, 최단경로에서 음수 가중치를 갖는 간선의 존재 여부가 중요합니다. 먼저 양수 가중치에 대한 알고리즘을 알아보겠습니다.

### Dijkstra Algorithm (다익스트라)

한 정점에서 다른 정점(1:N)의 최단경로를 구하는 알고리즘

다익스트라 알고리즘은 BFS처럼 시작점에서 부터 시작하여 dist[N]을 갱신시키는 과정에서 마지막 dist[N]에 최종적으로 1:N의 최단경로가 구해집니다.

Algorithm:
1. 시작점에서부터 다른 모든 정점간의 거리를 계산한 dist[N]을 정의한다.
2. 1.에서 dist[N] 중 가장 짧은 값을 가진 정점 i는 시작점에서 최단경로이다.
2-1 (2번에서 dist[N] 중 가장 짧은 값을 선택하는 것을 힙 구조를 이용한다.)
3. 시작점과 2.에서 정해진 정점i에서 모든 정점 간의 거리를 계산한 dist[N]을 갱신한다.
4. (2-3)N 번 반복하면 된다.

위의 알고리즘은 힙구조를 이용하면 O(ElogE) 또는 O(ElogV) 복잡도를 가지며, 만약 V가 작거나 E가 매우 큰 경우에는 힙 구조를 사용하지 않는 것이 오히려 더 빠를 수도 있습니다.
힙구조를 사용하지 않는 경우는 dist[N] 중 가장 짧은 정점에 대한 최단거리를 확정하는 과정으로 진행하면 O(V^2) 복잡도를 가집니다.

Time Complexity : O(V^2 + ElogE + ElogV)

Code:
```
	dist[X] = 0;
	ok[X] = 1;
	int now = X;
	for (int ii = 0; ii < N; ii++)
	{
		for (int i = 1; i <= N; i++) dist[i] = MIN(dist[i], dist[now] + W[now][i]);
		int mn = 2e9, idx = -1;
		for (int i = 1; i <= N; i++) if (!ok[i] && mn > dist[i]) mn = dist[i], idx = i;
		ok[idx] = 1;
		now = idx;
	}
	for (int i = 1; i <= N; i++) printf("%d \n", dist[i]);
```

***
연습문제(기초)
1. https://www.acmicpc.net/problem/1916
2. https://www.acmicpc.net/problem/11779
3. https://www.acmicpc.net/problem/1753

연습문제(응용)
1. https://www.acmicpc.net/problem/1261



### Bellman-Ford Algorithm (벨만-포드)

음의 가중치를 허용하는 한 정점에서 다른 정점(1:N)의 최단경로를 구하는 알고리즘

다익스트라는 음의 가중치가 있을 경우 최단경로를 구할 수 없습니다. 하지만 벨만포드 알고리즘은 음의 가중치를 포함해도 최단경로를 구합니다. 만약 그래프에 음의 사이클이 있을 경우에는 최단경로 자체가 정의되지 않습니다. 벨만-포드는 이 음의 사이클 존재여부도 구할 수 있습니다.

벨만-포드는 N개의 최단거리 상한선을 가지고 유지하면서 어떤 간선이 있으면 상한선을 줄일 수 있는가? 로 접근합니다. M개의 간선을 돌면서 상한선을 갱신하는 과정(Relax)을 반복합니다.

이 Relax과정은 단 한 가지 최단경로 특성을 가지고 접근합니다.

 * dist[u] <= dist[v] + w(u,v) (여기서 dist[u], dist[v]는 시작점 s로부터 최단 거리)

위의 특성은 귀류법으로 간단히 증명됩니다. dist[u]가 더 큰 경우 dist[v] + w(u,v)인 거리가 존재하므로 dist[u]가 최단거리임이 모순이기 때문입니다. 따라서 이 Relax과정을 반복하면 최단거리를 구할 수 있음이 자명합니다.

이 Relax 과정은 최대 V-1번 반복됩니다. 왜냐하면 모든 간선을 통하여 갱신할 경우 최소한 1개의 정점은 최단거리가 구해지기 때문입니다. (여기서, Relax과정이 V번 이상 반복되면 음의 사이클이 존재함을 의미합니다.)


또한, 각 정점을 완화시킨 간선을 저장하여 모으면 최소 스패닝트리가 되는데, 이 트리를 따라 한 정점에서 시작정점(루트)까지의 경로가 시작점부터 해당 정점까지의 최단거리 경로가 됩니다.

주의할 점은 시작점으로부터 도달할 수 없는 정점인 경우 임의의 큰값 dist[v]==2e9 임으로는 알 수 없습니다. 음의 간선으로 그 값이 조금 줄어들 수 있기 때문에 dist[v] >= 2e9-M 로 적당히 큰 값 M을 빼준 것 보다는 클 경우로 생각해야 합니다. 물론 2e9-M 값 자체가 나올 수 없는 큰 값이어야 함.

Time Complexity : O(VE)

Algorithm :
 1. dist[i]에 상한선(무한대값)을 넣는다.
 2. dist[i] = MIN(dist[i], dist[i], dist[x] + W[x][i]) (x는 M개의 간선 중 출발선)
 3. 2번에서 M개의 간선 동안 한번도 갱신하지 못할 경우 종료합니다. (최단경로)

Code:
```
	bool flag = true;
	while (flag)
	{
		flag = false;
		for (int i = 0; i < M; i++)
			if (path[v][e[i].b] > path[v][e[i].a] + e[i].r)
            	path[v][e[i].b] = path[v][e[i].a] + e[i].r, flag = true;
	}
```

***
연습문제(기초)

1. https://www.acmicpc.net/problem/11657
2. https://www.acmicpc.net/problem/1865

연습문제(응용)


### Floyld-Warshall Algorithm (플로이드 워셜)

모든 정점(N:N) 간 최단경로를 구하는 알고리즘

플로이드 워셜 알고리즘은 모든 정점간 최단 경로를 빠르게 구할 수 있는 알고리즘입니다.
음의 가중치가 있을 경우에도 동작하며, 그 핵심 아이디어는 경유점입니다.

어떤 정점 간 최단 경로는 경유점 v을 지나거나, 지나지 않거나 인데 이를 수식으로 표현하면
 * D{S}(a,b) = MIN( D{S-v}(a,b), D{S-v}(a,v) + D{S-v}(v,b))

이 됩니다. 즉 전체 정점S에 대하여 v를 경유할 때와 경유하지 않을때를 보면 동적계획법으로 풀수 있게 됩니다.

플로이드 워셜 알고리즘은 보통 N번 다익스트라 또는 벨만포드보다 빠르게 동작합니다. 내부의 복잡한 수식이 없기 때문에 시간복잡도는 같지만 상수배가 작기 때문입니다. 

플로이드 워셜 알고리즘 또한 Dist[a][b] 를 갱신한 경유점을 저장해두면 쉽게 최단거리의 경로를 구해낼 수 있습니다.

Time Complexity : O(V^3)

Algorithm :
 1. 경유점 v에 대하여 모든 정점 간 거리 dis[a][b] 값을 구한다.
 2. 경유점 v를 모든점으로 확장한다. (즉 바깥쪽 for문이 경유점이다.)

Code:
```
	for (int v = 1; v <= N; v++)
		for (int a = 1; a <= N; a++)
			for (int b = 1; b <= N; b++)
				if (w[a][b] > w[a][v] + w[v][b])
                	w[a][b] = w[a][v] + w[v][b], p[a][b]=v;
```

***
연습문제(기초)

1. https://www.acmicpc.net/problem/11404
2. https://www.acmicpc.net/problem/11780
3. https://www.acmicpc.net/problem/1389

연습문제(응용)
1. https://www.acmicpc.net/problem/1507

## 3. Lowest Common Ancestor (LCA)
트리가 주어졌을때 임의의 두 점의 가장 가까운 공통조상을 찾는 문제

트리에서 임의의 두 점은 공통조상을 가진다. 루트도 공통조상이겠지만 루트에서 내려가면서 두 점으로 가기위해 갈라지는 바로 그 지점이다.

만약 루트가 변경되면 공통조상은 달라지지만, 정점 경로(정점간 거리)는 동일하다.

Algorithm:
1. DFS를 통해 각 정점의 Parent와 루트로부터 Depth를 구한다.
2. 두 정점(a,b)에 대해 LCA는 Depth가 큰 정점b을 작은 정점a과 같을 떄까지 올린 정점c를 구한다.
3. 만약 정점 a와 c가 같은 경우 a가 LCA가 된다.
3-1. a와 c가 다른 경우 정점a와 c를 동시에 올려가면서 같아 질때 까지 올라간다.

위의 알고리즘은 한 쌍의 LCA를 구하는데 O(N) 시간이 걸린다. 최악의 경우 parent를 하나씩 끝까지 올리기 때문인데, Parent를 2^k번째 단위로 저장하여 Bit를 이용하면 알고리즘을 개선할 수 있다.

그 방법은 다음과 같다.

Algorithm:
1. DFS를 통해 각 정점의 Parent[i][k]와 Depth를 구한다. (P[i][k]는 정점i의 2^k번째 부모)
 1-1. DFS로 P[i][0]을 구하고 P[i][k] = P[P[i][k-1]][k-1] 로 구할 수 있다.
2. 두 정점(a,b)에 대해 Depth의 차이 |a-b|를 비트열로 x만큼 줄인다.
3. 만약 정점 a와 c가 같은 경우 a가 LCA가 된다.
3-1. a와 c가 다른 경우 정점a와 c를 동시에 올릴 때 가장 큰 비트열부터 보면서 올릴 수 있는 가장큰 k를 찾아 2^k만큼 올린다. 이를 같아 질때 까지 반복한다.

Time Complexity : O(logN)

Code:
```
int LCA(int a, int b)
{
	if (d[a] > d[b]) { int t = a; a = b; b = t; }
	
	int x = d[b] - d[a];
	for (int i = MAXP-1; i >= 0; i--)
		if (x & (1 << i)) b = p[b][i];

	if (a == b) return a;

	for (int i = MAXP-1; i>=0; i--)
	{
		if (p[a][i] != p[b][i])
		{
			a = p[a][i];
			b = p[b][i];
		}
	}
	return p[a][0];
}

void main()
{
	for (int i = 1; i <= N; i++) for (int j = 0; j < 20; j++) p[i][j] = -1;
	DFS(1, 0);

	for (int k = 1; k < MAXP; k++)
		for (int i = 1; i <= N; i++)
			if (p[i][k - 1] >= 0) p[i][k] = p[p[i][k - 1]][k - 1];
}

```

***
연습문제(기초)
1. https://www.acmicpc.net/problem/11438

연습문제(응용)
1. https://www.acmicpc.net/problem/1761
2. https://www.acmicpc.net/problem/3176


