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
1. https://www.acmicpc.net/problem/7453
2. https://www.acmicpc.net/problem/2632

## 2. Shortest Path (최단경로)
그래프에서 주어진 두 정점을 연결하는 가장 짧은 경로의 길이.

다양한 그래프의 종류와 특성에 따라 최적화된 많은 최단경로 알고리즘이 존재한다. 최단경로는 최대 V-1개의 간선으로 이루어져 있다.(최악의 경우 모든 정점을 한번씩 방문하기 때문)

정점 간에 가중치가 없는 경우 최단 경로는 BFS로 O(N) 복잡도로 구할 수 있지만, 가중치가 있는 경우는 다르다. 이 경우에 대하여 다뤄보는 것이 이 챕터이다.

먼저, 최단경로에서 음수 가중치를 갖는 간선의 존재 여부가 중요하게 된다. 만약 그래프 상에서 음의 사이클이 존재하면 최단경로는 정의되지 않는다. (음의 무한대로 가기 때문에)

먼저 양의 가중치만을 가진 그래프 상에서 최단경로는 다익스트라 알고리즘으로 구할 수 있고, 음의 가중치를 가진 경우에는 플로이드 알고리즘을 사용할 수 있다. 문제의 조건에 따라 적용 가능한 알고리즘들이 있고 다시 주어지는 정점과 간선 조건에 따라 효율적인 알고리즘이 달라질 수 있다.

### Dijkstra Algorithm (다익스트라)

한 정점에서 다른 정점(1:N)의 최단경로를 구하는 알고리즘

다익스트라 알고리즘은 BFS처럼 시작점에서 부터 시작하여 dist[N]을 갱신시키는 과정에서 마지막 dist[N]에 최종적으로 1:N의 최단경로가 구해진다.

Algorithm:
1. 시작점S 에서부터 다른 모든 정점간의 거리를 계산한 dist[N]을 정의한다.
2. dist[N]에서 확정되지 않은 정점들 중 가장 짧은 값을 가진 정점 i는 S~i의 최단경로로 확정한다.
3. 2.에서 정해진 정점i로 부터 연결된 모든 정점 간의 거리를 계산한 dist[N]을 갱신한다.
4. (2-3)N 번 반복하면 된다.

위의 알고리즘에서 2번에서 가장 짧은 값을 정점을 선택할 때 힙구조를 이용하면 O(ElogE) 또는 O(ElogV) 복잡도를 가지며, 만약 V가 작거나 E가 매우 큰 경우에는 힙 구조를 사용하지 않는 것이 오히려 더 빠를 수도 있습니다.

힙 구조 사용하지 않는 경우
Time Complexity : O(V^2)

힙 구조 사용하는 경우 (힙 대신 Set을 사용하면 ElogV)
Time Complexity : O(ElogE + ElogV)

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

다익스트라는 음의 가중치가 있을 경우 최단경로를 구할 수 없다. 하지만 벨만포드 알고리즘은 음의 가중치를 포함해도 최단경로를 구할 수 있다. 만약 그래프에 음의 사이클이 있을 경우에는 최단경로 자체가 정의되지 않는다. 벨만-포드는 이 음의 사이클 존재여부도 구할 수 있다.

벨만-포드는 N개의 최단거리 상한선을 가지고 유지하면서 어떤 간선이 있으면 상한선을 줄일 수 있는가? 로 접근한다. M개의 간선을 돌면서 상한선을 갱신하는 과정(Relax)을 반복한다.

이 Relax과정은 단 한 가지 최단경로 특성을 가지고 접근합니다.

 * dist[u] <= dist[v] + w(u,v) (여기서 dist[u], dist[v]는 시작점 s로부터 최단 거리)

위의 특성은 귀류법으로 간단히 증명할 수 있다. dist[u]가 더 큰 경우 dist[v] + w(u,v)인 거리가 존재하므로 dist[u]가 최단거리임이 모순이기 때문이다. 따라서 이 Relax과정을 반복하면 최단거리를 구할 수 있음이 자명하다.

이 Relax 과정은 최대 V-1번 반복된다. 왜냐하면 모든 간선을 통하여 갱신할 경우 최소한 1개의 정점은 최단거리가 구해지기 때문이다. (여기서, Relax과정이 V번 이상 반복되면 음의 사이클이 존재함을 의미한다.)

또한, 각 정점을 완화시킨 간선을 저장하여 모으면 최소 스패닝트리가 되는데, 이 트리를 따라 한 정점에서 시작정점(루트)까지의 경로가 시작점부터 해당 정점까지의 최단거리 경로가 된다.

주의할 점은 시작점으로부터 도달할 수 없는 정점인 경우 임의의 큰값 dist[v]==2e9 임으로는 알 수 없다. 음의 간선으로 그 값이 조금 줄어들 수 있기 때문에 dist[v] >= 2e9-M 로 적당히 큰 값 M을 빼준 것 보다는 클 경우로 생각해야 한다. 물론 2e9-M 값 자체가 나올 수 없는 큰 값이어야 함.

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

플로이드 워셜 알고리즘은 모든 정점간 최단 경로를 빠르게 구할 수 있는 알고리즘이다.
음의 가중치가 있을 경우에도 동작하며, 그 핵심 아이디어는 경유점이다.

어떤 정점 간 최단 경로는 경유점 v을 지나거나, 지나지 않거나 인데 이를 수식으로 표현하면
```
D[v][i][j] = 1~v를 경유할 수 있을 때 i~j 최단경로 라고 할 때,

D[v][i][j] = MIN(D[v-1][a][v] + D[v-1][v][b], D[v-1][i][j]);
```
위에서 v==0 일때는 D[v][i][j] = a[i][j]이다.

관계식을 보면 알겠지만 [v]에 있는 메모리는 없어도 구할 수 있다. 그리고 경유지 v가 가장 바깥 for문에 있어야 한다.

플로이드 워셜 알고리즘은 보통 N번 다익스트라 또는 벨만포드보다 빠르다. 내부의 복잡한 수식이 없기 때문에 시간복잡도는 같지만 상수배가 작기 때문이다. 플로이드 워셜 알고리즘 또한 Dist[a][b] 를 갱신한 경유점을 저장해두면 쉽게 최단거리의 경로를 구해낼 수 있다.

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
2. https://www.acmicpc.net/problem/1956

### SPFA (Shortest Path Faster Algorithm)
최단경로를 조금 더 빠르게 구할 수 있는 알고리즘.

이 알고리즘은 벨만포드 알고리즘에서 조금 더 개선한 알고리즘이라고 보면 된다. 시간복잡도도 벨만포드 알고리즘과 동일한 O(VE)이지만, 평균적인 경우 O(E)이다.

원리는 간단한데, 벨만포드 알고리즘 if(d[to] > d[from] + cost) 에서 정점 to로 가는 경로가 갱신되는 경우에만 다른 정점으로의 경로도 갱신될 가능성이 있기 때문에,
갱신될 가능성이 있을 때만 모든 간선을 확인해보는 것으로 바꾸는 것이다.

```
	while (!q.empty())
	{
		int from = q.front();
		c[from] = false; q.pop();
		for (int i = 0; i < E; i++)
		{
			int to = e[i].to;
			int cost = e[i].cost;
			if (d[to] > d[from] + cost)
			{
				d[to] = d[from] + cost;
				if (c[to] == false)
				{
					q.push(to);
					c[to] = true;
				}
			}
		}
	}
```


## 3. Minimum Spanning Tree(최소 스패닝트리)
그래프가 주어졌을 때 스패닝트리 중 간선의 합이 가장 작은 최소 스패닝트리를 구하는 문제

그래프가 주어지면 N-1개 간선으로 만들어진 최소 스패닝 트리를 구하는 알고리즘은 크게 2가지가 있다. 크루스칼 알고리즘과 프림 알고리즘이다.

### Kruskal Algorithm
Algorithm:
1. 처음 공집합인 트리에서 시작한다.
2. 그래프의 모든 간선을 가중치 오름차순으로 정렬한다.
3. 스패닝트리에 가장 작은 간선 하나를 추가한다.
4. 만약 사이클이 생기면 해당 간선을 지우고 다시 3번을 반복한다.
5. 추가된 간선이 N-1개면 종료한다. (최소 스패닝트리)

사이클 판정은 Union-Find 자료구조를 이용하면 된다.

Time Complexity: O(ElogE)

Code:
```
// MST의 모든 가중치의 합
for (int i = 0; i < M; i++)
	{
		if (uf.Find(E[i].s) != uf.Find(E[i].e))
		{
			uf.Union(E[i].s, E[i].e);
			ans += E[i].r;
			cnt++;
		}
		if (cnt == N - 1) break;
	}
```


***
연습문제(기초)

1. https://www.acmicpc.net/problem/1197

연습문제(응용)


### Prim Algorithm
Algorithm:
1. 그래프에서 하나의 꼭지점을 선택하여 트리를 구성한다.
2. 트리를 기준으로 방문하지 않은 정점에 대한 간선들의 집합 중 가장 작은 간선을 선택한다.
3. 모든 정점을 방문할 때 까지 2번을 반복한다.

가장 작은 간선을 선택할 때 우선순위 큐를 이용한다.

TIme Complexity : O(ElogE)


## 4. Lowest Common Ancestor (LCA)
트리가 주어졌을때 임의의 두 점의 가장 가까운 공통조상을 찾는 문제

트리에서 임의의 두 점은 공통조상을 가진다. 루트도 공통조상이겠지만 루트에서 내려가면서 두 점으로 가기위해 갈라지는 바로 그 지점이다. 공통조상 중에서 가장 가까운 조상을 찾고자 한다.

만약 루트가 변경되면 공통조상은 달라지지만, 정점 경로(정점간 거리)는 동일하다.

이 문제의 아이디어는 루트로부터 높이차를 구한 후 루트까지 올라가면서 같을 때까지 보는 것이다.

Algorithm:
1. DFS를 통해 각 정점의 Parent와 루트로부터 Depth를 구한다.
2. 두 정점(a,b)에 대해 LCA는 Depth가 큰 정점b을 작은 정점a과 같을 떄까지 올린 정점c를 구한다.
3. 만약 정점 a와 c가 같은 경우 a가 LCA가 된다. 
 - a와 c가 다른 경우 정점a와 c를 동시에 올려가면서 같아 질때 까지 올라간다.

위의 알고리즘은 한 쌍의 LCA를 구하는데 O(N) 시간이 걸린다. 최악의 경우 parent를 하나씩 끝까지 올리기 때문이다.

이를 개선한 O(logN) 방법이 존재하는데 아이디어가 중요한 문제다. 이를 소개하면 Parent를 2^k번째 단위로 저장하자는 것이다. Parent를 2^k번째 단위로 저장하는 이유는 간격을 줄일때 depth 차이를 Bit로 생각하여 2^k번째 비트가 있으면 Parent를 2^k만큼 점프할 수 있기 때문이다. 심지어 2^k = 2^(k-1)*2^(k-1) 이므로 2^k번째 부모를 쉽게 계산해낼 수 있다.

위의 아이디어는 지수에 대한 값을 구할 때 비트단위로 구할 수 있다는 점을 착안하면 떠올릴 수 있다.

Algorithm:
1. DFS를 통해 각 정점의 Parent[i][k]와 Depth를 구한다.(P[i][k]는 정점i의 2^k번째 부모)
2. DFS로 P[i][0]을 구하고 P[i][k]=P[P[i][k-1]][k-1]로 구할 수 있다.
3. 두 정점(a,b)에 대해 Depth의 차이 abs(a-b)를 비트열로 x만큼 줄인다.
4. 만약 정점 a와 c가 같은 경우 a가 LCA가 된다.
5. a와 c가 다른 경우 정점a와 c를 동시에 올릴 때 가장 큰 비트열부터 보면서 올릴 수 있는 가장큰 k를 찾아 2^k만큼 올린다. 이를 같아 질때 까지 반복한다.

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

O(logN)으로 구하면서도 좀 더 효율적일 것 같고 간결하게 구하는 아이디어도 있다.

루트를 기준으로 DFS를 돌면서 특정 정점을 방문을 시작하는 시간과 완전히 빠져나가는 시간을 구해놓으면
정점 a가 정점 b의 부모인지 O(1)만에 체크가 가능한데 이를 이용하여 구할 수 있다.

Code:
```
// Is a the parent of b ?
bool upper(int a, int b)
{
	if (in[a] <= in[b] && out[b] <= out[a]) return 1;
	return 0;
}

int LCA(int a, int b)
{
	if (upper(a, b)) return a;
	if (upper(b, a)) return b;

	for (int i = MAXP-1; i >= 0; i--)
		if (!upper(p[a][i], b))
			a = p[a][i];

	return p[a][0];
}
```


***
연습문제(기초)
1. https://www.acmicpc.net/problem/11438

연습문제(응용)
1. https://www.acmicpc.net/problem/1761
2. https://www.acmicpc.net/problem/3176

## 5. Segment Tree (세그먼트 트리)
구간에 대한 정보를 빠르게 처리할 수 있는 트리 구조

세그먼트 트리는 구간에 대한 정보를 이진트리로 표현하는 전처리를 통하여 특정 구간에 대한 쿼리를 빠르게 처리하는 용도로 사용할 수 있다.

세그먼트 트리를 구성하면 어떤 구간[a,b]이 주어져도 O(logN)만에 이 구간에 들어있는 구간들의 합집합으로 표현 가능한데, 이를 이용하여 어떤 구간의 합, 최소값 등 특정 구간의 정보들을 빠르게 구할 수 있게 된다.

아이디어는 간단하다. 특정 구간의 정보를 트리 형식으로 표현하여, 어떤 구간에 대한 쿼리가 들어왔을 때 빠르게 특정 구간들의 정보를 조합하여 답을 도출해내는 것이다.

ALgorithm:
0. N개의 데이터가 있으면 4*N개의 배열을 선언한다. (메모리를 4*N 잡으면 충분)
1. Initialize는 arr[N]의 정보를 가진채로 트리 배열을 구한다.
2. [1,N]은 tree[1]에 저장하며, 각 자식노드는 반반에 대한 구간 정보를 저장한다.
3. Update연산은 자식노드로 내려가면서 포함하는 인덱스가 있을 경우 데이터를 갱신한다.
4. Query연산은 [a,b]쿼리를 보낼 때 이를 포함하는 구간이 나올때까지 쪼개가며 return한다

Time Complexity :
1. Update O(logN)
2. Query O(logN)
3. Initialize O(N)

Code:
```
long long init(int n, int l, int r)
{
	if (l == r) // 구간의 요소가 1개일때 값
		return seg[n] = arr[l];
	else // 구간의 요소가 n개 일때 (합 정보를 저장할 때 예제)
    	return seg[n] = init(n * 2, l, (l + r) / 2) + init(n * 2 + 1, (l + r) / 2 + 1, r);
}

void update(int n, int s, int e, int idx, int x)
{
	if (idx < s || e < idx) return;
    
    // idx를 포함하는 구간
    // seg[n] = 데이터처리
    
	if (s != e) // 요소가 하나 일때는 쪼갤 수 없음.
	{
		update(n * 2, s, (s + e) / 2, idx, x);
		update(n * 2 + 1, ((s + e) / 2) + 1, e, idx, x);
	}
}

val query(int n, int s, int e, int l, int r)
{
	if (e < l || r < s) return { INF , 0 }; // Invalid Data
	if (l <= s && e <= r) return seg[n];  // 구간[s,e]에 대한 내용을 전부 사용함

    // query 처리
	// query(n * 2, s, (s + e) / 2, l, r);
	// query(n * 2 + 1, ((s + e) / 2) + 1, e, l, r);
	// return ;
}

```

***
연습문제(기초)
1. https://www.acmicpc.net/problem/2357
2. https://www.acmicpc.net/problem/2042

연습문제(응용)

### Lazy propagation
Segment Tree에서 Update 연산을 구현할 때 바로 Update하지 않고 필요시 Update하는 방법.

예를 들어보자면, Update 연산을 할 때 특정 요소의 데이터를 변경하는 것이 아니라 어떤 구간에 대한 데이터를 변경한다고 생각해보자. 어떤 구간에 대한 데이터를 변경할때도 Query로 해당 구간을 포함하는 Segment Tree를 변경하면 된다. 하지만 최악의 경우([1,N] 구간 변경) 세그먼트 트리의 모든 노드를 변경해야 한다. (O(NlogN))

Update 후 바로 Query를 진행하는 경우에는 Lazy하게 해도 의미가 없지만 위와 같은 Update가 M번 반복되면 느려지므로, 세그먼트 트리를 갱신해 나가다가 특정 노드의 자식노드들을 모두 갱신해야 되는 상황이 오면 갱신하지 않고 lazy 계수를 저장해놓고 필요할 때만 업데이트하여 사용하는 방법이다.

일반적인 세그먼트 트리와 구현상 다른점은 Lazy 계수를 두고 특정 노드를 방문할 때 마다 Lazy계수가 존재하는 지 보고 있으면 세그먼트 트리에 반영하고 Lazy계수를 자식노드 2노드에게 전파한다.

Code
```
// 합에 대한 쿼리를 예시로 작성하였음.

void update_lazy(int n, int l, int r)
{
	if (lazy[n] != 0)
	{
		seg[n] += (r - l + 1)*lazy[n];
		if (l != r)
		{
			lazy[n * 2] += lazy[n];
			lazy[n * 2 + 1] += lazy[n];
		}
		lazy[n] = 0;
	}
}

void update_range(int n, int l, int r, int ll, int rr, long long diff)
{
	update_lazy(n, l, r);
	if (rr < l || r < ll) return;

	if (ll <= l && r <= rr)
	{
		lazy[n] += diff;
		update_lazy(n, l, r);
		return;
	}
	
	if (l != r)
	{
		update_range(n * 2, l, (l + r) / 2, ll, rr, diff);
		update_range(n * 2 + 1, (l + r) / 2 + 1, r, ll, rr, diff);
		seg[n] = seg[n * 2] + seg[n * 2 + 1];
	}
}

long long query_sum(int n, int l, int r, int ll, int rr)
{
	update_lazy(n, l, r);
	if (rr < l || r < ll) return 0;

	if (ll <= l && r <= rr) return seg[n];
	return query_sum(n * 2, l, (l + r) / 2, ll, rr) + query_sum(n * 2 + 1, (l + r) / 2 + 1, r, ll, rr);
}
```

***
연습문제(기초)
1. https://www.acmicpc.net/problem/10999

연습문제(응용)
1. https://www.acmicpc.net/problem/1395
2. https://www.acmicpc.net/problem/7626
3. https://www.acmicpc.net/problem/5486