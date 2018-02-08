---
layout: post
title:  "Algorithm Tip"
date:   2017-04-11 11:46:00
categories: ProblemSolving
tags: ProblemSolving
---

* content
{:toc}

## 부동 소수점의 비교 연산
부동 소수점 소수표현은 제한된 precision을 가지면서 정확한 값을 가지기 어렵습니다.

따라서 비교연산을 해야할 경우에는 1e-9~1e-12 와 같은 작은 수의 오차값으로 비교연산을 진행하면 좋습니다.

```
if( fabs(a-b) < 1e-10)) or
if(-1e-10 < a-b && a-b < 1e-10)
```

## 최대공약수(GCD), 최대공배수(LCD)

최대공약수는 유클리드 호제법을 이용하여 빠르게 구할 수 있습니다.
최대공배수는 최대공약수를 알면 쉽게 구할 수 있습니다.

```
int GCD(int a, int b)
{
	return (b==0) ? a : GCD(b,a%b);
}

int LCD(int a, int b)
{
	return a*b/GCD(a,b);
}
```

## 숫자 뒤집기
```
int reverse(int x)
{
	int m = 0;
	while (x > 0)
	{
        m = m * 10 + x % 10;
        x /= 10;
	}
	return m;
}

```

## 제곱근 구하기
sqrt() 함수를 실제로 구현하는 방법입니다.

Babylonian method를 이용하여 제곱근을 빠르게 구할 수 있습니다.

√S의 근삿값 Xn을 찾고 다음으로 Xn+1을 아래와 같은 점화식으로 구할 수 있다.
```
Xn+1 = 0.5*(Xn+S/Xn);
```

수식 유도는 [WIKI 참조](https://en.wikipedia.org/wiki/Methods_of_computing_square_roots)



```
double sqrt(double x)
{
	double px=1, nx;
	for(int i=0; i<50; i++)
	{
		nx=0.5*(px+x/px);
        if(-1e-4 < px-nx && px-nx < 1e-4) return nx;
        px=nx;
	}
}

```



## 비트마스크
비트 연산을 이용하여 효율적으로 계산할 수 있는 방법입니다.

1. AND 연산 :  (a & b)
2. OR 연산 : (a l b)
3. XOR 연산 : (a ^ b)
4. NOT 연산 : (~a)

a를 왼쪽으로 b비트 시프트 : a << b

비트마스크를 이용한 집합구현

int set; , p ( 0<= p <=29)

![](https://raw.githubusercontent.com/laboputer/laboputer.github.io/master/images/Problem_Solving/01.PNG)
