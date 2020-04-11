---
layout: post
title:  "판다스(Pandas) 사용법 익히기 - 기본편"
subtitle:   "pandas"
categories: machine-learning
tags: python tutorial pandas
---
> 판다스 공식홈페이지 [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)에서 소개된 기본적인 사용법을 따라하면서 번역한 글입니다. 직역하기 보다는 간단하게 요약하고 설명이 더 필요한 부분은 추가하였으니 도움되시길 바랍니다.

> 이 글을 읽으신 후에 더 자세한 내용을 확인하시고 싶으신 분들은 [Cookbook](http://pandas.pydata.org/pandas-docs/stable/cookbook.html#cookbook)를 읽으시길 바랍니다.

포스팅 하나로 정리하였으나 내용이 많아 목차로 필요한 챕터만 링크가셔서 보시길 바랍니다.

## 목차
---
1. [오브젝트 생성 - Object Creation](#item1)
2. [데이터 확인하기 - Viewing Data](#item2)
3. [데이터 선택하기 - Selection](#item3)
4. [결측 데이터 - Missing Data](#item4)
5. [데이터 연산 - Operations](#item5)
6. [데이터 합치기 - Merge](#item6)
7. [그룹화 - Grouping](#item7)
8. [데이터 구조 변경하기 - Reshaping](#item8)
9. [시계열 데이터 - Time Series](#item9)
10. [범주형 데이터 - Categoricals](#item10)
11. [그래프 시각화 - Plotting](#item11)
12. [파일 입출력 - Getting Data In/Out](#item12)

## 시작하기 전에
---

[판다스(Pandas)]()를 쉽게 설명드리면 DB 테이블처럼 테이블 형식의 데이터를 쉽게 사용하기 위해 만들어진 패키지입니다.

이 글에서는 아래와 같이 3가지 패키지가 활용됩니다.
> Python에서 아래 코드가 실행되지 않는 경우 관련 패키지를 pip롤 통해 설치하시기 바랍니다.

```python
import pandas as pd
import numpy as np

# 시각화 패키지
import matplotlib.pyplot as plt
%matplotlib inline
```

<a name="item1"></a>

## 1. 오브젝트 생성 - Object Creation
---
판다스에서 사용하는 오브젝트(Object)가 무엇인지 설명부터 하고 생성하는 방법에 대해 소개합니다.

> 이 장에 대해 자세한 설명은 [Data Structure Intro section](https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html#cookbook)를 참고하세요.


판다스에서의 오브젝트(Object)가 무엇일까요?

![](https://laboputer.github.io/assets/img/ml/python/pandas/1.JPG)

위 그림처럼 판다스에서는 2가지 오브젝트 `Series` 와 `DataFrame`가 있습니다. 이 2가지만 기억하면 됩니다. 

- `Series`: 1차원 데이터와 각 데이터의 위치정보를 담는 인덱스로 구성
- `DataFrame`: 2차원 데이터와 인덱스, 컬럼으로 구성
 
쉽게 생각하면 `DataFrame`에서 하나의 컬럼만 가지고 있는 것이 `Series`입니다.

`Series`는 어떻게 생성할 수 있을까요?

```python
s = pd.Series([1,3,5,np.nan,6,8])
# 0    1.0
# 1    3.0
# 2    5.0
# 3    NaN
# 4    6.0
# 5    8.0
# dtype: float64
```
위와 같이 `Series()` 안에 list로 1차원 데이터만 넘기면 됩니다. index는 입력하지 않아도 자동으로 0부터 입력됩니다. 만약 다른 index를 입력하고 싶으시면 똑같이 list 형식으로 입력하면 됩니다.

`DataFrame`은 어떻게 생성할까요?

```python
dates = pd.date_range('20130101', periods=6)
# DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
#                '2013-01-05', '2013-01-06'],
#               dtype='datetime64[ns]', freq='D')

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
#                       A           B           C           D
# 2013-01-01	 1.571507	 0.160021	-0.015071	-0.118588
# 2013-01-02	-1.037697	-0.891196	 0.495447	 0.453095
# 2013-01-03	-1.682384	-0.026006	-0.152957	-0.212614
# 2013-01-04	-0.108757	-0.958267	 0.407331	 0.187037
# 2013-01-05	 1.092380	 2.841777	-0.125714	-0.760722
# 2013-01-06	 1.638509	-0.601126	-1.043931	-1.330950
```

먼저 `index`가 시간에 관련된 데이터라고 가정하고 `data_range()`를 이용하여 시간에 대한 1차원 데이터를 생성하였습니다. 그리고 `np.random.randn()`을 통해 6x4에 해당하는 2차원 데이터를 생성하였습니다. `columns` 또한 `DataFrame`에서 사용될 컬럼의 이름을 1차원 데이터(A,B,C,D)로 생성합니다. 위 3가지 데이터 (컬럼, 인덱스, 2차원 데이터)를 가지고 만들었습니다.

자주 사용하는 딕셔너리 형식으로도 `DataFrame`을 만들 수 있습니다.

```python
df2 = pd.DataFrame({'A':1.,
                   'B':pd.Timestamp('20130102'),
                   'C':pd.Series(1,index=list(range(4)),dtype='float32'),
                   'D':np.array([3]*4,dtype='int32'),
                   'E':pd.Categorical(["test","train","test","train"]),
                   'F':'foo'})

#     A          B    C D       E     F
# 0	1.0	2013-01-02	1.0	3	test	foo
# 1	1.0	2013-01-02	1.0	3	train	foo
# 2	1.0	2013-01-02	1.0	3	test	foo
# 3	1.0	2013-01-02	1.0	3	train	foo
```

각 Key값과 Value(1차원 데이터)가 `DataFrame`의 하나의 컬럼과 2차원 데이터가 됩니다.
당연하겠지만 모든 딕셔너리의 Value의 리스트 길이가 같아야 생성할 수 있습니다.

`DataFrame`의 `.dtypes`라는 값에는 각 컬럼이 어떤 데이터 형식인지가 저장되어 있습니다. 만약 섞여있을 경우 `object`가 됩니다.

```python
df2.dtypes
# A           float64
# B    datetime64[ns]
# C           float32
# D             int32
# E          category
# F            object
# dtype: object
```

`.dtypes`외의 다른 속성값을 보고 싶은 경우,
- Jupyter Notebook: df. 입력후 TAB 입력
- Visual Studio Code: df. 입력후 CTRL + SPACE 입력

```python
df2.<TAB>
# df2.A                  df2.bool
# df2.abs                df2.boxplot
# df2.add                df2.C
# df2.add_prefix         df2.clip
# df2.add_suffix         df2.clip_lower
# df2.align              df2.clip_upper
# df2.all                df2.columns
# df2.any                df2.combine
# df2.append             df2.combine_first
# df2.apply              df2.consolidate
# df2.applymap
# df2.D
```
각 컬럼의 이름이 속성값으로 `Series`가 들어가 있는 것을 확인할 수 있습니다.

```python
# df2의 'A' 컬럼 값을 보여줍니다.
df2.A
# 0    1.0
# 1    1.0
# 2    1.0
# 3    1.0
# Name: A, dtype: float64
```

<a name="item2"></a>

## 2. 데이터 확인하기 - Viewing Data
---
> 이 장에 대해 자세한 설명은 [Basics section](https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#basics)을 확인하세요.

`DataFrame`은 `head()`, `tail()`의 함수로 처음과 끝의 일부 데이터만 살짝 볼 수 있습니다. 데이터가 큰 경우에 데이터가 어떤식으로 구성되어 있는지 확인할 때 자주 사용합니다.

```python
# 첫번째 행부터 5개(기본값)를 보여줍니다.
df.head()
#                       A	        B	        C	        D
# 2013-01-01	 1.571507	 0.160021	-0.015071	-0.118588
# 2013-01-02	-1.037697	-0.891196	 0.495447	 0.453095
# 2013-01-03	-1.682384	-0.026006	-0.152957	-0.212614
# 2013-01-04	-0.108757	-0.958267	 0.407331	 0.187037
# 2013-01-05	 1.092380	 2.841777	-0.125714	-0.760722

# 마지막 행에서 3개를 보여줍니다.
df.tail(3)
#                       A	        B	        C	        D
# 2013-01-04	-0.108757	-0.958267	 0.407331	 0.187037
# 2013-01-05	 1.092380	 2.841777	-0.125714	-0.760722
# 2013-01-06	 1.638509	-0.601126	-1.043931	-1.330950
```

`DataFrame`의 대표적인 값인 `.columns`, `.index`, `.values`는 다음과 같이 각각 확인할 수 있습니다.

```python
df.index
# DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
#                '2013-01-05', '2013-01-06'],
#               dtype='datetime64[ns]', freq='D')

df.columns
# Index([u'A', u'B', u'C', u'D'], dtype='object')

df.values
# array([[ 1.57150668,  0.16002149, -0.01507058, -0.11858772],
#        [-1.03769719, -0.89119559,  0.49544709,  0.45309462],
#        [-1.68238391, -0.02600566, -0.15295699, -0.21261369],
#        [-0.10875693, -0.9582675 ,  0.40733062,  0.18703669],
#        [ 1.09238026,  2.84177746, -0.12571427, -0.76072186],
#        [ 1.63850927, -0.60112596, -1.04393119, -1.33095031]])
```

`DataFrame`의 `to_numpy()`를 이용하여 인덱스와 컬럼을 제외한 2차원 데이터만을 `numpy`의 형식으로 반환해줍니다. 사실은 `.values`와 동일합니다.

```python
df.to_numpy()
# array([[ 0.34867481, -1.35115579, -0.17998376, -0.61736245],
#        [ 0.57944517,  0.47589539, -0.02466693,  1.25133379],
#        [-1.50771281,  0.12265966, -1.56775663, -0.3830442 ],
#        [-0.90118096,  0.36020193,  1.15019657, -0.29792194],
#        [-0.97330279,  1.36258901, -0.77611896,  0.64432274],
#        [ 0.65122666,  0.75397091, -1.69294079,  1.95252915]])
```

`DataFrame`의 `desribe()`를 통해 각 컬럼의 통계적인 수치를 요약하여 보여줄 수 있습니다.
- count: 데이터 개수
- mean: 평균값
- std: 표준편차
- min: 최소값
- 25%: 1사분위값
- 50%: 중앙값
- 75%: 3사분위값
- max: 최대값

```python
df.descrbie()
#               A	        B	        C	        D
# count	 6.000000	 6.000000	 6.000000	 6.000000
# mean	 0.245593	 0.087534	-0.072483	-0.297124
# std	 1.407466	 1.423367	 0.549378	 0.651149
# min	-1.682384	-0.958267	-1.043931	-1.330950
# 25%	-0.805462	-0.818678	-0.146146	-0.623695
# 50%	 0.491812	-0.313566	-0.070392	-0.165601
# 75%	 1.451725	 0.113515	 0.301730	 0.110631
# max	 1.638509	 2.841777	 0.495447	 0.453095
```

`DataFrame`의 `.T` 속성은 `values`를 Transpose한 결과를 보여줍니다. Transpose는 인덱스를 컬럼으로, 컬럼을 인덱스로 변경하여 보여주는 것입니다.

```python
df.T
#    2013-01-01 2013-01-02   2013-01-03  2013-01-04  2013-01-05  2013-01-06
# A	 1.571507	-1.037697	-1.682384	-0.108757	 1.092380	 1.638509
# B	 0.160021	-0.891196	-0.026006	-0.958267	 2.841777	-0.601126
# C	-0.015071	 0.495447	-0.152957	 0.407331	-0.125714	-1.043931
# D	-0.118588	 0.453095	-0.212614	 0.187037	-0.760722	-1.330950
```

`DataFrame`의 `sort_index()`를 통해 인덱스 또는 컬럼의 이름으로 정렬을 할 수도 있습니다.
- `axis`: 축 기준 정보 (0: 인덱스 기준, 1: 컬럼 기준)
- `ascending`: 정렬 방식 (false : 내림차순, true: 오름차순)

```python
df.sort_index(axis=1, ascending=False)
# 	                    D	        C	        B	        A
# 2013-01-01	-0.118588	-0.015071	 0.160021	 1.571507
# 2013-01-02	 0.453095	 0.495447	-0.891196	-1.037697
# 2013-01-03	-0.212614	-0.152957	-0.026006	-1.682384
# 2013-01-04	 0.187037	 0.407331	-0.958267	-0.108757
# 2013-01-05	-0.760722	-0.125714	 2.841777	 1.092380
# 2013-01-06	-1.330950	-1.043931	-0.601126	 1.638509
```

`DataFrame`의 `sort_values()` 를 이용하여 value 값 기준으로 정렬할 수도 있습니다.
- `by`: 데이터 정렬에 기준이 되는 컬럼

```python
# 'B' 컬럼 기준으로 정렬됩니다.
df.sort_values(by='B')
#                       A	        B	        C	        D
# 2013-01-04	-0.108757	-0.958267	 0.407331	 0.187037
# 2013-01-02	-1.037697	-0.891196	 0.495447	 0.453095
# 2013-01-06	 1.638509	-0.601126	-1.043931	-1.330950
# 2013-01-03	-1.682384	-0.026006	-0.152957	-0.212614
# 2013-01-01	 1.571507	 0.160021	-0.015071	-0.118588
# 2013-01-05	 1.092380	 2.841777	-0.125714	-0.760722
```

<a name="item3"></a>

## 3. 데이터 선택하기 - Selection 
---
> 이 장에 대해 자세한 설명은 [Indexing and Selecting Data](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing)와 [MultiIndex / Advanced Indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced)을 확인하세요.

### 데이터 가져오기 - Getting 

컬럼을 기준으로 데이터를 가져올 수 있습니다.

```python
# df.A 또는 df['A'] 로 컬럼의 데이터를 얻을 수 있습니다.
df['A']
# 2013-01-01    1.571507
# 2013-01-02   -1.037697
# 2013-01-03   -1.682384
# 2013-01-04   -0.108757
# 2013-01-05    1.092380
# 2013-01-06    1.638509
# Freq: D, Name: A, dtype: float64
```

`[]`을 이용하여 특정 범위의 행을 슬라이싱할 수 있습니다.

```python
# 0번~2번 행을 슬라이싱 합니다.
df[0:3]
# 	                    A	        B           C	        D
# 2013-01-01	 1.571507	 0.160021	-0.015071	-0.118588
# 2013-01-02	-1.037697	-0.891196	 0.495447	 0.453095
# 2013-01-03	-1.682384	-0.026006	-0.152957	-0.212614

# 20130102 부터 20130104 까지 행을 슬라이싱 합니다.
df['20130102':'20130104']
#                       A	        B	        C	        D
# 2013-01-02	-1.037697	-0.891196	 0.495447	 0.453095
# 2013-01-03	-1.682384	-0.026006	-0.152957	-0.212614
# 2013-01-04	-0.108757	-0.958267	 0.407331	 0.187037
```

### 이름으로 데이터 가져오기 - Selection by label
> 이 장에 대해 자세한 설명은 [Selection by Label](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-label)을 참고하세요.

이름(Label)로 가져오는 것은 `DataFrame`의 `.loc` 속성을 이용합니다.

`.loc`은 2차원으로 구성되어 있습니다. 1번째 인덱스는 인덱스명을 2번째 인덱스는 컬럼명으로 접근가능 합니다. 만약 인덱스명만 입력하면 행의 값으로 결과가 나옵니다.
또한 인덱스명, 컬럼명을 선택할때 리스트 형식으로 멀티인덱싱이 가능합니다.

```python
# 0번 인덱스명으로 데이터 가져오기
df.loc[dates[0]]
# A   -0.485592
# B   -0.658570
# C    0.529168
# D   -0.123461
# Name: 2013-01-01 00:00:00, dtype: float64
```

```python
# 행은 전체 선택, 컬럼명은 'A', 'B' 두개 선택하여 가져오기
df.loc[:,['A','B']]
#                   	A	        B
# 2013-01-01	-0.485592	-0.658570
# 2013-01-02	-0.259961	 1.221440
# 2013-01-03	-2.292462	-0.869762
# 2013-01-04	 0.468613	-0.739848
# 2013-01-05	-0.758953	-1.444800
# 2013-01-06	-0.577741	-0.926498
```

```python
# 행은 슬라이싱으로 범위 선택, 컬럼명은 'A','B' 선택
df.loc['20130102':'20130104',['A','B']]
# 	                    A	        B
# 2013-01-02	-0.259961	 1.221440
# 2013-01-03	-2.292462	-0.869762
# 2013-01-04	 0.468613	-0.739848
```

```python
# 행은 20130102 선택, 컬럼명은 'A', 'B' 선택
df.loc['20130102',['A','B']]
# A   -0.259961
# B    1.221440
# Name: 2013-01-02 00:00:00, dtype: float64
```

인덱스명, 컬럼명을 하나씩 선택하면 스칼라값을 가져올 수 있습니다.

```python
# 행은 첫번째 선택, 컬럼은 'A' 선택
df.loc[dates[0],'A']
# -0.48559222322550516
```

### 인덱스로 데이터 가져오기 - Selection by Position
여기서 말하는 인덱스는 위치(숫자) 정보를 말합니다.
> 이 장에 대해 자세한 설명은 [Selection by Position](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-integer)을 확인하세요.

`DataFrame`의 `.iloc` 속성을 이용합니다.

`.iloc`도 2차원 형태로 구성되어 있어 1번째 인덱스는 행의 번호를, 2번째 인덱스는 컬럼의 번호를 의미합니다. 마찬가지로 멀티인덱싱도 가능합니다.

```python
# 3번 인덱스 행 가져오기
df.iloc[3]
# A    0.468613
# B   -0.739848
# C    0.786264
# D    1.431795
# Name: 2013-01-04 00:00:00, dtype: float64
```

```python
# 3~4번 인덱스 행, 0~1번 컬럼 값 가져오기
df.iloc[3:5,0:2]
# 	                    A	        B
# 2013-01-04	 0.468613	-0.739848
# 2013-01-05	-0.758953	-1.444800
```

```python
# 1,2,4번 인덱스 행과 0,2번 인덱스 컬럼 가져오기
df.iloc[[1,2,4],[0,2]]
#                     A	        C
# 2013-01-02	-0.259961	-0.262255
# 2013-01-03	-2.292462	-0.551750
# 2013-01-05	-0.758953	 0.253333
```

```python
# 1~2번 인덱스 행과 전체 컬럼 값 가져오기
df.iloc[1:3,:]
#                       A	        B	        C	        D
# 2013-01-02	-0.259961	 1.221440	-0.262255	-0.973870
# 2013-01-03	-2.292462	-0.869762	-0.551750	 0.489118
```

```python
# 전체 행과 1~2번 인덱스 컬럼 값 가져오기
df.iloc[:,1:3]
# 	                    B	        C
# 2013-01-01	-0.658570	 0.529168
# 2013-01-02	 1.221440	-0.262255
# 2013-01-03	-0.869762	-0.551750
# 2013-01-04	-0.739848	 0.786264
# 2013-01-05	-1.444800	 0.253333
# 2013-01-06	-0.926498	 0.005663
```

```python
# 1번 행, 1번 컬럼 값 가져오기
df.iloc[1,1]
# 1.2214396062156994

# 위와 동일하지만 스칼라값을 가져오는 속도가 .iat이 빠르다고 합니다.
df.iat[1,1]
# 1.2214396062156994
```

### 조건으로 가져오기 - Boolean Indexing

하나의 컬럼의 값에 따라 행들을 선택할 수 있습니다.

```python
df[df['A'] > 0]
# 	                    A	        B	        C	        D
# 2013-01-01	0.348675	-1.351156	-0.179984	-0.617362
# 2013-01-02	0.579445	 0.475895	-0.024667	 1.251334
# 2013-01-06	0.651227	 0.753971	-1.692941	 1.952529
```

`DataFrame`의 값 조건에 해당하는 것만 선택할 수도 있습니다.

```python
df[df > 0]
#                       A           B	       C	       D
# 2013-01-01	0.348675	NaN	        NaN	        NaN
# 2013-01-02	0.579445	0.475895	NaN	        1.251334
# 2013-01-03	NaN 	    0.122660	NaN	        NaN
# 2013-01-04	NaN	        0.360202	1.150197	NaN
# 2013-01-05	NaN	        1.362589	NaN	        0.644323
# 2013-01-06	0.651227	0.753971	NaN	        1.952529
```

`isin()`을 이용하여 필터링을 할 수 있습니다.

```python
# df를 복사합니다.
df2 = df.copy()
# 새로운 컬럼 E에 값을 넣습니다.
df2['E'] = ['one','one', 'two','three','four','three']
#                       A	        B	        C	       D	   E
# 2013-01-01	-1.160801	 1.350440	-1.126817	 1.624509	one
# 2013-01-02	 0.528919	 2.245356	-0.256795	-0.803721	one
# 2013-01-03	-0.253946	-0.951220	 0.223419	-0.781845	two
# 2013-01-04	-0.093062	 0.100206	 0.890981	 1.078229	three
# 2013-01-05	-0.098203	-2.113593	-0.457045	 0.789558	four
# 2013-01-06	-0.585436	-0.533013	 0.558793	-1.323578	three

# 컬럼 E에 들어있는것만 필터링합니다.
df2[df2['E'].isin(['two','four'])]
#                       A	        B	        C	        D	   E
# 2013-01-03	-0.253946	-0.951220	 0.223419	-0.781845	 two
# 2013-01-05	-0.098203	-2.113593	-0.457045	 0.789558	four
```

### 데이터 변경하기 - Setting

`DataFrame` 안에 있는 데이터를 변경하려고 합니다.

우선 `Series`를 하나 만들고 기존에 생성했던 `DataFrame`에 붙여보겠습니다.

```python
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102',periods=6))
# 2013-01-02    1
# 2013-01-03    2
# 2013-01-04    3
# 2013-01-05    4
# 2013-01-06    5
# 2013-01-07    6
# Freq: D, dtype: int64

# Series를 'F' 컬럼에 넣는다.
df['F'] = s1
```

데이터 선택하기와 같은 속성 `at`, `iat`, `loc`, `iloc` 등을 그대로 사용하면 됩니다.

```python
# 0번째 인덱스, 'A' 컬럼을 0으로 변경
df.at[dates[0],'A'] = 0

# 0번째 인덱스, 1번째 컬럼을 0으로 변경
df.iat[0,1] = 0

# 전체 인덱스, 'D' 컬럼 데이터를 변경
df.loc[:,'D'] = np.array([5] * len(df))

df
#                       A	        B	        C	D	  F
# 2013-01-01	 0.000000	 0.000000	-1.126817	5	NaN
# 2013-01-02	 0.528919	 2.245356	-0.256795	5	1.0
# 2013-01-03	-0.253946	-0.951220	 0.223419	5	2.0
# 2013-01-04	-0.093062	 0.100206	 0.890981	5	3.0
# 2013-01-05	-0.098203	-2.113593	-0.457045	5	4.0
# 2013-01-06	-0.585436	-0.533013	 0.558793	5	5.0
```

조건문(where)으로 선택하여 데이터를 변경할 수도 있습니다.

```python
# 기존 DataFrame 복사
df2 = df.copy()
# 0보다 큰 데이터만 음수로 변경
df2[df2 > 0] = -df2

df2
#                       A	        B	        C	 D	   F
# 2013-01-01	 0.000000	 0.000000	-1.126817	-5	 NaN
# 2013-01-02	-0.528919	-2.245356	-0.256795	-5	-1.0
# 2013-01-03	-0.253946	-0.951220	-0.223419	-5	-2.0
# 2013-01-04	-0.093062	-0.100206	-0.890981	-5	-3.0
# 2013-01-05	-0.098203	-2.113593	-0.457045	-5	-4.0
# 2013-01-06	-0.585436	-0.533013	-0.558793	-5	-5.0
```

<a name="item4"></a>

## 4. 결측 데이터 - Missing Data
---
데이터를 다루다보면 값이 없는 경우가 자주 생깁니다. 데이터가 없는 것을 결측 데이터라고 합니다. 판다스에서는 이러한 값이 `NaN` 으로 표현됩니다. 
기본적으로 결측 데이터가 있는 경우에는 연산에 포함되지 않습니다.

> 이 장에 대해 자세한 설명은 [Missing Data section](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html#missing-data)을 확인하세요.

`reindex()`을 통해 컬럼이나 인덱스를 추가하거나, 삭제하거나 변경하는 등의 작업을 진행할 수 있습니다. 먼저 결측 데이터를 만들기 위해 'E' 컬럼을 생성합니다.

```python
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1],'E'] = 1

df1
#                       A	        B	        C	D	  F	  E
# 2013-01-01	 0.000000	 0.000000	-1.126817	5	NaN	1.0
# 2013-01-02	 0.528919	 2.245356	-0.256795	5	1.0	1.0
# 2013-01-03	-0.253946	-0.951220	 0.223419	5	2.0	NaN
# 2013-01-04	-0.093062	 0.100206	 0.890981	5	3.0	NaN
```

`DataFrame`의 `dropna()`를 통해 결측데이터를 삭제(drop)할 수 있습니다.
`how='any'`는 값들 중 하나라도 NaN인 경우 삭제입니다. `how='all'`은 전체가 NaN인 경우 삭제입니다.

```python
df1.dropna(how='any')
#                       A	       B	        C	D	  F   E
# 2013-01-02	0.528919	2.245356	-0.256795	5	1.0	1.0
```

`DataFrame`의 `fillna()`를 통해 결측데이터에 값을 넣을 수도 있습니다.

```python
df1.fillna(value=5)
#                       A	        B	        C	D	  F	  E
# 2013-01-01	 0.000000	 0.000000	-1.126817	5	5.0	1.0
# 2013-01-02	 0.528919	 2.245356	-0.256795	5	1.0	1.0
# 2013-01-03	-0.253946	-0.951220	 0.223419	5	2.0	5.0
# 2013-01-04	-0.093062	 0.100206	 0.890981	5	3.0	5.0
```

`pd.isnull()`을 통해 결측데이터 여부를 Boolean으로 가져올 수 있습니다.

```python
pd.isnull(df1)
#                   A	    B	    C	    D	    F	    E
# 2013-01-01	False	False	False	False	True	False
# 2013-01-02	False	False	False	False	False	False
# 2013-01-03	False	False	False	False	False	True
# 2013-01-04	False	False	False	False	False	True
```

<a name="item5"></a>

## 5. 데이터 연산 - Operations
---
> 이 장에 대해 자세한 설명은 [Basic section on Binary Ops](https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#basics-binop)을 참고하세요.

### 통계지표 - Stats
일반적으로 결측데이터는 빼고 계산됩니다.

`DataFrame`의 `mean()`으로 평균을 구할 수 있습니다.

```python
df.mean()
# A   -0.083621
# B   -0.208711
# C   -0.027911
# D    5.000000
# F    3.000000
# dtype: float64
```

같은 `mean()`을 다른 축(axis)에 대한 평균을 구할 수 있습니다.
여기서 축이란 0은 컬럼 기준, 1은 인덱스 기준을 말합니다.

```python
# axis = 1로 평균 구하기
df.mean(1)
# 2013-01-01    0.968296
# 2013-01-02    1.703496
# 2013-01-03    1.203651
# 2013-01-04    1.779625
# 2013-01-05    1.266232
# 2013-01-06    1.888069
# Freq: D, dtype: float64
```

만약 다른 차원의 오브젝트들 간 연산이 필요한 경우 축만 맞춰진다면 자동으로 연산을 수행합니다.

```python
# 데이터 시프트 연산 (2개씩 밀립니다.)
s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
# 2013-01-01    NaN
# 2013-01-02    NaN
# 2013-01-03    1.0
# 2013-01-04    3.0
# 2013-01-05    5.0
# 2013-01-06    NaN
# Freq: D, dtype: float64

# DataFrame와 Series를 index 축 기준으로 빼기 연산
df.sub(s, axis='index')
#                       A	        B	        C	  D	   F
# 2013-01-01	      NaN	      NaN	      NaN	NaN	 NaN
# 2013-01-02	      NaN	      NaN	      NaN	NaN	 NaN
# 2013-01-03	-1.253946	-1.951220	-0.776581	4.0	 1.0
# 2013-01-04	-3.093062	-2.899794	-2.109019	2.0	 0.0
# 2013-01-05	-5.098203	-7.113593	-5.457045	0.0	-1.0
# 2013-01-06	      NaN	      NaN         NaN	NaN	 NaN
```

### 함수 적용 - Apply
데이터에 대해 정의된 함수들이나 lamdba 식을 이용하여 새로운 함수도 적용할 수 있습니다.

```python
# 각 컬럼별(기본 axis = 0은 컬럼 기준) 누적합을 구합니다.
df.apply(np.cumsum)
#                       A	        B	        C	D	  F
# 2013-01-01	 0.000000	 0.000000	-1.126817	5	NaN
# 2013-01-02	 0.528919	 2.245356	-1.383612	10	1.0
# 2013-01-03	 0.274973	 1.294136	-1.160193	15	3.0
# 2013-01-04	 0.181910	 1.394342	-0.269212	20	6.0
# 2013-01-05	 0.083707	-0.719251	-0.726257	25	10.0
# 2013-01-06	-0.501728	-1.252264	-0.167465	30	15.0

# lambda 식을 이용하여 max-min의 값을 구합니다.
df.apply(lambda x: x.max() - x.min())
# A    1.114354
# B    4.358949
# C    2.017798
# D    0.000000
# F    4.000000
# dtype: float64
```

### 히스토그램 - Histogramming
> 이 장에 대해 자세한 설명은 [Histogramming and Discretization](https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#basics-discretization)을 참고하세요.

`Series` 데이터의 각 값이 어떤 분포를 이루는지 히스토그램 형식으로 볼 수 있습니다.

```python
s = pd.Series(np.random.randint(0, 7, size=10))
# 0    5
# 1    3
# 2    2
# 3    3
# 4    0
# 5    4
# 6    3
# 7    4
# 8    3
# 9    2
# dtype: int32

s.value_counts()
# 3    4
# 4    2
# 2    2
# 5    1
# 0    1
# dtype: int64
```

### 문자열 처리 - String Methods

> 이 장에 대해 자세한 설명은 [regular expressions](https://docs.python.org/3/library/re.html)와 [Vectorized String Methods](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#text-string-methods) 을 참고하세요.

`Series`에서 문자열 관련된 함수들은 `.str` 속성에 포함되어 있습니다.

`str.lower()`를 통해 문자를 소문자로 변경할 수 있습니다.

```python
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])

s.str.lower()
# 0       a
# 1       b
# 2       c
# 3    aaba
# 4    baca
# 5     NaN
# 6    caba
# 7     dog
# 8     cat
# dtype: object
```

<a name="item6"></a>

## 6. 데이터 합치기 - Merge
---
판다스는 `Series`와 `DataFrame` 간에 쉽게 데이터를 합칠 수 있도록 `join`과 `merge`와 같은 연산을 제공합니다.

### 이어붙이기 - Concat

> 이 장에 대해 자세한 설명은 [Merging section](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#merging)을 확인하세요.

`concat()`을 이용하여 이어붙이는 연산(Concatenating)을 할 수 있습니다.

```python
df = pd.DataFrame(np.random.randn(10, 4))
#           0	       1	        2       	3
# 0	 1.590092	 0.564066	 0.521476	 0.628626
# 1	-0.675627	 0.061639	-1.419665	-1.683719
# 2	-1.192757	-0.346602	 2.246563	 0.432095
# 3	 0.777386	-1.272256	-0.512404	 0.086339
# 4	 0.006978	 0.058503	-1.481919	 0.138789
# 5	-1.505119	-0.665700	-1.032970	 0.504922
# 6	-0.209184	-1.718691	-0.282240	-0.151445
# 7	 0.450164	 0.177133	-0.444645	 1.146489
# 8	-0.689884	 0.748130	 0.608955	-0.664917
# 9	 1.122287	-0.345145	 0.083216	 1.772749

# 위에서 생성한 DataFrame을 여러개로 분리합니다.
pieces = [df[:3], df[3:7], df[7:]]

# concat()을 이용하여 다시 이어붙일 수 있습니다.
pd.concat(pieces)
#           0	        1	        2	       3
# 0	 1.590092	 0.564066	 0.521476	 0.628626
# 1	-0.675627	 0.061639	-1.419665	-1.683719
# 2	-1.192757	-0.346602	 2.246563	 0.432095
# 3	 0.777386	-1.272256	-0.512404	 0.086339
# 4	 0.006978	 0.058503	-1.481919	 0.138789
# 5	-1.505119	-0.665700	-1.032970	 0.504922
# 6	-0.209184	-1.718691	-0.282240	-0.151445
# 7	 0.450164	 0.177133	-0.444645	 1.146489
# 8	-0.689884	 0.748130	 0.608955	-0.664917
# 9	1.122287	-0.345145	 0.083216	 1.772749
```

### 조인 - Join
SQL에서 자주 사용하는 `join` 연산도 제공됩니다.
> 이 장에 대해 자세한 설명은 [Database style joining](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#merging-join)을 참고하세요.

```python
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
#   key	 lval
# 0	foo	    1
# 1	foo	    2

right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
#   key	rval
# 0	foo	   4
# 1	foo	   5

# 위에서 생성한 left, right를 'key' 컬럼값 기준으로 조인합니다.
pd.merge(left, right, on='key')
# 	key	 lval	 rval
# 0	foo	    1	    4
# 1	foo	    1	    5
# 2	foo	    2	    4
# 3	foo	    2	    5
```

key가 다른 경우도 보겠습니다.

```python
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
#   key	 lval
# 0	foo	    1
# 1	bar	    2

right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
#   key	 rval
# 0	foo	    4
# 1	bar	    5

pd.merge(left, right, on='key')
#   key	 lval	 rval
# 0	foo	    1	    4
# 1	bar	    2	    5
```

아래 예제처럼 조인 연산 자체가 key가 같은 두 테이블을 합쳐서 하나의 테이블로 확인하기 위해 사용하는데, 만약 Key가 중복된 경우가 있으면 어떻게 나오는지는 위 예제로 확인하실 수 있습니다.

<a name="item7"></a>

## 7. 그룹화 - Grouping
---
`group by`에 관련된 내용은 아래와 같은 과정을 말합니다.
- Spltting : 특정 기준으로 데이터 나누기
- applying : 각 그룹에 함수를 독립적으로 적용시키는 것
- Combining : 결과를 데이터 구조로 저장하는 것

> 이 장에 대해 자세한 설명은 [Grouping section](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#groupby)을 참고하세요.

먼저 `DataFrame`을 하나 생성하겠습니다.

```python
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                                    'B' : ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
                                    'C' : np.random.randn(8),
                                    'D' : np.random.randn(8)})
#     A	  B	            C	        D
# 0	foo	one	     0.234679	-0.742428
# 1	bar	one	     0.794077	-0.444526
# 2	foo	two	    -1.103167	 1.191702
# 3	bar	three	-0.385300	-0.530512
# 4	foo	two	     0.476553	 1.637311
# 5	bar	two	     1.394824	 0.391403
# 6	foo	one	     0.677522	-0.441027
# 7	foo	three	 0.461757	 0.321676
```

특정 컬럼 기준으로 먼저 그룹화를 진행한 후 sum()을 적용합니다.

```python
#A 컬럼이 같은 것끼리 묶고, sum()
df.groupby('A').sum()
#             C	            D
#   A		
# bar	1.803601	-0.583636
# foo	0.747345	 1.967233

# 'A기준으로 묶고'. 'B' 기준으로 다시 묶은 후 sum()
df.groupby(['A','B']).sum()
#                       C	        D
#   A	B		
# bar	one	     0.794077	-0.444526
#       three	-0.385300	-0.530512
#       two	     1.394824	 0.391403
# foo	one	     0.912201	-1.183455
#       three	 0.461757	 0.321676
#       two	    -0.626613	 2.829013
```

<a name="item8"></a>

## 8. 데이터 구조 변경하기 - Reshaping
---
> 이 장에 대해 자세한 설명은 [Hierarchical Indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced-hierarchical)와 [Reshaping](https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-stacking)을 확인하세요.

### 스택 - Stack

먼저 새로운 `DataFrame`을 하나 생성하겠습니다.

```python
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]))
# 각 리스트에서 첫번째를 'first'로 두번째를 'second'로 멀티인덱스를 만듭니다.
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
# 두 개의 컬럼을 생성하고 랜덤값을 부여합니다.
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])

df2 = df[:4]
#                       A	        B
# first	second		
# bar	one	    -0.550713	-2.632594
#       two	     1.114643	-1.231132
# baz	one	     2.022526	-0.852112
#       two	     1.194659	 0.729303
```

`DataFrame`의 `stack()`은 모든 데이터들을 인덱스 레벨로 변형합니다. 이를 압축(compresses)한다고 표현합니다.

```python
stacked = df2.stack()

# first  second   
# bar    one     A   -0.550713
#                B   -2.632594
#        two     A    1.114643
#                B   -1.231132
# baz    one     A    2.022526
#                B   -0.852112
#        two     A    1.194659
#                B    0.729303
# dtype: float64
```

`unstack()` 을 통해 "stacked"된 `DataFrame` 이나 `Series` 를 원래 형태로 되돌릴 수 있습니다. 되돌리는(압축 해제) 것의 레벨을 정할 수 있습니다.

```python
# 원래 상태로 되돌립니다. (레벨 -1)
stacked.unstack()
#                       A	        B
# first	second		
# bar	one	    -0.550713	-2.632594
#       two	     1.114643	-1.231132
# baz	one	     2.022526	-0.852112
#       two	     1.194659	 0.729303

# 레벨1은 second 인덱스가 해제되어 one과 two 컬럼이 생깁니다.
stacked.unstack(1)
# 	    second	  one 	    two
# first			
# bar	A	-0.550713	 1.114643
#       B	-2.632594	-1.231132
# baz	A	 2.022526	 1.194659
#       B	-0.852112	 0.729303

# 레벨0은 first 인덱스가 해제되어 bar와 baz 컬럼이 생깁니다.
stacked.unstack(0)
# 	        first	  bar	     baz
# second			
# one	    A	-0.550713	 2.022526
#           B	-2.632594	-0.852112
# two	    A	 1.114643	 1.194659
#           B	-1.231132	 0.729303
```

### 피벗 테이블 - Pivot Tables
> 이 장에 대해 자세한 설명은 [Pivot Tables](https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-pivot)을 참고하세요.

우선 `DataFrame`을 하나 생성하겠습니다.

```python
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                   'B' : ['A', 'B', 'C'] * 4,
                   'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D' : np.random.randn(12),
                   'E' : np.random.randn(12)})
#         A	    B	  C	        D	        E
# 0	    one	    A	foo	 1.841912	-1.254732
# 1	    one	    B	foo	-0.205527	 0.927007
# 2	    two	    C	foo	-0.519763	 0.900022
# 3	    three	A	bar	-0.240530	-0.124600
# 4	    one	    B	bar	-0.090635	-1.355138
# 5	    one	    C	bar  0.918863	-1.493705
# 6	    two 	A	foo	 1.225386	 1.518278
# 7	    three	B	foo	-1.902415	-1.774154
# 8	    one	    C	foo	 0.028049	-1.728172
# 9	    one	    A	bar	 1.617027	 0.273678
# 10	two	    B	bar	-0.839234	-0.650577
# 11	three	C	bar	 0.618831	 0.138547                                    
```

`pd.pivot_table()`을 통해 새로운 피벗으로 테이블을 만들 수 있습니다.

```python
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
# 	    C	      bar	     foo
#   A	B		
# one	A	 1.617027	 1.841912
#       B	-0.090635	-0.205527
#       C	 0.918863	 0.028049
# three	A	-0.240530	      NaN
#       B	      NaN	-1.902415
#       C	 0.618831	      NaN
# two	A	      NaN	 1.225386
#       B	-0.839234	      NaN
#       C	      NaN	-0.519763
```

<a name="item9"></a>

## 9. 시계열 데이터 - Time Series
---
판다스는 시계열 데이터를 주기를 변경하거나 샘플링하는데 간단하고 강력한 기능을 제공합니다. 또한 금융 데이터를 다루기에도 편리합니다.
(예를 들어 1초마다 쌓은 데이터를 5분 단위로 변경하고 싶을 때)

> 이 장에 대해 자세한 설명은 [Time Series section](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries)을

`resample()`을 이용하여 데이터 샘플링을 할 수 있습니다.

```python
# 1초 단위로 100개의 index 생성
rng = pd.date_range('1/1/2012', periods=100, freq='S')
# 0~500 사이 랜덤값 입력
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)

# 5분 단위로 샘플링하여 sum()
ts.resample('5Min').sum()
# 2012-01-01    23289
# Freq: 5T, dtype: int32
```

다양한 타임존으로 변경할 수도 있습니다.

```python
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
# 2012-03-06    1.308232
# 2012-03-07    0.936681
# 2012-03-08    0.013171
# 2012-03-09    0.042598
# 2012-03-10    0.877987
# Freq: D, dtype: float64

# 표준시(UTC)로 변경
ts_utc = ts.tz_localize('UTC')
# 2012-03-06 00:00:00+00:00    1.308232
# 2012-03-07 00:00:00+00:00    0.936681
# 2012-03-08 00:00:00+00:00    0.013171
# 2012-03-09 00:00:00+00:00    0.042598
# 2012-03-10 00:00:00+00:00    0.877987
# Freq: D, dtype: float64
```

```python
# US 동부 시각으로 변경
ts_utc.tz_convert('US/Eastern')
# 2012-03-05 19:00:00-05:00    1.308232
# 2012-03-06 19:00:00-05:00    0.936681
# 2012-03-07 19:00:00-05:00    0.013171
# 2012-03-08 19:00:00-05:00    0.042598
# 2012-03-09 19:00:00-05:00    0.877987
# Freq: D, dtype: float64
```

시간 간격(TimeSpan)도 쉽게 표현할 수 있습니다.

```python
# 매달('M') 기준으로 생성
rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
# 2012-01-31    0.641649
# 2012-02-29   -0.148456
# 2012-03-31    0.169228
# 2012-04-30    0.769451
# 2012-05-31   -0.389726
# Freq: M, dtype: float64

ps = ts.to_period()
# 2012-01    0.641649
# 2012-02   -0.148456
# 2012-03    0.169228
# 2012-04    0.769451
# 2012-05   -0.389726
# Freq: M, dtype: float64

ps.to_timestamp()
# 2012-01-01    0.641649
# 2012-02-01   -0.148456
# 2012-03-01    0.169228
# 2012-04-01    0.769451
# 2012-05-01   -0.389726
# Freq: MS, dtype: float64
```

기간(period)과 시간(timestamp) 사이에 산술적인 기능들을 적용할 수 있습니다.
아래 예제에서는 분기별로 9시간을 더한 시각을 기준으로 볼 수 있습니다.

```python
# 분기 단위로 시간 인덱스를 생성합니다.
prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9

ts.head()
# 1990-03-01 09:00    0.518888
# 1990-06-01 09:00   -1.919605
# 1990-09-01 09:00   -0.489806
# 1990-12-01 09:00    0.352551
# 1991-03-01 09:00   -0.355452
# Freq: H, dtype: float64
```

<a name="item10"></a>

## 10. 범주형 데이터 - Categoricals
---
`DataFrame` 안에는 범주형 데이터도 넣을 수 있습니다.

> 이 장에 대해 자세한 설명은 [categorical introduction](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html#categorical)와 [API documentation](https://pandas.pydata.org/pandas-docs/stable/reference/arrays.html#api-arrays-categorical)을 참고하세요.

`.astype("category")`로 범주형 데이터 타입으로 변환할 수 있습니다.

```python
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
# category형 데이터 타입으로 변환합니다.
df["grade"] = df["raw_grade"].astype("category")

df["grade"]
# 0    a
# 1    b
# 2    b
# 3    a
# 4    a
# 5    e
# Name: grade, dtype: category
# Categories (3, object): [a, b, e]
```

`cat.categories` 속성을 이용하여 카테고리명을 다시 만들 수 있습니다.

```python
df["grade"].cat.categories = ["very good", "good", "very bad"]
```

`cat.set_categories()`를 이용하여 카테고리를 재정의할 수 있습니다. 재정의는 현재 갖고 있지 않은 범주도 추가할 수 있습니다.

```python
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
# 0    very good
# 1         good
# 2         good
# 3    very good
# 4    very good
# 5     very bad
# Name: grade, dtype: category
# Categories (5, object): [very bad, bad, medium, good, very good]
```

범주형 데이터를 정렬할 수도 있습니다. 정렬 기준은 문자 기준이 아닌 범주형 자료를 정의할 때 만든 순서로 정렬됩니다.

```python
df.sort_values(by="grade")
#   id	raw_grade	grade
# 5	6	e	     very bad
# 1	2	b	         good
# 2	3	b	         good
# 0	1	a	    very good
# 3	4	a	    very good
# 4	5	a	    very good
```

범주형 데이터 기준으로 그룹화하여 빈도수를 출력하면, 비어있는 범주도 쉽게 확인할 수 있습니다.

```python
df.groupby("grade").size()
# grade
# very bad     1
# bad          0
# medium       0
# good         2
# very good    3
# dtype: int64
```

<a name="item11"></a>

## 11. 그래프 시각화 - Plotting
---
> 이 장에 대해 자세한 설명은 [Plotting](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#visualization)을 참고하세요.

데이터 시각화는 [matplotlib](https://matplotlib.org/) API를 사용합니다.

```python
# 랜덤 값으로 1000개의 행을 생성합니다.
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
# 누적합을 구합니다.
ts = ts.cumsum()
# 시각화합니다.
ts.plot()
```

![](https://laboputer.github.io/assets/img/ml/python/pandas/2.png)

`DataFrame`에서 `plot()`은 모든 컬럼을 한번에 보여주는 편리함도 있습니다.

```python
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                  columns=['A', 'B', 'C', 'D'])
df = df.cumsum()

plt.figure()
df.plot()
plt.legend(loc='best')
```

![](https://laboputer.github.io/assets/img/ml/python/pandas/3.png)

<a name="item12"></a>

## 12. 파일 입출력 - Getting Data In/Out
---
`DataFrame`을 파일로 생성하거나 파일을 `DataFrame`으로 읽는 방법입니다. 

## CSV
> 이 장에 대해 자세한 설명은 [Writing to a csv file](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-store-in-csv), [Reading from a csv file](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-read-csv-table)을 참고하세요.

`.to_csv()`로 `DataFrame`을 쉽게 csv 파일로 쓸 수 있습니다.

```python
df.to_csv('foo.csv')
```

`pd.read_csv()`로 csv파일을 `DataFrame`으로 읽어올 수 있습니다.

```python
pd.read_csv('foo.csv')
# Unnamed:              0	A	B	C	D
# 0	2000-01-01	-0.196030	-0.159549	-1.110689	-0.043820
# 1	2000-01-02	 0.804142	-1.825197	-1.041056	 1.157358
# 2	2000-01-03	-0.927607	-2.698263	-0.554641	 0.644156
# 3	2000-01-04	-0.504833	-1.787893	-0.499680	 1.257813
# 4	2000-01-05	-1.117570	-1.162072	-1.229462	 1.547675
# ...	...	...	...	...	...
# 995	2002-09-22	83.381870	-26.304967	-23.922886	-17.836663
# 996	2002-09-23	83.593856	-26.037092	-23.621604	-17.229214
# 997	2002-09-24	84.450600	-26.228553	-22.055964	-17.320377
# 998	2002-09-25	84.606621	-26.459262	-21.544691	-17.587936
# 999	2002-09-26	84.434890	-24.759643	-22.093779	-18.191066
# 1000 rows × 5 columns
```

### HDF5
> 이 장에 대해 자세한 설명은 [HDFStores](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-hdf5)을 참고하세요.

`to_hdf()`로 HDF5 형식으로 쓸 수 있습니다.

```python
df.to_hdf('foo.h5','df')
```

`pd.read_hdf()`로 hdf5 형식을 `DataFrame`으로 읽어올 수 있습니다.

```python
pd.read_hdf('foo.h5','df')
# 	                A	        B	        C	        D
# 2000-01-01	-0.067377	-0.035638	-0.021100	-0.752226
# 2000-01-02	-0.341526	 0.867899	 0.147792	-2.190171
# 2000-01-03	-0.118822	 1.501776	-0.472641	-1.448117
# ...........
# 1000 rows × 4 columns
```

### Excel
> 이 장에 대해 자세한 설명은 [MS Excel](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-excel)을 참고하세요.

`to_excel()`로 xlsx 파일 형식으로 쓸 수 있습니다.

```python
df.to_excel('foo.xlsx', sheet_name='Sheet1')
```

`pd.read_excel()`로 xlsx 파일로부터 `DataFrame`으로 읽어올 수 있습니다.

```python
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
```

---
이 포스팅에서 사용한 코드는 [이곳](https://github.com/Laboputer/LearnML/blob/master/02.%20%5BPOST%5D/50.%20%5BCode%5D%20Pandas%20to%2010%20minutes.ipynb)에 공개되어 있으며 다운로드 받으실 수 있습니다.