---
layout: post
title:  "맷플롯립(Matplotlib), 데이터 시각화 알아보기 - 기본"
subtitle:   "matplotlib"
categories: machine-learning
tags: python tutorial matplotlib
---

> 이 글은 Matplotlib 공식홈페이지에서 소개된 [Pyplot tutorial](https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py)을 기본으로 데이터 분석 시 자주 사용하는 시각화 방법들을 추가하여 정리하였습니다. 처음 시작하는 분들도 이해할 수 있게 설명하려고 노력하였으니 도움이 되시길 바랍니다.

이 글을 읽으신 후에 더 자세한 내용이 필요하시면 아래 링크를 확인하세요.

- [공식홈페이지](https://matplotlib.org/)
- [Documentation](https://matplotlib.org/contents.html)
- [Usage Guide](https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py)
- [Gallery](https://matplotlib.org/gallery/index.html)

## 목차
---

 1. [기본 개념](#item1)
 2. [Line Plot and Figure](#item2)
 3. [Histogram](#item3)
 4. [Bar Chart](#item4)
 5. [Pie Chart](#item5)
 6. [Scatter Chart](#item6)
 7. [Color Map](#item7)
 8. [Axes3D](#item8)


## Matplotlib
---

Matplotlib은 Python에서 데이터를 시각화해주는 패키지입니다. 빅데이터들을 분석 시에 한 눈에 파악하기가 어려울 때 시각화하여 직관적으로 이해하여 분석에 활용할 수 있습니다.

`matplotlib`은 `pyplot`이라는 서브패키지를 사용합니다. 기본적인 시각화는 이것으로 충분합니다.
> 패키지가 없는 경우, 설치 명령어 : pip install matplotlib

`numpy`는 데이터를 쉽게 생성하기 위해 사용하였습니다.
> 궁금하신 분은 제 포스팅 [넘파이(Numpy) 사용법 알아보기 - 기본](https://laboputer.github.io/machine-learning/2020/04/25/numpy-quickstart/)을 읽어보세요.


```python
# 주피터 노트북 사용 시, 노트북 내 이미지를 그릴 수 있도록 설정하는 매직 명령
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
```

<a name="item1"></a>

## 1. 기본 개념
---

### Plot
먼저 플롯(Plot)한다라는 의미는 그래픽으로 렌더링하여 화면으로 출력하기 전에 데이터를 가상의 공간에 미리 그려놓는 것을 말합니다. 이 plot은 실제로 차트를 그리는 것이 아니라 내부적으로 그릴 준비를 한다고 생각하시면 됩니다. 최종적으로 `plt.show()`를 하여야 차트가 시각화되어 보이게 됩니다.

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/2.JPG)

위 그림에 나온 시각화들은 각 장에서 다뤄볼 예정입니다. 각 차트들의 이름은 대부분 코딩할 때 사용하므로 영어 그대로 익숙해지는 것이 좋습니다. 

> 더 다양한 시각화 방법이 궁금하신 분은 [Gallery](https://matplotlib.org/gallery/index.html)를 확인하세요.
 
 ### 참고사항
1장에서는 가장 기본적인 Line plot(선 차트)을 가지고 `Figure`의 구성요소들을 정리하고 2장부터는 다양한 모양의 시각화 방법들을 정리하였습니다. `Figure`에 다양하게 넣을 수 있는 구성요소들을 알아보고 싶으시면 1장을 보시고, 단순히 다양한 시각화 방법을 보시고 싶은 분들은 각 장에서 사용법만 읽으셔도 됩니다.

<a name="item2"></a>

## 2. Line plot, Figure
---

### Line plot, 선 그리기
2D 데이터(X와 Y 값)를 기준으로 선이 이어지게 시각화합니다.
> 자세한 내용은 [pyplot.plot](https://matplotlib.org/api/pyplot_api.html#Matplotlib.pyplot.plot)을 확인하세요

`plt.plot()`을 이용하여 기본적으로 plot(x, y) 순서로 2개의 리스트를 넣어주지만, 만약 하나의 리스트만 넣을 경우 x는 자동적으로 0부터 할당됩니다.


```python
plt.plot([3, 1, 5, 2])
plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_7_0.png)


Line plot을 이용하면 'y = sin(x)'과 같은 수학 그래프도 쉽게 그릴 수 있습니다.
> `np.linspace()`는 0 ~ 2*PI 사이의 값을 50등분한 값을 생성합니다.


```python
x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)

plt.plot(x, y)
plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_9_0.png)


기본적으로 Line Plot을 그리는 방법을 알았으니, 1장에서는 아래 이미지에 나오는 용어들을 먼저 정리해보려고 합니다. 앞으로 자주 사용할 용어이므로 하나씩 어떤 코드로, 어떻게 나타나는지 알아볼 예정이니 역시 익숙해지는 것이 좋습니다.

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/1.JPG)


### 중복데이터 그리기

같은 차트에 2개 이상의 그래프를 그리고 싶을 때는 `plot()`에 데이터를 추가로 넣으면 하나의 차트에 여러개의 데이터를 보여줄 수 있습니다.


```python
x = np.linspace(0, 2*np.pi, 50)

plt.plot(x, np.sin(x), x, np.cos(x))

plt.show()
```


![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_12_0.png)


하나의 plot()에 한번에 넣지 않고, 여러번의 `plot()`을 해도 동일한 결과가 나타납니다.


```python
x = np.linspace(0, 2*np.pi, 50)

plt.plot(x, np.sin(x))
plt.plot(x, 10*np.cos(2*x))

plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_14_0.png)


같은 plot()으로 x축은 공유하되, y축의 스케일을 다르게 하고 싶을 때는 `twinx()` 를 이용합니다.
1번째 plot()은 차트 왼쪽에 y축 스케일이, 2번째 plot()은 우측에 y축 스케일이 보이는 것을 확인하실 수 있습니다.


```python
x = np.linspace(0, 2*np.pi, 50)

ax1 = plt.gca()
ax2 = plt.gca().twinx()

ax1.plot(x, np.sin(x))
ax2.plot(x, 10*np.cos(2*x), color='y')

plt.show()
```


![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_16_0.png)


### Subplot, 차트를 여러개로 나누기

이번에는 하나의 차트에 여러 데이터를 중복해서 그리는 것이 아니라, 하나의 이미지에서 여러 부분으로 나누어 각각의 차트를 만들 수도 있습니다. 이것은 `subplot()`을 이용합니다. 세 가지(nrows, ncols, index)를 입력을 받으면 nrows는 행의 개수, ncols는 열의 개수, index는 보여줄 위치로 1부터 지정하고 왼쪽 위부터 오른쪽 아래로 하나씩 지정됩니다. 세 개의 인수 (2,2,3)을 넣고 싶다면 숫자 223을 넣어도 동작합니다.

- `subplot(nrows, ncols, index)`: 행의 개수, 열의 개수, 차트 위치 순으로 입력
- `subplot(3digit int)`: 3자리 숫자로, 행/열/위치를 한번에 설정


```python
x1 = np.linspace(0, 2*np.pi, 50)
y1 = np.sin(x1)

x2 = np.linspace(0.0, 2*np.pi, 50)
y2 = np.tan(x2)

# 1x2구간으로 나누고 1번째에 그리기
plt.subplot(1, 2, 1)
plt.plot(x1, y1)

# 1x2구간으로 나누고 2번째에 그리기
plt.subplot(1, 2, 2)
plt.plot(x2, y2)

plt.show()
```


![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_18_0.png)


`subplot()`을 4가지 영역으로 나눠서 표현하는 예제입니다. 입력값을 편하게 하기 위해서 (2, 2, 3) 대신 223을 넣어도 됩니다. 
> `.cumsum()`은 누적합 데이터를 생성합니다.


```python
y1 = npr.normal(size = 50).cumsum()
y2 = npr.normal(size = 100).cumsum()
y3 = npr.normal(size = 50).cumsum()
y4 = npr.normal(size = 100).cumsum()

plt.subplot(2, 2, 1)
plt.plot(y1)

# same as plt.subplot(222)
plt.subplot(2, 2, 2)
plt.plot(y2)

# same as plt.subplot(2,2,3)
plt.subplot(223)
plt.plot(y3)

plt.subplot(224)
plt.plot(y4)

plt.tight_layout()
plt.show()
```


![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_20_0.png)


### Legend, 차트의 범례 그리기

`legend()`를 이용하여 각 데이터가 무엇인지 설명해주는 Legend(범례)를 넣을 수도 있습니다.
- `legend()` : 차트에 각 데이터의 설명 추가


```python
x = np.linspace(0, 2*np.pi, 50)

plt.plot(np.sin(x), label='sin')
plt.plot(np.cos(x), label='cos')
plt.legend()

plt.show()
```


![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_22_0.png)


`plot()`안에 label 값으로 legend를 넣지 않고 `legend()` 안에 리스트 형식으로 순서대로 넣어도 동일한 결과를 나타납니다.


```python
x = np.linspace(0, 2*np.pi, 50)

plt.plot(np.sin(x))
plt.plot(np.cos(x))
plt.legend(['sin', 'cos'])

plt.show()
```


![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_24_0.png)


### Title and Axis Label, 차트 및 축 이름 설정하기

`title()`을 통해 차트의 이름(Title)을 입력할 수 있고, `xlabel()`, `ylabel()`을 통해 각 축의 이름(Label)을 넣을 수도 있습니다.

- `.title()` : 차트의 제목 설정
- `.xlabel()` : x축의 이름 설정
- `.ylabel()` : y축의 이름 설정


```python
x = np.linspace(0, 2, 100)

plt.plot(x, np.sin(x))
plt.title('Title : sin(x)', fontsize=15)
plt.xlabel('X : radians')
plt.ylabel('Y : amplitude')

plt.show()
```


![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_26_0.png)


### Grid, 그리드 설정

`plt.grid()`를 통해 차트의 배경을 Grid 형식으로 라인을 그려줄 수도 있습니다. 
`.show()` 대신 `.grid()`로 차트를 출력하면 됩니다.

- `.grid()` : 차트에 Grid 표시


```python
x = np.linspace(0, 2, 100)

plt.plot(x, np.sin(x))
plt.title('Line Plot : sin(x)', fontsize = 15)
plt.xlabel('X : radians')
plt.ylabel('Y : amplitude', fontsize='large')

plt.grid()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_28_0.png)


### 차트 범위 지정

기본적으로 차트에 그려질 x, y 범위에 따라 자동으로 차트의 스케일이 정해지는데, `xlim()`, `ylim()`을 이용하여 스케일을 원하는 범위로 설정할 수 있습니다.

- `xlim()` : x축 범위 설정
- `ylim()` : y축 범위 설정


```python
y1 = npr.normal(size = 50).cumsum()

# (왼쪽) default xlim, ylim
plt.subplot(121)
plt.plot(y1)
plt.title('Y1')

# (오른쪽) xlim, ylim 사용자 설정
plt.subplot(122)
plt.plot(y1)
plt.xlim(0, 100)
plt.ylim(-10, 10)
plt.title('Y2')

plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_30_0.png)


### 스타일 지정
`plot()`에 보기가 편하도록 여러가지 스타일을 정해줄 수도 있습니다.
점의 색깔(Color), 점의 모양(Markers), 선 스타일(Line style)를 문자열로 쉽게 지정할 수도 있습니다. 위 순서대로 각 약자로 지정하고, 이 중 생략된 것은 디폴트값이 적용됩니다.

- Color (색깔) : 약자 또는 컬러이름 또는 '#RGB'로 지정가능
    - `b` : blue
    - `g` : green
    - `r` : red
    - `y` : yellow 
    - `k` : black
- Markers (점 모양): 문자 형태로 지정가능
    - `.` : point marker
    - `o` :	circle marker
    - `^` :	triangle_up marker
    - `*` : star marker
    - `x` : x marker
    - `D` : diamond marker
- Line style (선 스타일) : 문자 형태로 지정가능
    - `-` : solid line
    - `--` : dashed line
    - `-.` : dash-dot line
    - `:` : dotted line

> 더 많은 스타일이 궁금하시면 아래 링크를 확인하세요.
> - [Colors](https://matplotlib.org/examples/color/named_colors.html)
> - [Markers](https://matplotlib.org/examples/lines_bars_and_markers/marker_reference.html)


```python
x = np.linspace(0, 2*np.pi, 20)

# b(blue: 파란색), s(Square: 네모 점)
plt.plot(x, np.sin(x), color='blue', linestyle='', marker='s')
# g(green: 초록색), --(대시로 선 잇기), ^(세모 모양의 점)
plt.plot(x, np.cos(x), color='green', linestyle='--', marker='^')
# r(red: 빨간색), :(점선으로 잇기), o(동그라미 점)
plt.plot(x, np.sin(2*x), color='red', linestyle=':', marker='o')

plt.title('Line Plot')
plt.legend(['item1', 'item2', 'item3'])
plt.show()
```


![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_32_0.png)


Style을 각각 지정해도 되지만 아래와 같이 하나의 문자열로 편하게 지정할 수도 있습니다.
위와 동일한 결과를 나타내는 코드입니다.


```python
x = np.linspace(0, 2*np.pi, 20)

# same as plt.plot(x, np.sin(x), color='blue', marker='s')
plt.plot(x, np.sin(x), 'bs')
# same as plt.plot(x, np.cos(x), color='green', linestyle='--', marker='^')
plt.plot(x, np.cos(x), 'g--^')
# same as plt.plot(x, np.sin(2*x), color='red', linestyle=':', marker='o')
plt.plot(x, np.sin(2*x), 'r:o')

plt.title('Line Plot')
plt.legend(['item1', 'item2', 'item3'])
plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_34_0.png)


### Errorbar, 선에 에러바 표기

`plt.plot()` 대신 `plt.errorbar()`를 사용하면 Line Plot의 각 포인트에 errorbar를 넣을 수 있습니다.

- `.errorbar(x, y, yerr, xerr)`: `plot(x,y)` 내부 각 포인트에 yerr크기의 세로선, xerr 크기의 가로선


```python
# Plot with Error bars
x = np.linspace(0, 2*np.pi, 10)
y = np.sin(x)
xerr = np.linspace(0, 1, 10) / 5
yerr = np.linspace(1, 0, 10) / 5

plt.errorbar(x, y, yerr, xerr)

plt.title('Line with errorbars')
plt.legend(['error'])
plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_36_0.png)


`errorbar()` 안에 입력 값으로 `uplims`, `lolims` 값 설정을 통해 화살표 형식으로 표기할 수도 있습니다.

- `uplims` : 아래로 향하는 화살표
- `lolims` : 위로 향하는 화살표


```python
x = np.linspace(0, 2*np.pi, 10)
y = np.sin(x)
xerr = np.linspace(0, 1, 10) / 5
yerr = np.linspace(1, 0, 10) / 5

plt.errorbar(x, y, yerr, xerr, 
             label = 'both limits (default)')

plt.errorbar(x, y + 1, yerr=yerr, uplims=True, 
             label='uplims=True')

plt.errorbar(x, y + 2, yerr=yerr, uplims=[True, False] * 5, lolims=[False, True] * 5, 
             label='subsets of uplims and lolims')

plt.title('Line with errorbars')
plt.legend()
plt.show()
```


![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_38_0.png)


<a name="item3"></a>

## 3. Histogram
---

`plt.hist()`를 통하여 히스토그램을 그릴 수 있습니다. 히스토그램은 연속적인 데이터를 막대그래프로 나타냅니다.
> 자세한 내용은 [pyplot.hist](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist)를 확인하세요.

> `np.random.uniform()` 을 통해 균일한 랜덤 값을, `np.random.normal()`을 통해 가우시안 랜덤 분포값을 생성합니다.


```python
x = npr.uniform(size = 1000)

plt.hist(x)

plt.title('Histogram')
plt.legend(['items'])
plt.show()
```


![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_40_0.png)



```python
x = npr.normal(size = 1000)

plt.hist(x)

plt.title('Histogram')
plt.legend(['items'])
plt.show()
```


![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_41_0.png)


<a name="item4"></a>

## 4. Bar Chart
---
`plt.bar()`를 이용하여 히스토그램과 비슷한 형식의 막대 그래프를 만들 수 있습니다.
> 자세한 내용은 [pyplot.bar](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.bar)를 확인하세요.

`.hist()`는 연속적인 데이터를 표현하고 `.bar()`는 카테고리를 나눌 수 있는 데이터를 표현합니다. `.barh()` 를 이용하면 가로 막대 형식으로 출력할 수 있습니다.

- `.bar()` : 세로 막대 차트
- `.barh()` : 가로 막대 차트



```python
x = ['item1', 'item2', 'item3', 'item4']
y = [32, 123, 53, 11]

plt.bar(x,y)

plt.title('Bar Chart')
plt.legend(['items'])
plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_43_0.png)

```python
x = ['item1', 'item2', 'item3', 'item4']
y = [32, 123, 53, 11]

plt.barh(x,y)

plt.title('Bar Chart')
plt.legend(['items'])
plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_44_0.png)

<a name="item5"></a>

## 5. Pie Chart
---

`plt.pie()`를 이용하여 파이 모양의 차트를 표현할 수 있습니다. 
> 자세한 내용은 [pyplot.pie](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.pie)를 확인하세요.

자주 사용하는 옵션 2가지만 소개하겠습니다. `autopct`은 각 파이 각각에 문자열을 출력하는 것인데 예제에서 '%.2f'는 소수점 두번째만 출력하라는 의미이고 '%%'는 문자 '%'를 표현하는 것입니다. `explode`는 보통 강조할 때 쓰이는 것으로 리스트를 입력하면 각 파이가 떨어진 정도를 나타낼 수 있습니다.

- `autopct` : 각 파이 문자열 출력 형식 설정
- `explode` : 각 파이를 분리하여 표현하는 정도


```python
x = ['item1', 'item2', 'item3', 'item4']
y = [32, 123, 53, 11]

plt.pie(y, labels=x, autopct='%.2f%%', explode=(0, 0.1, 0, 0))

plt.title('Bar Chart')
plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_46_0.png)

<a name="item6"></a>

## 6. Scatter Chart
---

`plt.scatter()`를 이용하여 2차원 상에서 점을 차트를 표현할 수 있습니다.

> 자세한 내용은 [pyplot.scatter](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter)를 확인하세요.

두 개의 실수 데이터를 상관 관계를 한눈에 볼 수 있습니다.


```python
x = npr.normal(size=100)
y = npr.normal(size=100)

plt.scatter(x,y, c='red')

plt.title('Scatter Chart')
plt.legend(['(x,y)'])
plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_48_0.png)

`plt.scatter()` 에서 점의 반지름을 설정할 수 있습니다. 그렇게 되면 3차원의 데이터 형식을 표현할 수 있게 됩니다. 이를 'Bubble Chart'라고 부르기도 합니다.


```python
x = npr.normal(size=100)
y = npr.normal(size=100)
z= npr.normal(size=100)*100

plt.scatter(x,y,z, c='violet')

plt.title('Bubble Chart')
plt.legend(['(x,y,z)'])
plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_50_1.png)


<a name="item7"></a>

## 7. ColorMap
---

`plt.imshow()`를 이용하면 행과 열로 표현된 2D 데이터를 각 수치 정도에 따라 색으로 표현할 수 있습니다. 다시 말하면 이미지화 시키는 것을 말합니다.

> 자세한 내용은 [Colormap](https://matplotlib.org/tutorials/colors/colormaps.html) 또는 [Heatmap](https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py)을 확인하세요.


```python
x = npr.normal(size = 100).reshape(10, 10) / 100
print(x)

# [[-0.0158119   0.0008522   0.01655042 -0.00064181  0.01636717 -0.00431618
#   -0.00923994  0.00524193 -0.00141358 -0.00023031]
#   [ 0.0016493  -0.0001559   0.01560591  0.01966844  0.01251797 -0.01903377
#     0.00532141  0.0069562   0.01258736  0.00230315]
#   [ 0.01667079 -0.00875421 -0.00204732 -0.00342779  0.01002274  0.00358062
#     0.0012052   0.00320403  0.00276938  0.00489377]
#   [-0.01962862 -0.01302604  0.00774131 -0.01036093 -0.00331815  0.00068648
#     0.00182531 -0.0036629   0.00545105  0.00185453]
#   [ 0.00072316  0.021296   -0.0099748  -0.01951083 -0.02903795 -0.01822232
#     0.00862724 -0.01246434 -0.01157945  0.01329665]
#   [ 0.00641971  0.01705669  0.00098373 -0.0066299  -0.00908504 -0.01359947
#   -0.00071791 -0.00886265 -0.0021053  -0.00093484]
#   [-0.01116248 -0.00283873 -0.01351742 -0.00746876  0.01435057 -0.00149578
#   -0.01902301 -0.02355377 -0.0020175   0.0046488 ]
#   [-0.00733449 -0.01572126  0.01201231 -0.01953659  0.01178389 -0.02817181
#     0.00963386  0.00599131  0.00295523 -0.00533833]
#   [ 0.00467253 -0.01102508 -0.01272875  0.02023078 -0.01444144 -0.00363434
#     0.00590492 -0.0093903  -0.00711697  0.00442914]
#   [-0.00622509  0.00496323 -0.00268984  0.01460282 -0.00498456  0.02567855
#   -0.00786389 -0.00354781  0.00212546 -0.01473025]]

plt.imshow(x)
plt.colorbar()

plt.title('Color Map')
plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_52_1.png)


`matplotlib.image`의 `imread()`를 이용하여 실제 이미지 파일을 읽어왔습니다. 여러분이 가지고 있는 다양한 이미지 파일을 똑같이 읽어올 수 있습니다. 이미지 데이터를 그대로 `imshow()`를 이용하여 시각화하면 이미지가 됩니다. `cmap = 'gray'`로 흑백 시각화를 하였습니다.


```python
import matplotlib.image as mpimg
img = mpimg.imread('./dog.jpg')
x = img
print(x)

# [[ 88  88  89 ...  62  61  61]
#  [ 88  88  89 ...  62  61  61]
#  [ 89  89  90 ...  61  61  61]
#  ...
#  [145 145 145 ... 104 103 103]
#  [142 144 148 ... 104 103 103]
#  [142 146 152 ... 104 103 103]]

plt.imshow(x, cmap='gray')

plt.colorbar()
plt.title('Gray Colormap')
plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_54_1.png)

<a name="item8"></a>

## 8. Axes3D
---
`Axes3D` 패키지를 이용하여 axes를 3차원 공간으로 생성할 수도 있습니다.

> 자세한 내용은 [Axes3D](https://matplotlib.org/3.1.1/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html?highlight=axes3d)를 확인하세요.

아래 예제는 `plot_surface()`를 이용하여 3D 형태의 표면을 생성하여 입체도형을 만들었습니다. 


```python
from mpl_toolkits.mplot3d import Axes3D
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, cmap='hot')

ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_zlabel('Z values')
plt.title("3D Surface Plot")
plt.show()
```

아래 예제는 2차원 공간에 `scatter()`를 사용한 것처럼 Axes3D 공간에 `scatter()`를 생성한 모습입니다.

```python
from mpl_toolkits.mplot3d import Axes3D
X = npr.normal(size = 100)
Y = npr.normal(size = 100)
Z = npr.normal(size = 100)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X, Y, Z, marker='o')

ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_zlabel('Z values')
plt.title('3D Scatter chart')
plt.show()
```

![](https://laboputer.github.io/assets/img/ml/python/matplotlib/output_58_0.png)

---
이 포스팅에서 사용한 코드는 [이곳](https://github.com/Laboputer/LearnML/blob/master/02.%20%5BPOST%5D/52.%20%5BCode%5D%20Matplotlib%20Tutorial.ipynb)에 공개되어 있으며 다운로드 받으실 수 있습니다.