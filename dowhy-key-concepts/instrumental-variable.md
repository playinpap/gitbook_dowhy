---
description: Instrumental Variable
---

# 도구변수 (Instrumental Variable)

* 작성자 : 경윤영

## 도구변수란?

설명변수를 통해서만 y에 영향을 미치는 변수 (설명변수가 통제될 때 도구변수의 변화는 y에 영향을 미치지 않는다.)

① 도구변수는 설명변수와 관련됨

② 도구변수는 외생적

## 도구변수를 사용할 때는?

설명변수가 내생성을 가질때 
$$E(U|X)\neq 0$$ 일 때 사용할 수 있다. 

즉, 내생성이란 설명변수와 오차가 서로 correlated 일때를 말하고, 

오차항과 상관된 설명변수들은 2가지 문제가 있다. 

1) OLS 추정량은 비일관적 

2) 표본의 크기가 아무리 커도 OLS 추정값은 참값과 다를 수 있음 

## 설명변수를 내생적으로 만드는 3가지

 

1. **변수누락** 

log(임금) = $$\beta_0+\beta_1$$학력+$$\beta_2$$경력+$$u$$

if omitted variable = 능력

학력, 능력은 서로 관련되어 있고 능력은 오차항의 일부를 구성하게됨 

→ 설명변수와 오차항이 서로 관련되는 문제 발생!!   

즉, 동일한 경력에서 임금차이가 발생하게 되고 이 것이 학력차이인지 능력차이인지 알 수 없게됨 

1. **동시성(Simultaneity)**

 설명변수와 종속변수가 동시에 결정될 때 

1. **설명변수의 측정오차** 

설명변수 측정시 오차가 존재할 때 

소비=$$\beta_0$$+$$\beta_1$$항상소득  

소비를 추정하려고 할 때 항상소득(현재부터 미래까지 자신에게 올 소득의 평균)은 관측이 힘들기 때문에 실제소득을 사용함  

**실제소득 = 항상소득 + 일시소득** 

(여기에서 일시소득은 항상소득과 무관하게 발생함)

소비=$$\beta_0+\beta_1$$실제소득 + $$(-\beta_1$$일시소득)

일시소득은 관측이 불가능하여 $$(-\beta_1$$일시소득)이 오차항이 됨, 이때 일시소득은 실제소득의 구성항목으로 오차항인 일시소득과 설명변수인 실제소득이 관련되어 내생성을 가지게 됨 

## 도구변수 수식으로 정리

### 1. just identified 일 때

$$y = \beta_0+\beta_1x_1+\beta_2x_2 +u$$

$$x_1$$와 $$x_2$$ 가 외생적이면 $$\beta_0$$, $$\beta_1$$, $$\beta_2$$는 $$E(u)=0$$, $$E(x_1u)=0$$, $$E(x_2u) =0$$ 에 대응하는 관계에 의해 정의가 가능하다.

$$E(u)=0\leftrightarrow E(y-\beta_0 -\beta_1x_1 - \beta_2x_2 ) = 0$$

$$E(x_1u)=0\leftrightarrow E[x_1(y-\beta_0 -\beta_1x_1 - \beta_2x_2 )] = 0$$

$$E(x_2u)=0\leftrightarrow E[x_2(y-\beta_0 -\beta_1x_1 - \beta_2x_2 )] = 0$$

위의 직교방정식은 3개이고 결정해야할 모수 $$\beta_0$$, $$\beta_1$$, $$\beta_2$$가 3개이므로  특이한 상황이 아닌 이상 세 모수들은 관측변수들의 분포(평균, 분산, 공분산 등)에 의해 식별된다(identified). 

아래와 같이 우리는 적률법(method of moments)를 사용하여 OLS 추정량을 구한다. 

*적률법: 모집단의 평균이 표본평균과 일치하는 모수를 찾는 방법  

$$E(y)=\beta_0+\beta_1E(x_1)+\beta_2E(x_2)$$

$$E(x_1y)=\beta_0E(x_1)+\beta_1E(x_1^2)+\beta_2E(x_1x_2)$$

$$E(x_2y)=\beta_0E(x_2)+\beta_1E(x_1x_2)+\beta_2E(x_2^2)$$


즉, $$E(y)$$를 $$n^{-1}\sum_{i=1}^{n}y_i$$ 로 추정하고 $$E(x_1y)$$를 $$ n^{-1}\sum_{i=1}^{n}x_{i1}y_i$$ 로 추정,

$$E(x_2y)$$를 $$ n^{-1}\sum_{i=1}^{n}x_{i2}y_i$$ 추정하는 등의 방식을 사용하여 모집단 상수를 구한 후 위의 등식에 대입한다면 $$\beta_0$$, $$\beta_1$$, $$\beta_2$$가 결정된다. (단, 비특이성을 만족시킨다: 3원 1차 연립방정식의 해가 유일할 조건이 있다.)

하지만, $$x_2$$가 내생적이라면 $$E(x_2, u) \neq 0$$ 이고

$$E(x_2u)=0\leftrightarrow E[x_2(y-\beta_0 -\beta_1x_1 - \beta_2x_2 )] \neq 0$$ 이게 된다.


추가 정보가 없다면 위의 식을 만족시키는 $$\beta_0$$, $$\beta_1$$, $$\beta_2$$ 는 무수히 많게 된다.

그렇기에 세 모수들을 식별하려면 별도의 식이 요구되고 이를 위해 추가적인 외생변수인 도구변수 $$(z_2)$$ 를 사용한다.

$$E(u)=0\leftrightarrow E(y-\beta_0 -\beta_1x_1 - \beta_2x_2 ) = 0$$

$$E(x_1u)=0\leftrightarrow E[x_1(y-\beta_0 -\beta_1x_1 - \beta_2x_2 )] = 0$$

즉, 기존의 2개의 식과 아래의 식이 추가되어 $$\beta_0$$, $$\beta_1$$, $$\beta_2$$ 를 식별할 수 있게 된다(just identified). 

$$E(z_2u)=0\leftrightarrow E[z_2(y-\beta_0 -\beta_1x_1 - \beta_2x_2 )] = 0$$

### 2. over-identified 일 때

모수들의 식별에 필요한 만큼보다 더 많은 제약식이 생길 때 over-identified 라고 한다. 

예를 들어 내생적 설명변수 1개인데 도구변수가 2개이상 일때 over-identified라고 한다!

도구변수 $$z_{2a}$$, $$z_{2b}$$가 있으면 아래의 두 개의 식이 추가된다. 

$$E[z_{2a}(y-\beta_0 -\beta_1x_1 - \beta_2x_2 )] = 0$$
$$E[z_{2b}(y-\beta_0 -\beta_1x_1 - \beta_2x_2 )] = 0$$

그렇다면 세 모수  $$\beta_0$$, $$\beta_1$$, $$\beta_2$$ 는 기존의 식들과 새롭게 생긴 위의 식들을 만족시켜야한다. 

$$E(u)=0\leftrightarrow E(y-\beta_0 -\beta_1x_1 - \beta_2x_2 ) = 0$$

$$E(x_1u)=0\leftrightarrow E[x_1(y-\beta_0 -\beta_1x_1 - \beta_2x_2 )] = 0$$

### 3. under identified 일 때

반대로 제약조건의 개수가 모수의 개수보다 작으면 모수들은 under-identified 된다. 

단순한 모형 $$y = \beta_2x_2+u$$ 일때 설명변수 $$x_2$$가 내생적이며 $$z_2$$가 도구변수라고 가정하자. 

이 때 $$E(z_2u)=0$$ 이면 $$\beta_2$$와 관계없이 항상 0이 성립하게 되어 $$\beta_2$$를 식별할 수 없게 된다. 

$$E(z_2u)=0\leftrightarrow E[z_{2}(y- \beta_2x_2 )] = 0$$

$$E(z_2y) = \beta_2E(z_2x_2) = 0$$

### 4. 2단계 최소제곱법 이용하기

2단계 최소제곱법은 회귀분석을 두 번 하는 것이다. 

$$y = \beta_0 + \beta_1x_1+\beta_2x_2 +u$$  

여기에서 $$x_1$$은 외생적이고 $$x_2$$는 내생적일 때, 추가 도구 변수는 $$z_{2a}$$ 이다. 

1단계)  <img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;\hat{x_2}&space;=&space;\hat{x_0}&space;&plus;&space;\hat{x_1}z_{2a}" title="\hat{x_2} = \hat{x_0} + \hat{x_1}z_{2a}" />

1단계에서는 내생적 설명변수인 <img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;x_2" title="x_2" />를 <img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;x_1" title="x_1" />과 <img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;z_2" title="z_2" />에 대해 회귀하여 맞춘값을 구한다. 

그리고 <img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;z_2" title="z_2" />의 유의성을 점검하는 것이 좋다. 

2단계)  <img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;y&space;=&space;\beta_0&space;&plus;&space;\beta_1x_1&space;&plus;&space;\beta_2\hat{x_2}&space;&plus;&space;u" title="y = \beta_0 + \beta_1x_1 + \beta_2\hat{x_2} + u" />

2단계에서는 y를 <img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;x_1" title="x_1" />과 <img src="https://latex.codecogs.com/gif.image?\dpi{110}&space;\hat{x_2}" title="\hat{x_2}" />에 대해 OLS 회귀를 시킨다.
