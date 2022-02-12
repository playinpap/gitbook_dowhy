---
description: Estimating the effect of a Member Rewards program
---

# 멤버십 리워드 프로그램의 효과 추정하기

* 작성자: [김가연](https://www.facebook.com/profile.php?id=1721702213)
* 원문: [Estimating the effect of a Member Rewards program](https://microsoft.github.io/dowhy/example_notebooks/dowhy_example_effect_of_memberrewards_program.html)

DoWhy 를 사용하여 고객에 대한 **구독 또는 리워드 프로그램의 효과**를 추정하는 방법을 알아보겠습니다.

웹사이트에 고객이 가입하면 추가 혜택을 받는 멤버쉽 리워드 프로그램이 있다고 가정해봅시다. 해당 프로그램이 효과적인지 어떻게 알 수 있을까요? 여기서 우리는 **인과적인 질문**을 할 수 있습니다. 

> 멤버쉽 리워드 프로그램 제공이 총 매출에 미치는 영향은 무엇인가?
> 

그리고 이와 동등한 반사실적(counterfactual) 질문은 다음과 같습니다.

> 만약 고객들이 멤버쉽 리워드 프로그램에 가입하지 않았다면, 그들은 웹사이트에 얼마나 더 적은 돈을 썼을 것인가?
> 

다시 말하면, 우리는 처치집단에 대한 평균 처치 효과(the Average Treatment Effect On the Treated, ATT)에 관심이 있습니다.

## I. Formulating the causal model

리워드 프로그램이 2019년 1월에 도입되었다고 가정해봅시다. 결과 변수는 연말 총 지출액입니다. 우리는 모든 유저의 모든 월별 거래 내역과 리워드 프로그램 가입을 선택한 유저들의 가입 시간에 대한 데이터를 가지고 있습니다. 데이터는 다음과 같습니다.

```python
# Creating some simulated data for our example example
import pandas as pd
import numpy as np
num_users = 10000
num_months = 12

signup_months = np.random.choice(np.arange(1, num_months), num_users) * np.random.randint(0,2, size=num_users)
df = pd.DataFrame({
    'user_id': np.repeat(np.arange(num_users), num_months),
    'signup_month': np.repeat(signup_months, num_months), # signup month == 0 means customer did not sign up
    'month': np.tile(np.arange(1, num_months+1), num_users), # months are from 1 to 12
    'spend': np.random.poisson(500, num_users*num_months) #np.random.beta(a=2, b=5, size=num_users * num_months)*1000 # centered at 500
})
# Assigning a treatment value based on the signup month
df["treatment"] = (1-(df["signup_month"]==0)).astype(bool)
# Simulating effect of month (monotonically increasing--customers buy the most in December)
df["spend"] = df["spend"] - df["month"]*10
# The treatment effect (simulating a simple treatment effect of 100)
after_signup = (df["signup_month"] < df["month"]) & (df["signup_month"] !=0)
df.loc[after_signup,"spend"] = df[after_signup]["spend"] + 100
df
```

|  | user_id | signup_month | month | spend | treatment |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 6 | 1 | 526 | True |
| 1 | 0 | 6 | 2 | 464 | True |
| 2 | 0 | 6 | 3 | 473 | True |
| 3 | 0 | 6 | 4 | 502 | True |
| 4 | 0 | 6 | 5 | 436 | True |
| ... | ... | ... | ... | ... | ... |
| 119995 | 9999 | 7 | 8 | 533 | True |
| 119996 | 9999 | 7 | 9 | 518 | True |
| 119997 | 9999 | 7 | 10 | 485 | True |
| 119998 | 9999 | 7 | 11 | 504 | True |
| 119999 | 9999 | 7 | 12 | 459 | True |

120000 rows × 5 columns

### The importance of time

이 문제를 모델링하는 데 있어서 **시간이 중요한 역할**을 합니다.

리워드 프로그램 가입은 향후 거래에 영향을 미칠 수 있지만, 이전 거래에는 영향을 미치지 않습니다. 사실 리워드 가입 이전의 거래는 리워드 가입 결정을 유발한다고 가정할 수 있습니다.

따라서 각 유저의 변수들을 다음과 같이 나눌 수 있습니다.

1. 처치 전 활동 (처치의 원인)
2. 처치 후 활동 (처치 적용 결과)

물론 가입과 총 지출에 영향을 미치는 많은 중요한 변수가 누락되어 있습니다(e.g., 구입한 제품 유형, 유저 계정 사용 기간, 지역 등). 관측되지 않은 교란 변수(`Unobserved Confounders`)를 나타내는 노드가 필요합니다. 

아래는 `i=3` 개월에 가입한 유저에 대한 인과 그래프입니다. 모든 `i`에 대해서 분석은 유사할 것입니다.

```python
import os, sys
sys.path.append(os.path.abspath("../../../"))
import dowhy

# Setting the signup month (for ease of analysis)
i = 3
```

```python
causal_graph = """digraph {
treatment[label="Program Signup in month i"];
pre_spends;
post_spends;
Z->treatment;
U[label="Unobserved Confounders"];
pre_spends -> treatment;
treatment->post_spends;
signup_month->post_spends; signup_month->pre_spends;
signup_month->treatment;
U->treatment; U->pre_spends; U->post_spends;
}"""

# Post-process the data based on the graph and the month of the treatment (signup)
df_i_signupmonth = df[df.signup_month.isin([0,i])].groupby(["user_id", "signup_month", "treatment"]).apply(
    lambda x: pd.Series({'pre_spends': np.sum(np.where(x.month < i, x.spend,0))/np.sum(np.where(x.month<i, 1,0)),
                        'post_spends': np.sum(np.where(x.month > i, x.spend,0))/np.sum(np.where(x.month>i, 1,0)) })
).reset_index()
print(df_i_signupmonth)
model = dowhy.CausalModel(data=df_i_signupmonth,
                     graph=causal_graph.replace("\n", " "),
                     treatment="treatment",
                     outcome="post_spends")
model.view_model()
from IPython.display import Image, display
display(Image(filename="causal_model.png"))
```

```
 user_id  signup_month  treatment  pre_spends  post_spends
0           0             6       True       480.2   511.000000
1           2             0      False       477.8   408.166667
2           5             0      False       472.6   423.333333
3           6             6       True       465.2   505.833333
4           8             0      False       447.6   396.333333
...       ...           ...        ...         ...          ...
5446     9993             0      False       488.8   415.166667
5447     9994             0      False       471.4   392.666667
5448     9995             0      False       489.4   402.666667
5449     9997             0      False       498.0   414.333333
5450     9998             0      False       469.6   400.000000

[5451 rows x 5 columns]
```

![i개월에 가입한 유저에 대한 인과 그래프](https://user-images.githubusercontent.com/76609403/153709477-d7e76bde-aaf9-47bb-bd7b-42955602ca55.png)

더 일반적으로, 고객에 대한 모든 활동 데이터를 위 그래프에 포함시킬 수 있습니다. 모든 사전 및 사후 활동 데이터는 이미 사용된 사전 및 사후 노드들과 동일한 위치 및 엣지를 차지합니다.

## **II. Identifying the causal effect**

관측되지 않은 교란 변수가 큰 역할을 하지 않는다고 가정해봅시다.

```python
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)
```

```
Estimand type: nonparametric-ate

### Estimand : 1
Estimand name: backdoor
Estimand expression:
     d
────────────(Expectation(post_spends|signup_month,pre_spends))
d[treatment]
Estimand assumption 1, Unconfoundedness: If U→{treatment} and U→post_spends then P(post_spends|treatment,signup_month,pre_spends,U) = P(post_spends|treatment,signup_month,pre_spends)

### Estimand : 2
Estimand name: iv
Estimand expression:
Expectation(Derivative(post_spends, [Z])*Derivative([treatment], [Z])**(-1))
Estimand assumption 1, As-if-random: If U→→post_spends then ¬(U →→{Z})
Estimand assumption 2, Exclusion: If we remove {Z}→{treatment}, then ¬({Z}→post_spends)

### Estimand : 3
Estimand name: frontdoor
No such variable found!
```

DoWhy 는 그래프를 바탕으로, 가입 월과 처지 이전 월(`signup_month`, `pre_spend`)에 소요된 금액을 조건화할 필요가 있다고 판단합니다.

## **III. Estimating the effect**

이제 backdoor estimand(추정치)를 기반으로 target_units 를 “att”로 설정하여 효과를 추정합니다.

```python
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor1.propensity_score_matching",
                                target_units="att")
print(estimate)
```

```
*** Causal Estimate ***

## Identified estimand
Estimand type: nonparametric-ate

## Realized estimand
b: post_spends~treatment+signup_month+pre_spends
Target units: att

## Estimate
Mean value: 115.21872571872572
```

위와 같이 평균 처치 효과를 알려줍니다. 즉, `i=3` 개월에 리워드 프로그램에 등록한 고객의 총 지출에 대한 평균 효과입니다. i 값을 변경한 후 분석을 다시 실행하여 다른 달에 가입한 고객에 대한 효과를 비슷한 방법으로 계산할 수 있습니다.

다만 좌측 및 우측 관측 중단으로 인해 효과 추정에 어려움을 겪는 경우가 존재합니다.

1. **좌측 관측 중단 (Left-censoring)**
    
    : 고객이 첫 달에 가입하는 경우, 우리는 가입하지 않은 고객과 비교할 만큼 충분한 거래 이력이 없습니다. 따라서 backdoor identified estimand 를 적용해야 합니다.
    
2. **우측 관측 중단 (Right-censoring)**
    
    : 고객이 마지막 달에 가입하는 경우, 가입 후 결과를 추정하기에는 향후(사후) 거래 이력이 부족합니다.
    

따라서 아무리 가입 효과가 모든 달에 걸쳐 동일했더라도, 데이터가 부족한 상황에서는 추정된 pre-treatment 또는 post-treatment 거래 활동의 변동이 클 수 있기 때문에 가입 월별로 추정 효과가 다를 수 있습니다.

## **IV. Refuting the estimate**

Placebo treatment refuter 를 사용하여 추정치를 반박합니다. 이 refuter 는 treatment 를 독립적인 랜덤 변수로 대체하고 추정치가 0이 되는지 여부를 확인합니다. (0이 되어야 합니다!)


```python
refutation = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter",
                     placebo_type="permute", num_simulations=2)
print(refutation)
```

```
Refute: Use a Placebo Treatment
Estimated effect:115.21872571872572
New effect:1.3245920745920756
p value:0.015071603412401298
```
