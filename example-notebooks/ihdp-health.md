---
description: 가정 방문과 특수 아동 발달 센터 방문이 조산아의 건강 및 발달에 얼마나 영향을 줄까?
---

# IHDP 데이터셋에 DoWhy 적용하기

* 작성자: 허현
* 원문: [DoWhy example on ihdp (Infant Health and Development Program) dataset](https://microsoft.github.io/dowhy/example\_notebooks/dowhy\_ihdp\_data\_example.html)

> 역자주 - 원문의 내용 자체가 IHDP 데이터셋에 대한 설명이나 인과분석 과정에 대한 설명이 생략되어 있습니다.\
> 해당 문서의 주요 목적은 여러 Propensity Score 방법론과 Refute 방법론을 사용하는 코드 레퍼런스를 남기는 것에 있기 때문에 관련 내용에 대한 학습이 필요한 경우 DOWHY KEY CONCEPTS의 [성향점수(Propensity Score)](https://playinpap.gitbook.io/dowhy/dowhy-key-concepts/propensity-score)와 [추정치를 검증하는 방법](https://playinpap.gitbook.io/dowhy/dowhy-key-concepts/sensitivity-analysis)를 참고하시면 좋습니다.

```python
# importing required libraries : 필요 라이브러리 불러오기
import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np
```

## 데이터 로드

```python
data= pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header = None)
col =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1" ,]
for i in range(1,26):
    col.append("x"+str(i))
data.columns = col
data = data.astype({"treatment":'bool'}, copy=False)
data.head()
```

|   | treatment | y\_factual | y\_cfactual | mu0      | mu1      | x1        | x2        | x3        | x4        | x5        | ... | x16 | x17 | x18 | x19 | x20 | x21 | x22 | x23 | x24 | x25 |
| - | --------- | ---------- | ----------- | -------- | -------- | --------- | --------- | --------- | --------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | True      | 5.599916   | 4.318780    | 3.268256 | 6.854457 | -0.528603 | -0.343455 | 1.128554  | 0.161703  | -0.316603 | ... | 1   | 1   | 1   | 1   | 0   | 0   | 0   | 0   | 0   | 0   |
| 1 | False     | 6.875856   | 7.856495    | 6.636059 | 7.562718 | -1.736945 | -1.802002 | 0.383828  | 2.244320  | -0.629189 | ... | 1   | 1   | 1   | 1   | 0   | 0   | 0   | 0   | 0   | 0   |
| 2 | False     | 2.996273   | 6.633952    | 1.570536 | 6.121617 | -0.807451 | -0.202946 | -0.360898 | -0.879606 | 0.808706  | ... | 1   | 0   | 1   | 1   | 0   | 0   | 0   | 0   | 0   | 0   |
| 3 | False     | 1.366206   | 5.697239    | 1.244738 | 5.889125 | 0.390083  | 0.596582  | -1.850350 | -0.879606 | -0.004017 | ... | 1   | 0   | 1   | 1   | 0   | 0   | 0   | 0   | 0   | 0   |
| 4 | False     | 1.963538   | 6.202582    | 1.685048 | 6.191994 | -1.045229 | -0.602710 | 0.011465  | 0.161703  | 0.683672  | ... | 1   | 1   | 1   | 1   | 0   | 0   | 0   | 0   | 0   | 0   |

5 rows × 30 columns

## 1. Model

```python
# Create a causal model from the data and given common causes. : 데이터와 공통 원인으로 인과 모델 생성
xs = ""
for i in range(1,26):
    xs += ("x"+str(i)+"+")

model=CausalModel(
        data = data,
        treatment='treatment',
        outcome='y_factual',
        common_causes=xs.split('+')
        )
```

## 2. Identify

```python
#Identify the causal effect : 인과 효과 규명하기
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)
```

```
Estimand type: nonparametric-ate

### Estimand : 1
Estimand name: backdoor1 (Default)
Estimand expression:
     d
────────────(Expectation(y_factual|x18,x17,x11,x2,x5,x14,x22,x23,x24,x21,x16,x20,x8,x4,x7,x19,x10,x15,x25,x9,x12,x3,x6,x1,x13))

Estimand assumption 1, Unconfoundedness: If U→{treatment} and U→y_factual then P(y_factual|treatment,x18,x17,x11,x2,x5,x14,x22,x23,x24,x21,x16,x20,x8,x4,x7,x19,x10,x15,x25,x9,x12,x3,x6,x1,x13,U) = P(y_factual|treatment,x18,x17,x11,x2,x5,x14,x22,x23,x24,x21,x16,x20,x8,x4,x7,x19,x10,x15,x25,x9,x12,x3,x6,x1,x13)
```

## 3. Estimate (using different methods)

### 3.1 Using Linear Regression

```python
# Estimate the causal effect and compare it with Average Treatment Effect : 인과 효과를 추정하고 ATE와 비교
estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression", test_significance=True
)

print(estimate)

print("Causal Estimate is " + str(estimate.value))
data_1 = data[data["treatment"]==1]
data_0 = data[data["treatment"]==0]

print("ATE", np.mean(data_1["y_factual"])- np.mean(data_0["y_factual"]))
```

```
*** Causal Estimate ***

## Identified estimand
Estimand type: nonparametric-ate

## Realized estimand
b: y_factual~treatment+x18+x17+x11+x2+x5+x14+x22+x23+x24+x21+x16+x20+x8+x4+x7+x19+x10+x15+x25+x9+x12+x3+x6+x1+x13
Target units: ate

## Estimate
Mean value: 3.928671750872714
p-value: [1.58915682e-156]

Causal Estimate is 3.928671750872714
ATE 4.021121012430829
```

### 3.2 Using Propensity Score Matching

```python
estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_matching"
)

print("Causal Estimate is " + str(estimate.value))

print("ATE", np.mean(data_1["y_factual"])- np.mean(data_0["y_factual"]))
```

```
Causal Estimate is 3.9791388232170393
ATE 4.021121012430829
```

### 3.3 Using Propensity Score Stratification

```python
estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_stratification", method_params={'num_strata':50, 'clipping_threshold':5}
)

print("Causal Estimate is " + str(estimate.value))
print("ATE", np.mean(data_1["y_factual"])- np.mean(data_0["y_factual"]))
```

```
Causal Estimate is 3.4550471588628207
ATE 4.021121012430829
```

### 3.4 Using Propensity Score Weighting

```python
estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_weighting"
)

print("Causal Estimate is " + str(estimate.value))

print("ATE", np.mean(data_1["y_factual"])- np.mean(data_0["y_factual"]))
```

```
Causal Estimate is 3.409737824407883
ATE 4.021121012430829
```

## 4. Refute

### 4.1 random\_common\_cause

```python
refute_results=model.refute_estimate(identified_estimand, estimate,
        method_name="random_common_cause")
print(refute_results)
```

```
Refute: Add a Random Common Cause
Estimated effect:3.409737824407883
New effect:3.4652727093798434
```

### 4.2 placebo\_treatment\_refuter

```python
res_placebo=model.refute_estimate(identified_estimand, estimate,
        method_name="placebo_treatment_refuter", placebo_type="permute")
print(res_placebo)
```

```
Refute: Use a Placebo Treatment
Estimated effect:3.409737824407883
New effect:-0.023837129277084368
p value:0.44999999999999996
```

### 4.3 Data Subset Refuter

```python
res_subset=model.refute_estimate(identified_estimand, estimate,
        method_name="data_subset_refuter", subset_fraction=0.9)
print(res_subset)
```

```
Refute: Use a subset of data
Estimated effect:3.409737824407883
New effect:3.3635040963705256
p value:0.29000000000000004
```
