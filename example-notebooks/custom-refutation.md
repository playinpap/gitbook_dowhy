---
description: >-
  A Simple Example on Creating a Custom Refutation Using User-Defined Outcome
  Functions
---

# 사용자 정의 결과 함수를 사용해 추정치 반박하기

*작성자: 경윤영

*원문: [Dowhy -Simple Example on Creating a Custom Refutation Using User-Defined Outcome Functions](https://microsoft.github.io/dowhy/example_notebooks/dowhy_demo_dummy_outcome_refuter.html) 

본 글은 Refute 방법론을 사용하는 코드 레퍼런스를 남기는 목적으로 작성되었습니다. 그렇기에 Refute에 대한 학습이 필요하신 경우 DOWHY KEY CONCEPTS의 [추정치를 검증하는 방법](https://playinpap.gitbook.io/dowhy/dowhy-key-concepts/sensitivity-analysis) 을 참고하시길 바랍니다. 

본 실험에서는 선형데이터(linear dataset)를 구축하였고, 선형 회귀를 estimator로 사용합니다. 

Refute의 방법으로는  outcome을 대체하는 dummy outcome refuter 방법을 사용하여 진행합니다. 

## 1. Insert Dependencies

```python
from dowhy import CausalModel
import dowhy.datasets
import pandas as pd 
import numpy as np 
# Config dict to set the logging level
import logging.config
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers':False,
    'loggers': {
        '': {
            'level': 'WARN',
        },
    }
}

logging.config.dictConfig(DEFAULT_LOGGING)
```

## 2. Create the dataset

Hyper parameter의 값을 바꾸며 effect의 변화를 확인할 수 있습니다. 

| Variable Name | Data Type | Interpretation |
| --- | --- | --- |
| Zi | float | 도구변수(Instrument Variable) |
| Wi | float | 교란변수(Confounder) |
| V0 | float | 처치변수(Treatment) |
| Y | float | 결과변수(Outcome) |

```python
# Value of the coefficient [BETA]
BETA = 10
# Number of Common Causes
NUM_COMMON_CAUSES = 2
# Number of Instruments
NUM_INSTRUMENTS = 1
# Number of Samples
NUM_SAMPLES = 100000
# Treatment is Binary
TREATMENT_IS_BINARY =False
data = dowhy.datasets.linear_dataset(beta=BETA,
                                 num_common_causes=NUM_COMMON_CAUSES,
                                 num_instruments=NUM_INSTRUMENTS,
                                 num_samples=NUM_SAMPLES,
                                 treatment_is_binary=TREATMENT_IS_BINARY)
data['df'].head()
```

|  | Z0 | W0 | W1 | V0 | y |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.0 | 0.112689 | -0.501474 | 8.076574 | 80.106461 |
| 1 | 0.0 | 0.645347 | -0.072829 | -0.219279 | -0.092377 |
| 2 | 0.0 | 0.323480 | 0.989825 | 0.365947 | 6.900517 |
| 3 | 0.0 | 0.030437 | 1.334423 | 1.740524 | 20.319910 |
| 4 | 1.0 | 1.377841 | 0.628397 | 11.938058 | 125.523936 |

## 3. Creating the Causal Model

```python
model = CausalModel(
    data = data['df'],
    treatment = data['treatment_name'],
    outcome = data['outcome_name'],
    graph = data['gml_graph'],
    instruments = data['instrument_names']
)

model.view_model()
```

아래의 그림은 처치변수(treatment), 결과변수(outcome), 교란변수(confounders), 도구변수(instrument variable)의 관계를 살펴볼 수 있습니다. 

W0, W1 : 교란변수(confounders)

Z0 : 도구변수(instrument variable)

V0: 처치변수(treatment)

Y: 결과변수(outcome)Y

![사용자 정의 함수 이미지1](https://user-images.githubusercontent.com/39981604/153433647-39b2fd58-d7f1-485c-9f5e-39e1aa8e8899.png)

## 4. Identify the Estimand

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
─────(Expectation(y|W1,W0))
d[v₀]
Estimand assumption 1, Unconfoundedness: If U→{v0} and U→y then P(y|v0,W1,W0,U) = P(y|v0,W1,W0)

### Estimand : 2
Estimand name: iv
Estimand expression:
Expectation(Derivative(y, [Z0])*Derivative([v0], [Z0])**(-1))
Estimand assumption 1, As-if-random: If U→→y then ¬(U →→{Z0})
Estimand assumption 2, Exclusion: If we remove {Z0}→{v0}, then ¬({Z0}→y)

### Estimand : 3
Estimand name: frontdoor
No such variable found!
```

## 5. Estimating the Effect

```python
causal_estimate = model.estimate_effect( identified_estimand,
                                       method_name="iv.instrumental_variable",
                                       method_params={'iv_instrument_name':'Z0'}
                                       )
print(causal_estimate)
```

```
*** Causal Estimate ***

## Identified estimand
Estimand type: nonparametric-ate

### Estimand : 1
Estimand name: iv
Estimand expression:
Expectation(Derivative(y, [Z0])*Derivative([v0], [Z0])**(-1))
Estimand assumption 1, As-if-random: If U→→y then ¬(U →→{Z0})
Estimand assumption 2, Exclusion: If we remove {Z0}→{v0}, then ¬({Z0}→y)

## Realized estimand
Realized estimand: Wald Estimator
Realized estimand type: nonparametric-ate
Estimand expression:
                                                              -1
Expectation(Derivative(y, Z0))⋅Expectation(Derivative(v0, Z0))
Estimand assumption 1, As-if-random: If U→→y then ¬(U →→{Z0})
Estimand assumption 2, Exclusion: If we remove {Z0}→{v0}, then ¬({Z0}→y)
Estimand assumption 3, treatment_effect_homogeneity: Each unit's treatment ['v0'] is affected in the same way by common causes of ['v0'] and y
Estimand assumption 4, outcome_effect_homogeneity: Each unit's outcome y is affected in the same way by common causes of ['v0'] and y

Target units: ate

## Estimate
Mean value: 9.99706705820163
```

## 6. Refuting the Estimate

### 1) Using a Randomly Generated Outcome

```python
ref = model.refute_estimate(identified_estimand,
                           causal_estimate,
                           method_name="dummy_outcome_refuter"
                           )
print(ref[0])
```

```
Refute: Use a Dummy Outcome
Estimated effect:0
New effect:4.657654751543888e-05
p value:0.49

```

### 2) Using a Function that Generates the Outcome from the Confounders

The basic expression is of the form 

$$y_{new} = \beta_0W_0 + \beta_1W_1 + \gamma_0$$

where, $$\beta_0 = 1, \beta_1 = 2, \gamma_0 = 3$$

```python
coefficients = np.array([1,2])
bias = 3
def linear_gen(df):
    y_new = np.dot(df[['W0','W1']].values,coefficients) + 3
return y_new
```

```python
ref = model.refute_estimate(identified_estimand,
                           causal_estimate,
                           method_name="dummy_outcome_refuter",
                           outcome_function=linear_gen
                           )

print(ref[0])
```

```
Refute: Use a Dummy Outcome
Estimated effect:0
New effect:-1.1692081553648758e-05
p value:0.47

```
