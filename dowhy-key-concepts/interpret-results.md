# DoWhy 모델 결과 해석하기

- 작성자: [최보경](https://www.facebook.com/pagebokyung/)

## DoWhy에서 Estimate Effect란?

![](<../.gitbook/assets/dowhy-diagram.png>)  

DoWhy는 인과추론을 위한 메커니즘을 4단계로 구성했습니다. 1단계 (Model)에서는 데이터를 인과 그래프로 인코딩하고, 2단계 (Identify)에서는 모델의 인과 관계를 식별하고 원인을 추정합니다. 3단계 (Estimate)에서는 식별된 인과관계에 대해 추정치를 구하고 4단계 (Refute)에서는 얻어진  추정치에 대해 반박을 시도합니다.

```python
model = CausalModel(data, graph, treatment, outcome)
estimand = model.identify_effect()
estimate = model.estimate_effect(estimand, method_name="propensity_score_weighting")
refute = model.refute_estimate(estimand, estimate, method_name="placebo_treatment_refuter")
```

인과 효과를 추정할 때, 효과의 방향성, 도구 변수 또는 매개자의 존재, 모든 관련 교란 요인이 관찰되는지 여부 등의 중요한 가정을 기반으로 이루어집니다. 이 가정을 위반하게 되면, 인과 효과 추정치에 유의한 오류가 발생합니다. 머신 러닝을 활용한 예측 모델 같은 경우, 교차 검증 방식이 존재하지만 인과추론에 대한 전역 검증 방식은 현재 존재하지 않습니다.

따라서, 인과적 가정을 최대한 공식적으로 표현하고 검증하는 것이 중요합니다. 이를 위해 DoWhy는 1단계에서 인과 모형을 명시적으로 선언할 수 있도록 제공합니다. 또한 가정의 부분 집합들을 확인하기 위해 여러 검증 방식을 제공합니다. 평균 인과 효과 및 조건부 인과 효과 등 오류를 더 잘 감지할 수 있는 검증 테스트가 개발되어 4단계에서 다뤄지고 있습니다. ([Sharma, 2021](https://arxiv.org/pdf/2108.13518.pdf))

이 장에서는 DoWhy에서 결과를 확인하게 되는 3단계인 인과 효과 추정 단계에 대해서 전체적인 그림을 이해하고, 함수의 출력 결과를 어떻게 해석해서 실무에 전달할 수 있을지 안내하고자 합니다.

---

## 인과 효과 추정을 할 수 있는 방식들

### 1. **어떤 메소드들이 있을까?**

```python
# Estimate the causal effect and compare it with Average Treatment Effect
estimate = model.estimate_effect(identified_estimand,
                                 method_name = "backdoor.linear_regression", 
                                 test_significance = True, # Linear regression 에 적합
                                 confidence_intervals = True # Linear regression 에 적합
                                )
```

estimate_effect 함수에 들어가는 `method_name` 입력값으로는 아래와 같은 종류의 방식들이 있습니다.

- Propensity Score Matching: “backdoor.propensity_score_matching”
- Propensity Score Stratification: “backdoor.propensity_score_stratification”
- Propensity Score-based Inverse Weighting: “backdoor.propensity_score_weighting”
- Linear Regression: “backdoor.linear_regression”
- Generalized Linear Models (e.g., logistic regression): “backdoor.generalized_linear_model”
- Instrumental Variables: “iv.instrumental_variable”
- Regression Discontinuity: “iv.regression_discontinuity”

### 2. estimate_effect **함수를 사용하기 위한 조건**

1. 하나의 그래프 `graph` 입력이 필요합니다.
2. `common_causes` 또는 `instruments` 가 입력이 되어야 합니다.

---

## 함수 출력 이해하기

```python
## Identified estimand
### Estimand 
## Realized estimand
## Estimate
```

estimate_effect 함수의 출력은 4가지로 구성됩니다. 어떤 메소드를 선택하느냐에 따라 세부적인 항목은 바뀌지만 4가지 프레임은 동일하게 유지됩니다. 하나씩 해석해보겠습니다. ([지명진, 2021](https://www.koreascience.or.kr/article/CFKO202125036032267.pdf)) 

### **1. Identified estimand**

```python
## Identified estimand
Estimand type: nonparametric-ate
```

- **Identifed estimand** : 어떤 Estimand를 추론하는지 그 종류를 정의하고 추정치를 식별한 결과
- **Estimand** : 통계 분석에서 추론의 대상이 되는 추정값 (e.g. 가우시안 분포를 추정하고 싶으면 평균과 분산 2가지 Estimand를 추론)
- **Estimand type** : 어떤 Estimand를 추론하는지 그 종류. nonparametric-ate, nonparametric-nde, nonparametric-nie로 구성됩니다. ([github](https://github.com/microsoft/dowhy/blob/95be0350818db3233051f1bbb849b6c3925e2e0b/dowhy/causal_identifier.py#L22))
    - **nonparametric-ate** : 인과추론에서의 구조적 가정 (Ignorability, Parametric Assumption) 중 Parametric Assumption을 만족시키지 않고도 인과 효과를 추정할 수 있는 방식. Propensity score 및 매칭 기법을 통한 접근, 반응 표면에서 선형적인 모델링이 아닌 유연한 모델링 접근일 때 가능합니다.
        
        현재 DoWhy 패키지에서는 현실 문제에 사용하기 비교적 쉬운 스탠다드 인과추론 방식인 nonparametric만 제공하고 있습니다. 향후에 더 다양한 parametric form 의 identification 이 추가될 예정이라고 합니다. (주석)
        
        - **Ignorability Assumption** : 가능한 모든 교란 변수를 통제했을 때 잠재적 결과가 처치에 독립일 것이다. 즉, 모든 교란 변수가 통제되었을 것이다. ([이은지님 블로그, 2020](https://assaeunji.github.io/bayesian/2020-04-10-causal/#strong-ignorability-assumption))
            
            ![](<../.gitbook/assets/interpret-img1.png>)
            
        - **Parametric Assumption** : 강한 Ignorability Assumption 아래에서 ATE, ATT와 같은 대표적인 인과적 Estimand를 추정하는 것은, E[Y (1) | X ] 와 E[Y(0) | X] 를 추정하는 것을 필요로 합니다. 이 때, 두 E[Y (1) | X ] 와 E[Y(0) | X] 가 Linear Regression 으로 피팅될 것을 가정으로 합니다. 이는 X 자체가 고차원이거나, Common Support Assumption 을 만족시키지 못할 수 있기 때문에 현실에서 이 가정이 만족되기 어렵습니다. ([J Hill of NYU, 2015](https://cepim.northwestern.edu/calendar-events/2015-09-29))
            
            ![](<../.gitbook/assets/interpret-img2.png>)
            
            > Causal inference is important but hard because it requires strong assumptions (ignorability, parametric assumptions).
            
            Standard causal inference tools (e.g. most matching techniques) do not have these properties. ([J Hill of NYU, 2015](https://cepim.northwestern.edu/calendar-events/2015-09-29))
            > 
            > 
            > [Slides-2015-09-29 Hill.pdf](DoWhy%20Key%20Concept%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%20%E1%84%92%E1%85%A2%E1%84%89%E1%85%A5%E1%86%A8%E1%84%92%E1%85%A1%E1%84%80%E1%85%B5%20f1bace74b64b49fab834081d8ac2cb4e/Slides-2015-09-29_Hill.pdf)
            > 
            - **Common Support Assumption ( = Positivity / Overlap Assumption)**: 처치가 주어지거나, 주어지지 않을 확률이 0 초과 1 미만일 것. 성향점수매칭이 유효할 수 있는 2가지 조건 중 하나입니다. (나머지 하나는 조건부 독립) 만족시키지 못할 경우, 매칭 기법에서 해결될 수 없는 불균형 / inverse-probability-of-treatment weighting (IPTW) 기법에서 불안정한 가중치를 야기합니다. [(J Hill, 2013](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-7/issue-3/Assessing-lack-of-common-support-in-causal-inference-using-Bayesian/10.1214/13-AOAS630.pdf))
                
                ![[F Baum, 2013](http://fmwww.bc.edu/EC-C/S2013/823/EC823.S2013.nn12.slides.pdf)](<../.gitbook/assets/interpret-img3.png>)
                
                ([F Baum, 2013](http://fmwww.bc.edu/EC-C/S2013/823/EC823.S2013.nn12.slides.pdf)) 
               
                

3단계의 출력값 중 **Estimand type**을 이해하기 위해서는 2단계의 인과 관계 식별 방법에 대해서 조금 더 알아보는 것이 좋은데요. 1단계 (Model)에서 입력된 인과 그래프를 기반으로 가능한 모든 식별 방법을 2단계 (Identify)에서 찾게 됩니다. 이 때, 2단계에서 찾는 가능한 모든 **식별 방법**은 총 3가지로 크게 분류가 됩니다 : `Back-door / Front-door / Instrumental-Variables` 



- 2단계 (Identify) 출력 예시
    
    ```python
    Estimand type: nonparametric-ate
    
    ### Estimand : 1
    Estimand name: backdoor1 (Default)
    Estimand expression:
         d────────────(Expectation(y_factual|x6,x18,x9,x21,x2,x23,x25,x17,x3,x20,x16,x14
    d[treatment],x12,x10,x13,x22,x24,x7,x1,x11,x4,x15,x8,x5,x19))
    Estimand assumption 1, Unconfoundedness: If U→{treatment} and U→y_factual then P(y_factual|treatment,x6,x18,x9,x21,x2,x23,x25,x17,x3,x20,x16,x14,x12,x10,x13,x22,x24,x7,x1,x11,x4,x15,x8,x5,x19,U) = P(y_factual|treatment,x6,x18,x9,x21,x2,x23,x25,x17,x3,x20,x16,x14,x12,x10,x13,x22,x24,x7,x1,x11,x4,x15,x8,x5,x19)
    
    ### Estimand : 2
    Estimand name: iv
    No such variable found!
    
    ### Estimand : 3
    Estimand name: frontdoor
    No such variable found!
    ```
    
    - **Estimand** : 통계 분석에서 추론의 대상이 되는 추정값
        - **Estimand 1 (Back-door)** : 원인 변수 X 가 결과 변수 Y에 가지는 인과 관계를 식별하기 위해서, X 와 Y 에 영향을 주는 측정이 가능한 일반적인 원인들이 있을 경우, 그 원인 들을 Conditioning 하는 방식
        - **Estimand 2 (Front-door)** : Two-stage Linear Regression 을 통해서 인과 관계를 식별하는 방식. 이 때 헤크만 2단계 추정(Heckman Two-Stage Selection Model)을 활용합니다. 2SLS와의 차이에 대해 이해를 돕기 위해 [이 링크](https://stats.stackexchange.com/questions/172508/two-stage-models-difference-between-heckman-models-to-deal-with-sample-selecti)를 첨부합니다.
        - **Estimand 3 (Instrumental-Variables**) : 도구변수를 통해 원인과 결과가 관찰되지 않는 경우에도 결과를 예측하도록 한 방식. 도구 변수는 2가지 ([Exclusion Restriction](https://medium.com/bondata/instrumental-variable-1-6c249de6ea34) / Randomness)에 대해서 가정을 하여 진행됩니다.

일반적으로 Estimand 1 (Back-door)이 기본값이며 이를 사용해서 추정하게 됩니다.

### 2. **Estimand**

```python
### Estimand : 1
Estimand name: iv
Estimand expression:
Expectation(Derivative(y, [Z0, Z1])*Derivative([v0], [Z0, Z1])**(-1))
Estimand assumption 1, As-if-random: If U→→y then ¬(U →→{Z0,Z1})
Estimand assumption 2, Exclusion: If we remove {Z0,Z1}→{v0}, then ¬({Z0,Z1}→y)
```

- **Estimand name** : 2단계에서 어떤 식별 방법을 선택했는지 출력
- **Estimand expression** : Identified Estimand 의 확률적 표현으로, 2단계에서 선택한 식별 방법의 Estimand expression을 출력
- **Estimand assumption** : 추정값을 식별하는 과정에서 사용된 가정들을 명시. Estimand type 에 따라 정해진 가정이 존재합니다.
    - Back-door 일 때 ([github](https://github.com/microsoft/dowhy/blob/95be0350818db3233051f1bbb849b6c3925e2e0b/dowhy/causal_identifier.py#L460))
        
        ```python
        'Unconfoundedness': (
                        u"If U\N{RIGHTWARDS ARROW}{{{0}}} and U\N{RIGHTWARDS ARROW}{1}"
                        " then P({1}|{0},{2},U) = P({1}|{0},{2})"
                    ).format(",".join(treatment_name), outcome_name, ",".join(common_causes))
        ```
        
    - IV 일 때 ([github](https://github.com/microsoft/dowhy/blob/95be0350818db3233051f1bbb849b6c3925e2e0b/dowhy/causal_identifier.py#L486))
        
        ```python
        "As-if-random": (
                        "If U\N{RIGHTWARDS ARROW}\N{RIGHTWARDS ARROW}{0} then "
                        "\N{NOT SIGN}(U \N{RIGHTWARDS ARROW}\N{RIGHTWARDS ARROW}{{{1}}})"
                    ).format(outcome_name, ",".join(instrument_names)),
                    "Exclusion": (
                        u"If we remove {{{0}}}\N{RIGHTWARDS ARROW}{{{1}}}, then "
                        u"\N{NOT SIGN}({{{0}}}\N{RIGHTWARDS ARROW}{2})"
                    ).format(",".join(instrument_names), ",".join(treatment_name),
                             outcome_name)
        ```
        
    - Front-door 일 때 ([github](https://github.com/microsoft/dowhy/blob/95be0350818db3233051f1bbb849b6c3925e2e0b/dowhy/causal_identifier.py#L517))
        
        ```python
                   "Full-mediation": (
                        "{2} intercepts (blocks) all directed paths from {0} to {1}."
                    ).format(",".join(treatment_name), ",".join(outcome_name), ",".join(frontdoor_variables_names)),
                    "First-stage-unconfoundedness": (
                        u"If U\N{RIGHTWARDS ARROW}{{{0}}} and U\N{RIGHTWARDS ARROW}{{{1}}}"
                        " then P({1}|{0},U) = P({1}|{0})"
                    ).format(",".join(treatment_name), ",".join(frontdoor_variables_names)),
                    "Second-stage-unconfoundedness": (
                        u"If U\N{RIGHTWARDS ARROW}{{{2}}} and U\N{RIGHTWARDS ARROW}{1}"
                        " then P({1}|{2}, {0}, U) = P({1}|{2}, {0})"
                    ).format(",".join(treatment_name), outcome_name, ",".join(frontdoor_variables_names))
        ```
        

### 3. **Realized Estimand**

```python
## Realized estimand
b: y~v0+W3+W4+W2+Z0+X0+W0+Z1+W1
Target units: ate

## Realized estimand
b: is_canceled~different_room_assigned+previous_bookings_not_canceled+hotel+total_of_special_requests+market_segment+is_repeated_guest+guests+lead_time+meal+days_in_waiting_list+country+booking_changes+total_stay+required_car_parking_spaces
Target units: ate

## Realized estimand
Realized estimand: Wald Estimator
Realized estimand type: nonparametric-ate
Estimand expression:
                                                              -1
Expectation(Derivative(y, Z0))⋅Expectation(Derivative(v0, Z0))
Estimand assumption 1, As-if-random: If U→→y then ¬(U →→{Z0,Z1})
Estimand assumption 2, Exclusion: If we remove {Z0,Z1}→{v0}, then ¬({Z0,Z1}→y)
Estimand assumption 3, treatment_effect_homogeneity: Each unit's treatment ['v0'] is affected in the same way by common causes of ['v0'] and y
Estimand assumption 4, outcome_effect_homogeneity: Each unit's outcome y is affected in the same way by common causes of ['v0'] and y

Target units: ate
```

- **Realized Estimand** : 실제로 실현된 추정. 2단계 (Identify)에서 다양한 식별 방법이 가능함을 모델이 알려주지만, 그 중에서 사용자가 직접 선택한 Estimand type 으로 필터링하여 업데이트합니다. ([github](https://github.com/microsoft/dowhy/blob/95be0350818db3233051f1bbb849b6c3925e2e0b/dowhy/causal_estimator.py#L793))
    
    위 첫번째 예시의 경우 v0, w3, ~ w1 의 변수를 통해서 y를 예측했다는 것으로 보입니다. 세번째 예시는 도구변수를 활용했을 경우의 출력인데, Wald Estimator 란 아래와 같이 정의됩니다.
    
    ![](<../.gitbook/assets/interpret-img4.png>)
    
- **Target Units** : ate, att, atc 중 어떤 단위로 분석된 결과인지 출력. Lambda 함수를 통해서 직접 정의할 수도 있습니다.
    - ate : 평균 처치 효과 (`Average Treatment Effect`)
    - att : 실험군 대상의 평균 처치 효과 (`Average Treatment Effect on Treated`)
    - atc : 대조군 대상의 평균 처치 효과 (`Average Treatment Effect on Control`)

### 4. **Estimate**

```python
## Estimate
Mean value: 10.503496778665827

Causal Estimate is 10.503496778665827

## Estimate
Mean value: 3.9286717508727174
p-value: [1.58915682e-156]
95.0% confidence interval: [[3.70717741 4.15016609]]
```

마지막 단계는 Realized Estimand 에 대해서 실제로 추정한 값 (Estimate)을 계산하여 출력합니다. Linear Regression 메소드를 사용해서 추정할 경우, 95.0% confidence interval 을 추가로 출력할 수 있습니다.

- **Mean value** : 우선, 처치가 가질 수 있는 값이 0,1이라는 가정을 기반으로 합니다. 하나의 관측 데이터에 대해서 Y(1), Y(0)는 가질 수 없기 때문에 개별 수준의 처치 효과를 추정하기 어렵습니다.  따라서 개별 효과들의 평균을 구하여 인과 효과를 계산합니다.
    - τATE=E[Y(1)−Y(0)]=E[Y(1)]−E[Y(0)]
    - τATT=E[Y(1)−Y(0)|Z=1] (Z : 처치 받은 여부)
    - τATC=E[Y(1)−Y(0)|Z=0]
    
    아래 함수를 보시면 처치 받은 여부에 따라 데이터 프레임을 분리한 후, 데이터 프레임의 결과값 컬럼을 np.mean 으로 평균낸 후 두 평균값의 차이를 통해 인과 효과를 계산합니다. ([github](https://github.com/microsoft/dowhy/blob/95be0350818db3233051f1bbb849b6c3925e2e0b/dowhy/causal_estimator.py#L186))
    
    ```python
    def estimate_effect_naive(self):
            # TODO Only works for binary treatment
            df_withtreatment = self._data.loc[self._data[self._treatment_name] == 1]
            df_notreatment = self._data.loc[self._data[self._treatment_name] == 0]
            est = np.mean(df_withtreatment[self._outcome_name]) - np.mean(df_notreatment[self._outcome_name])
            return CausalEstimate(est, None, None, control_value=0, treatment_value=1)
    ```
    
- **p-value** : 인과 효과 추정치가 같다는 가정 하에, 무작위 표본 추출 오류로 인해 1000개의 표본 중 __% 에서 관찰된 수준 이상의 차이가 도출됩니다. 모든 Estimator 에서 `test_significance="bootstrap"` 입력을 통해서 p-value 를 계산할 수 있습니다. 기본적으로 Two-sided test 를 진행하며 부트스트랩을 몇 회 진행할 지는 num_null_simulations 입력을 받습니다. 초기값은 1000입니다.
    
    > 1. Dummy outcome 을 np.random.permutation 을 통해서 생성 
    2. 기존 데이터에 dataframe.assign 를 통해서 Dummy outcome 을 할당 
    3. 학습된 Estimator 에 2에서 만들어진 데이터 셋을 넣고 인과 효과를 추정 
    4. 1000개의 인과 효과 추정치가 만들어짐 
    5. 1000개의 인과 효과 추정치의 중앙값을 계산한 후, 실제 인과 효과 추정치와 비교함 ([github](https://github.com/microsoft/dowhy/blob/95be0350818db3233051f1bbb849b6c3925e2e0b/dowhy/causal_estimator.py#L471))
    > 
    
- **95.0% confidence interval** : 인과 효과 추정치의 95% 신뢰 구간. 인과 효과 추정에서 사용된 Estimator 의 특정한 방식이 없다면 부트스트랩을 통해서 계산됩니다. 부트스트랩을 어떻게 진행할지는 (num_simulations, sample_size_fraction) 2가지 입력을 받습니다. 두 값의 초기값은 (100, 1) 입니다.

## (예시) 함수 출력 결과 해석해서 전달하기

- **문제** 명시 : 멤버십 프로그램을 가입한 여부가 구매액 증가에 영향을 줬을까?
- **인과 모형** 에 대한 설명 :
    
    멤버십 프로그램을 가입한 여부 → 구매액 증가 사이의 관계에 영향을 주는 외부 요인들이 많으며, 정성적으로 고려할 수 있는 주요한 외부 요인은 가입 월, 멤버십 프로그램 가입 전의 구매액이었습니다. 
    
    다만 정성적으로 고려할 수 없는 / 측정할 수 없는 외부 요인도 존재하여 Unobserved Confounders 라는 이름으로 반영이 되었습니다.
    
    ![Causal Graph](<../.gitbook/assets/interpret-img5.png>)
    
    
- **결과** 요약 :
    - 멤버십 보상 프로그램이 적용되었던 집단에 한정하여, 멤버십 보상 프로그램에 한 명의 사용자가 더 가입할수록, 평균 $102.40의 구매액 증가가 있었습니다.
    - 다만, 이 결과는 몇 가지 가정을 기반으로 합니다 :
        - 출력 결과에서 Assumption 에 대해서 이해할 수 있게 풀어서 표기함.
    - 이 결과는 __, __, __ 의 검증 테스트를 통과하여 추정치의 강건성을 확인했습니다. 
    - 참고용 실제 출력
        
        ```python
        *** Causal Estimate ***
        
        ## Identified estimand
        Estimand type: nonparametric-ate
        
        ### Estimand : 1
        Estimand name: backdoor1 (Default)
        Estimand expression:
             d                                                        
        ────────────(Expectation(post_spends|signup_month,pre_spends))
        d[treatment]                                                  
        Estimand assumption 1, Unconfoundedness: If U→{treatment} and U→post_spends then P(post_spends|treatment,signup_month,pre_spends,U) = P(post_spends|treatment,signup_month,pre_spends)
        
        ## Realized estimand
        b: post_spends~treatment+signup_month+pre_spends
        Target units: att
        
        ## Estimate
        Mean value: 102.40033381020511
        ```
        



