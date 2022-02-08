---
description: Targeting Optimization Bidder at Uber
---

# Uplift Modeling을 활용한 광고 입찰 최적화 - Uber

* 작성자: [최보경](https://www.facebook.com/pagebokyung/)
* 원문: [KDD2021 자료](https://drive.google.com/file/d/1QJJUCo4LH5kGQP3kaJlG1RdhjhaJWp-5/view)와 [Colab 노트북](https://colab.research.google.com/drive/1fnZEHIAcNxrvSxFrlO1hRTHO7sazXbo0?usp=sharing)

Online Real Time Bidding에서 Uplift Modeling을 활용하여, HTE를 추정한 후 최적의 유저를 선택하는 방법론을 소개합니다.실시간 입찰에서 Uplift Modeling의 효과를 조사하기 위해 실제 캠페인 데이터에 대한 4개의 Meta-Learner 비교 분석을 실시했습니다. Offline Evaluation 및 Online Evaluation를 실시하고, 평가를 위한 근거 자료로 TML(Target Maximum likely Estimation) 기반 평균 처치 효과(ATE)를 사용하는 방법을 소개합니다.

## Background

우버가 **광고주(Demand-side)로서 어떤 광고를 어떤 퍼블리셔(Supply-side)에 내보내야** 주어진 예산 하에서 최적의 Performance를 낼 수 있을까?의 문제입니다.

퍼블리셔에 소속된 개별 유저가 Supply-side Platform에 광고 요청을 보내면, Uber Bidder 시스템이 4가지 단계를 통해서 관심이 있을 만한 유저에게 광고를 보여줍니다. 이를 통해 유저는 우버에서 더 많은 라이드를 타게 됩니다.

이 과정에서 핵심적인 역할을 하는 플랫폼을 ‘Uber Bidding Platform’이라고 부르며, 플랫폼의 중요한 컴포넌트는 3가지입니다.

1. Dynamic Bidding Strategies : 얼마나 실시간으로 빠르게 입찰을 진행하는가?
2. ML models
3. Incrementality Measurement : 얼마나 HTE를 추정이 잘 되었는가?

![](../.gitbook/assets/uber\_opt1.png)\
![POI : point-of-interest](../.gitbook/assets/uber\_opt2.png)

여기서의 Incrementality Measurement 에서는 Uplift Modeling 이 핵심입니다. 기본 컨셉에 대해 설명하면, **이론적으로 광고주에게는 ‘Persuadable’ 세그먼트에 해당하는 유저를 인과추론과 머신러닝을 통해서 찾는 것**에 관심이 있습니다. 투자한 금액의 효용을 극대화할 수 있기 때문입니다.

![](../.gitbook/assets/uber\_opt3.png)\
![설득 가능한 세그먼트 (Persuadables)는 타겟이 되고, 청개구리 (Sleeping dogs)는 타겟에서 반드시 제외합니다.무관심 (Lost causes), 잡은 물고기 (Sure things)는 비용 대비 임팩트를 내기 어려운 세그먼트입니다.](../.gitbook/assets/uber\_opt4.png)

위 다이어그램은 Uplift Modeling의 이론적인 동기가 됩니다.

#### Uplift Modeling (Meta-learners for estimating HTE)

Uplift Modeling은 그룹 단위의 유저에 대한 특정 개입(ex. 마케팅 캠페인, 프로모션 등)의 `인과 효과(증분)`를 모델링하는 기법입니다. 주로 프로모션, Up-selling, Cross-selling, 금융 서비스, 유저 이탈 및 유지(CRM) 분야에서 사용 되어 왔습니다.

Uplift modeling은 `인과 효과(증분`)을 모델링하기 때문에, 수요를 만들고 관리하는 모든 비즈니스 영역에서 예상되는 임팩트(더 나아가 ROI)를 설명해줄 수 있다는 점이 장점입니다.

A/B 테스트를 통한 실험 데이터를 Input으로 받아서 사용하기 때문에, A/B 테스트로 캠페인의 효과를 비교하는 것에 그치지 않고 **실제 Business value를 극대화해줄 수 있는 기법**입니다.

알아두어야 할 용어인, Heterogenous Treatment Effect (HTE)란 CATE간의 차이를 의미합니다. 데이터를 계층화(stratify)하고, 각 계층(strata) 내에서 ATE를 추정하여 계층 간의 차이를 비교(subgroup analysis)합니다.

이 CATE를 추정하는 모델은 다양한 종류가 있고, 그 중 하나가 Uplift modeling이며 머신 러닝을 통해 인과 효과를 추론하는 방법입니다. 좀 더 자세히는, Meta-learner를 활용해 CATE를 예측하는 방법입니다.

**Meta-learner**란, 일반적인 `Supervised learning model(i.e. Base-learner)을 인과 효과를 추정하기 위해 다양하게 활용`하는 알고리즘을 의미합니다.

인과 효과 모델링에서는 근본적으로 ground truth가 없기 때문에, 일반적인 Supervised learning에서의 가정을 만족하지 못합니다. 실제 인과 효과는 알 수 없는 신의 영역이기 때문에, **보통 가능한 모든 Meta-learner를 사용**해보고 가장 정확한 인과 효과 추정치를 내는 모델을 선택한다고 합니다.

Meta-learner 각 알고리즘의 장/단점을 인지하고, Uplift modeling을 사용하는 상황과 데이터셋의 크기에 맞게 선택해야 합니다. Meta-learner 알고리즘의 종류에는 S, T, X 가 있습니다. [인과 관계 분석 시리즈 (4): 머신러닝을 이용한 인과관계 추론 (feat. Metalearners)](https://assaeunji.github.io/machine%20learning/2020-07-05-causalml/) 에 상세한 설명이 있어서 참고하시면 좋을 것 같습니다.

다양한 레퍼런스를 찾아보며 용어가 헷갈렸는데요. 실제로 다양한 분야에서 연구되면서, HTE, CATE, Subgroup analysis, Uplift modeling 등 용어가 통일되지 않은 문제가 있다고 합니다. ([Rolling, 2014](https://core.ac.uk/download/pdf/76348572.pdf))

```
The lack of a common framework and language for
this problem may contribute to the disconnect; phrases used for conditional treatment
effect estimation include heterogeneous treatment effect estimation, subgroup analysis,
incremental response modeling, uplift modeling, and true lift modeling.
```

![](../.gitbook/assets/uber\_opt5.png)

[논문 Akshay Kumar 2018](http://cs229.stanford.edu/proj2018/report/296.pdf.) 에서, **유저에게 특정 offer를 보내는 것이 수익성이 있는지 결정하는 비즈니스 문제**에 있어서는 2가지 다른 관점에서 접근이 가능하다고 합니다.

1. Predictive response modeling (보통 알고 있는 classification 분류 문제로, 모델이 데이터가 각 클래스에 할당될 확률을 assign해주는 로직)
2.  Uplift modeling (구매 확률의 \*\*증분 `i.e. probability gain that the customer will buy`\*\*을 모델이 예측해주는 로직)

    _예측의 대상이 일반적인 분류 문제와 달라, Uplift modeling은 인과 추론의 성격을 가집니다._

## Overview

실험 셋업 → 데이터 수집 → 모델링 → 평가 과정을 거치며, 평가에서는 Offline / Online 두 타입으로 나뉩니다.

![](../.gitbook/assets/uber\_opt6.png)

실험 셋업의 절차에서는 실제로 실험을 실행하기 보다는 실험군과 대조군 간, Pre-treatment period에서의 행동 특성을 일치시켜준다는 특이점이 있습니다.

![](../.gitbook/assets/uber\_opt7.png)

데이터 수집은 다음과 같이 진행됩니다. 피쳐에는 우버 라이드 서비스와 우버 이츠 서비스가 모두 포함되었고, 앱 내에서 보게 된 광고, 도시의 특성이 포함되었습니다. Propensity Score 또한 pihat 형태로 포함이 됩니다만 0.5로 통일 되어 있는 상태입니다.

![](../.gitbook/assets/uber\_opt8.png)

코드에서의 데이터 셋 형태와 통계는 아래와 같습니다.

![](../.gitbook/assets/uber\_opt9.png)

#### Meta Learners : Modeling

모델링을 이해하기 위해서 Meta Learner 들에 대한 이해가 필요한데요. CausalML의 근간은 Meta Learner 가 큰 역할을 차지하기 때문입니다. 우선 CausalML은 각 Meta Learner에서 CATE를 계산하기 쉬운 인터페이스(함수 사용)를 제공합니다.

Uplift modeling에서는 2가지 common approach가 있습니다.

![](../.gitbook/assets/uber\_opt10.png)

1. **Two Model (이중 학습기)**
   * 기계학습 모델이 2가지로 생성됩니다. 한 모델은 실험군 관측치로 학습하고, 두번째 모델은 대조군 관측치로 학습합니다.
2. **One Model (단일 학습기)**
   * 1가지 기계학습 모델만 생성됩니다. 실험군, 대조군 관측치 함께 모델에 input으로 들어가 학습합니다.

Meta-learner를 어떤 구조, 모델로 만들 것인가? 에 따라 여러 종류로 나뉩니다. 아래 장표는 슬라이드와 코랩에서 제공하는 각 구조에 대한 설명인데요. 가장 쉽게 표현되어 있습니다.

![](../.gitbook/assets/uber\_opt11.png)

구조는 S, T, X 를 모두 사용하고 아래 함수로 표현이 됩니다.

![](../.gitbook/assets/uber\_opt12.png)

Meta-Learner 들을 통해서 Uplift Score 를 예측하는 것이 핵심인데요. 이 값을 이해하기 위해서, 스코어를 도출하는 예시 간단한 것을 가져왔습니다. ([출처](https://towardsdatascience.com/uplift-modeling-e38f96b1ef60))

> uplift score: the treatment effect between treatment and control

![](../.gitbook/assets/uber\_opt13.png)\
![](../.gitbook/assets/uber\_opt14.png)

## Offline Evaluation

다양한 방식들을 거칩니다. 상세 내용은 [KDD2021 자료](https://drive.google.com/file/d/1QJJUCo4LH5kGQP3kaJlG1RdhjhaJWp-5/view)를 확인해주세요.

1. Base Learner
   * ![](../.gitbook/assets/uber\_opt18.png)
2. Lift Curve
3. Gain Chart
4. Qini Curve
5. Area Under Uplift Curve
6. TMLE (Targeted Maximum Likelihood Estimator)
7. TMLE Evaluation
   * 이를 통해서 Targeting Strategy를 결정합니다. S\_XGB 를 선택했을 때, 60% 상위의 Uplift Score 유저를 타겟팅하는 것이 가장 높은 ATE를 도출할 것입니다. 이는 예산을 40% 줄이면서, Advertising 효율을 67% 증가시킬 수 있습니다.
   * [TMLE 관련 레퍼런스](https://towardsdatascience.com/targeted-maximum-likelihood-tmle-for-causal-inference-1be88542a749)
   * ![](../.gitbook/assets/uber\_opt15.png)
   * ![](../.gitbook/assets/uber\_opt16.png)

## Online Evaluation

![](../.gitbook/assets/uber\_opt17.png)&#x20;

Offline Evaluation을 통해서 결정한 Policy를 집행하는 온라인 실험을 진행했습니다. 결론적으로 Spend 는 46% 줄었고, GB기준 ROAS는 93% 증가했습니다.
