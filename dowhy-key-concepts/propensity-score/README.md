---
description: Propensity Score Matching, Stratification, Weighting
---

# 성향점수 (Propensity Score)

* 작성자: 김가연

***

인과관계의 성립을 위해서는 다음과 같은 3가지 조건이 필요합니다.

1. 원인변수의 시간적 선행
2. 원인 및 결과변수 간 상관(Correlation)
3. 혼동요인(Confounder)의 통제

하지만 혼동요인에 대한 통제는 엄격한 수준에서 불가능합니다. 무작위 통제 실험(Randomized Controlled Trial, RCT)이 아닌 이미 수집된 자료를 바탕으로 인과관계를 추론할 경우, 사전 동등성이 확보되어있지 않아 선택 편의(Selection Bias)의 문제가 발생합니다.

이러한 문제를 최소화하기 위한 통계적 해결법 중 하나가 '성향점수(Propensity Score)' 방법입니다.

> **성향점수(Propensity Score)란,** 개체들이 처치집단(Treatment Group)이나 비교집단(Control Group)에 속할 가능성에 영향을 주는 모든 변수들, 즉 공변량(Covariates)의 값이 주어졌을 때 해당 공변량 값을 가진 개체가 **처치집단에 배치될 조건부 확률**입니다.

성향점수를 식으로 표현하면 아래와 같습니다.

$$
e(X) = prob(T=1 | X)
$$

X 라는 공변량이 주어졌을 때 특정 개체가 통제집단이 아닌 처치집단에 배치될 확률로, 주어진 공변량 X 를 이용해 원인 배치 과정 T 를 모형화합니다.

즉, RCT 와 매우 비슷하게 만드는 통계적 접근 방법이라고 할 수 있겠습니다.

성향점수는 로지스틱 회귀모형, 프로빗 모형 등을 통해 추정될 수 있으며 이를 대응(Matching), 층화(Stratification), 가중(Weighting) 등의 방법을 통해 인과효과 추정에 활용합니다. 다음 장부터는 각 방법들에 대해 알아보겠습니다.

***

성향점수(Propensity Score) 챕터를 작성하며 참고한 자료는 아래와 같습니다.

\[1] [R 기반 성향점수분석 : 루빈 인과모형 기반 인과추론](https://tidyverse-korea.github.io/seoul-R/data/RMeetup\_PSA\_slide\_210414.pdf)

\[2] [경향점수를 활용한 인과효과 추정 방법 비교 : 대응, 가중, 층화, 이중경향점수 보정 The Journal of Curriculum and Evaluation 2019, Vol. 22, No. 2, pp. 269～291](https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE09249298)

\[3] [KDD Tutorial 2018](https://causalinference.gitlab.io/kdd-tutorial/methods.html)

\[4] [Microsoft DoWhy Package](https://microsoft.github.io/dowhy/\_modules/index.html)
