---
description: Long-Term Return-on-Investment at Microsoft via Short-Term Proxies
---

# 단기 Proxy Metric를 통한 장기 ROI 추정 - Microsoft

* 작성자: 허현
* 원문: [KDD 2021 Slide](https://drive.google.com/file/d/1FEKXFHHATntHjsEymXnEw6GAiUGMm8sG/view)


## Double Machine Learning (DML)이란?
두번의 머신러닝을 통해 Treatment-Outcome 효과 추정하는 방법론  

Double Machine Learning (DML) 통해 하고 싶은 것은 아래와 같이 추정 bias를 줄여서 정확도를 높이려는 것 (논문 제목도 [Double/Debiased Machine Learning for Treatment and Causal Parameters](https://arxiv.org/abs/1608.00060))  
![](<../.gitbook/assets/debias_dml.png>)    


### 대략적인 DML 진행 방식
1. T(Treatment), Y(outcome), X(covariates)가 있을 때 [T, X]와 [T, Y] 데이터셋 생성
2. T~X, Y~T ML 모델 훈련
3. ML 모델로 예측한 T_hat과 Y_hat을 각각 T와 Y에서 뺌 → T_tilda, Y_tilda
4. Y_tilda ~ T_tilda로 X의 영향력을 제거한 회귀를 통해 효과 추정

잔차끼리 회귀하여 X의 영향력을 제거한 효과를 구하는 것은 [프리슈-워-로벨 정리](https://datascienceschool.net/03%20machine%20learning/04.05%20%EB%B6%80%EB%B6%84%ED%9A%8C%EA%B7%80.html)에 근거


## 케이스 스터디
목표: 단기간의 데이터로 장기간의 ROI를 측정하게 하는 목적으로 진행  
![](<../.gitbook/assets/attribute_incremental_revenue.png>)    
위와 같이 어떤 액션의 Incremental Revenue를 구하기를 원함

### 문제상황1
인과 그래프는 아래와 같이 그릴 수 있는데 현재시점투자(treatment), 장기수익(outcome)에 과거 투자, 인구통계정보, 과거 수익, 과거 대리변수 등의 confounder가 영향을 주기 때문에 바로 treatment→outcome 구조로 모델링하면 공통원인으로 인한 편향된 수치가 나옴
![](<../.gitbook/assets/longtermroi_problem1.png>)    

DML을 통해 confounder의 영향에 의한 효과를 제거(통제)한 효과를 추정하고자 함
![](<../.gitbook/assets/longtermroi_problem1_sol.png>)    

**DML 절차**
![](<../.gitbook/assets/longtermroi_problem1_dml.png>)    

1. Y를 W로 예측하도록 모델을 만들어서 Y hat을 만듦
2. T를 W로 예측하도록 모델을 만들어서 T hat을 만듦
3. 각각을 잔차화시켜서 (Y-Yhat)을 (T-That)으로 회귀
4. 이렇게 causal effect 추정

### 문제상황2
outcome이 장기수익이기 때문에 아직 관측되지 않았음
![](<../.gitbook/assets/longtermroi_problem2.png>)    

과거 데이터셋을 통해 단기 매출로 장기 매출을 예측하는 ML 모델 만들고, 이 모델로 현재 가진 데이터셋으로 장기 매출을 예측하게 함
![](<../.gitbook/assets/longtermroi_problem2_sol.png>)    

### 문제상황3
현재의 투자가 미래의 투자에 영향을 주고 미래의 투자가 장기수익에 영향을 주어 결과적으로 현재 투자의 영향이 두 번 카운팅 됨
![](<../.gitbook/assets/longtermroi_problem3.png>)    

순차적으로 DML 적용하는 방식(Dynamic DML)으로 해결
![](<../.gitbook/assets/longtermroi_problem3_sol.png>)    

- T_t는 Y_t, Y_t+1, T_t+1 세 가지에 연결되는데 T_t와 Y_t는 (시점이 두 개만 있다 했을 때) 교란 요소가 없기 때문에 빼면 T_t, Y_t+1, T_t+1 세 변수가 남게 됨
- 이 때 DML을 하면 T_t의 영향력을 통제한 T_t+1 → Y_t+1이 나오게 되고(theta_t+1)
- Y_t+1에서 theta_t+1 * T_t+1을 빼면 T_t의 t+1 시점 장기적 영향으로 인한 Y 값이 나오고, 이를 Y_t와 더하면 두 시점에 걸친 T_t를 통해 발생한 Y_adj를 구할 수 있음

시점을 좀 더 확대하여 2년 데이터를 6개월씩 4번 나눴을 때, 4번째 기간 수익부터 여러 시점의 투자 영향을 구하고 그만큼 빼줌

![](<../.gitbook/assets/dynamicdml1.png>)    
![](<../.gitbook/assets/dynamicdml2.png>)    
![](<../.gitbook/assets/dynamicdml3.png>)    

```python
# on historical data construct adjusted outcomes
from econml.dynamic.dml import DynamicDML

panelYadj = panelY.copy()

est = DynamicDML(
    model_y=LassoCV(max_iter=2000), model_t=MultiTaskLassoCV(max_iter=2000), cv=2
)
for t in range(1, n_periods):  # for each target period 1...m
    # learn period effect for each period treatment on target period t
    est.fit(
        long(panelY[:, 1 : t + 1]),
        long(panelT[:, 1 : t + 1, :]),  # reshape data to long format
        X=None,
        W=long(panelX[:, 1 : t + 1, :]),
        groups=long(panelGroups[:, 1 : t + 1]),
    )
    # remove effect of observed treatments
    T1 = wide(panelT[:, 1 : t + 1, :])
    panelYadj[:, t] = panelY[:, t] - est.effect(
        T0=np.zeros_like(T1), T1=T1
    )  # reshape data to wide format
```
(코드상으로는 앞쪽 시점부터 하는 것 같아서 뭐가 맞는지 헷갈림)

### 최종형태
일련의 과정들을 파이프라인화 하면 ROI 측정을 할 수 있음
![](<../.gitbook/assets/unified_pipeline.png>)    
