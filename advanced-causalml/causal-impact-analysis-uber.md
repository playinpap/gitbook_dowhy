---
description: Causal Impact Analysis - CeViChE
---

# Causal Impact Analysis 활용한 고객 가치 분석 - Uber

* 작성자: 허현
* 원문: [KDD 2021 Slide](https://docs.google.com/presentation/d/1FvRtis2fm4c2R7XmRKWMTtZaZjUObW1fGxpNmapmjKI/edit?usp=sharing)

## CeViChE란?
![](<../.gitbook/assets/ceviche.png>)    

**Customer Value Changing Events의 줄임말**

세비체라고 하는 날생선 샐러드에 끼워 맞추려고 노력한 모습 (~~트렌드코리아 김난도 교수님한테 작명교육 좀 받아야 할 것 같음~~)

관측 데이터로 특정 이벤트를 통해 어떻게 고객의 가치가 변했는지/얼마나 효과 있었는지 추정하기 위한 방법론입니다.

### 사용하는 이유
실험이 아닌 관측 데이터로 인과성을 측정할 수 있습니다.  
- 실험과 달리 시간이 이대로 흘렀다면 이러했을 것이다에 기반하는 방식
- 비싸지 않다 (실험과 달리 비용 없음)
- (데이터와 실험디자인에 따라서) 외적 타당성을 가짐
- 행동적 걱정이 적다 (실험은 어떤 효과를 어떤 실험 설계를 통해 검증할 것인지 미리 생각할 게 많은 반면, 관측 데이터를 가공하면 다양한 피쳐를 원하는대로 만들 수 있기 때문)

### A/B 테스트가 어려울 때, 실패할 때 사용되는 접근법을 일반화 한 것
A/B 테스트가 어려운 상황들
- 크로스셀링, 업셀링
- 몰입 감소
- 로열티 프로그램
- 브랜드 효과
- 마케팅 캠페인

> 경고: SOTA 인과추론 방법론을 썼지만 실험데이터가 아닌 관측데이터이기 때문에 인과성을 보장하는 것은 아님


## CeViChE 프레임워크
![](<../.gitbook/assets/ceviche_framework.png>)    
준비 - 성향점수 - 매칭 - 추론 - 검증 과정  

세부 과정에서 다양한 방법론을 사용할 수 있으나 층화 PSM, Meta Learner, 민감도 분석을 사용하는 것으로 정형화 됨

### 준비
![](<../.gitbook/assets/ceviche_framework_setup.png>)    
- cohort preference, 피쳐 모음, 피쳐 엔지니어링을 진행
  - cohort preference : 어떤 시점에 대해 분석을 할지, 어떤 사람들을 대상(통제집단, 실험집단)으로 할지 등에 대한 선택 과정
- 숨겨진 confounder가 있을 수 있기 때문에 편향을 없애기 위해 최대한 피쳐 리스트들을 포괄하려 함

### 성향점수
- 숨겨진 confounder가 있을 수 있어 되도록 많은 피쳐를 사용하는데 피쳐가 많기 때문에 로지스틱 회귀 대신 elasticnetpropensitymodel을 사용
- 로지스틱 회귀에 elasticnet 쓴 것으로 추정됨 (github엔 pass로 코드 생략되었음)
- 머신러닝 예측모델 만들 때처럼 train 데이터셋이 따로 있음
- treatment와 control 그룹의 피쳐 분포를 비교해 최적의 매치를 찾게 함
  - 예측한 성향점수 값을 pihat으로 부르는데(예측하는 값이니까) pihat과 caliper의 범위를 조절하면서 최적의 매치를 찾음
- 층화 성향점수를 match_by_group 기능을 써서 그룹 내에서 성향점수 구하게 할 수도 있음  
  ![](<../.gitbook/assets/match_by_group_stratify.png>)    
  - 일반적으로는 층화를 상위 20%, 40%, 60%, 80%, 100%와 같이 백분위로 쪼갬
- 매칭할 때 SMD(Standard Mean Difference)를 많이 봄
  - 그룹 내 수치 차이의 평균 / 참가자의 표준 편차

  ```python
  def smd(feature, treatment):
      """Calculate the standard mean difference (SMD) of a feature between the
      treatment and control groups.
      The definition is available at
      https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/#s11title
      Args:
          feature (pandas.Series): a column of a feature to calculate SMD for
          treatment (pandas.Series): a column that indicate whether a row is in
                                     the treatment group or not
      Returns:
          (float): The SMD of the feature
      """
      t = feature[treatment == 1]
      c = feature[treatment == 0]
      return (t.mean() - c.mean()) / np.sqrt(.5 * (t.var() + c.var()))
  ```

### 추론
XGB모델과 LinearRegression 모델로 나누어 S/X/T/R Learner 메타러너로 추론
![](<../.gitbook/assets/ceviche_framework_inference.png>)    

### 검증
아래 방법론 활용
- Placebo treatment
- Replace/Add Irrelevant Confounder
- Subset Validation
- [Selection Bias](https://www.mattblackwell.org/files/papers/causalsens.pdf)

## 케이스 스터디
### 이번 Case Study에서 알고 싶은 것
- 전사적 사업 관점에서 우버 라이더를 우버 이터로 활동하도록 크로스셀링하는 것이 좋은지, 나쁜지
- 크로스셀링의 증분적 효과가 어느 정도인지

### 준비
![](<../.gitbook/assets/ceviche_treatment_setup.png>)    

Pre-Treatment 기간에는 둘 다 Uber 라이더인 사람만 대상으로

Treatment 기간에는 그대로 Uber 라이더인 경우 Control Group, 이터로 활동한 사람을 Treatment Group으로

Post-Treatment 기간에는 Control Group만 Uber 라이더 유지 조건

GB(Gross Booking, 총 요청 수)로 성과 확인

### 결론
Uber Rider → Eater로 전환할 때 인센티브를 줬더니 GB기준 1.5배 lift가 있었고, Uber Ride로 버는 돈도 늘었다  
![](<../.gitbook/assets/ceviche_treatment_lift.png>)    
