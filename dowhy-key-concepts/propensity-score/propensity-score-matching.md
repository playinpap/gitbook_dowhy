# Propensity Score Matching

- 작성자: 김가연

---

**성향점수 대응(Propensity Score Matching, PSM)** 은 성향점수가 동일하거나 유사한 처치집단의 개체와 비교집단의 개체를 한 쌍으로 매칭하는 방법입니다.

아래 그림은 운동 여부에 따른 콜레스테롤 수치를 비교하는 사례로, 성향점수 대응 결과 연령대가 비슷한 처치집단의 개체와 비교집단의 개체가 매칭되었습니다. 각 처치집단의 개체와 비교집단의 개체가 서로 반사실(counterfactual)임을 알 수 있습니다.

![운동을 하지 않는 비교집단과 운동을 하는 처치집단의 평균 콜레스테롤 수치 (출처: [kdd-tutorial 2018](https://causalinference.gitlab.io/kdd-tutorial/methods.html))](https://user-images.githubusercontent.com/76609403/151827491-4d3f84da-b76d-42df-a910-1c3b40404751.png)

![유사한 성향점수를 갖는 비교집단의 개체와 처치집단의 개체를 매칭 (출처: [kdd-tutorial 2018](https://causalinference.gitlab.io/kdd-tutorial/methods.html))](https://user-images.githubusercontent.com/76609403/151827715-9fc5f95a-e028-4c4a-a2db-3eb01098515b.png)

이러한 성향점수를 비교하는 방법에는 매칭 기준에 따라 최근린, 라디우스, 최적화 매칭 등으로 구분할 수 있습니다.

DoWhy 라이브러리의 [propensity_score_matching_estimator](https://microsoft.github.io/dowhy/_modules/dowhy/causal_estimators/propensity_score_matching_estimator.html#PropensityScoreMatchingEstimator.construct_symbolic_estimator) 모듈에서는 Nearest Neighbor Matching 을 사용합니다.

```python
# dowhy.causal_estimators.propensity_score_matching_estimator _estimate_effect 함수 코드 일부

from sklearn.neighbors import NearestNeighbors

# ATT (average treatment effect for the treated) 계산
control_neighbors = (
    NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    .fit(control['propensity_score'].values.reshape(-1, 1))
)
distances, indices = control_neighbors.kneighbors(treated['propensity_score'].values.reshape(-1, 1))
self.logger.debug("distances:")
self.logger.debug(distances)

att = 0
numtreatedunits = treated.shape[0]
for i in range(numtreatedunits):
    treated_outcome = treated.iloc[i][self._outcome_name].item()
    control_outcome = control.iloc[indices[i]][self._outcome_name].item()
    att += treated_outcome - control_outcome

att /= numtreatedunits

# ATC (average treatment effect for the control) 계산
treated_neighbors = (
    NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    .fit(treated['propensity_score'].values.reshape(-1, 1))
)
distances, indices = treated_neighbors.kneighbors(control['propensity_score'].values.reshape(-1, 1))
atc = 0
numcontrolunits = control.shape[0]
for i in range(numcontrolunits):
    control_outcome = control.iloc[i][self._outcome_name].item()
    treated_outcome = treated.iloc[indices[i]][self._outcome_name].item()
    atc += treated_outcome - control_outcome

atc /= numcontrolunits
```

다만 Nearest Neighbor Matching 은 가장 근접한 성향점수를 가져도 실제로 큰 차이를 가질 수 있다는 한계가 있습니다. 이 때에는 처치집단과 비교집단의 성향점수 차이를 일정 범위로 유지하는 threshold 를 설정한 매칭 방법을 사용합니다. 해당 범위 안에서 대응 짝을 찾을 수 있도록 강제함으로써, 좀 더 엄격하게 실험설계와 유사한 상황을 만들어 낼 수 있습니다.

DoWhy 라이브러리 또한 주어진 radius 값보다 큰 neighbors 는 제외할 수 있도록 [관련 코드](https://microsoft.github.io/dowhy/_modules/dowhy/causal_estimators/propensity_score_matching_estimator.html#:~:text=%23%20TODO%20remove%20neighbors%20that%20are%20more%20than%20a%20given%20radius%20apart)를 추가할 계획인 것으로 보입니다. (2022.01.31 기준)
