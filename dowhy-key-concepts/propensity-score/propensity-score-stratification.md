# Propensity Score Stratification

- 작성자: 김가연

---

**성향점수 층화(Propensity Score Stratification)** 는 성향점수가 유사한 처치집단과 통제집단 개체들을 K개 집단으로 층화하는 방법입니다.

아래 그림은 앞서 다뤘던 운동 여부에 따른 콜레스테롤 수치 비교 사례로, 성향점수의 범위에 따라 집단을 3개로 층화한 결과 연령대가 비슷한 처치집단의 개체들과 비교집단의 개체들이 같은 층(strata)에 포함되었습니다.

![운동을 하지 않는 비교집단과 운동을 하는 처치집단 (출처: kdd-tutorial 2018)](https://user-images.githubusercontent.com/76609403/152100158-c0b6078b-57ef-4021-b01a-086b951742ce.png)

![성향점수의 범위에 따라 집단을 3개로 층화 (출처: kdd-tutorial 2018)](https://user-images.githubusercontent.com/76609403/152100220-295ee8d3-6df6-4b91-8d76-cd1e825c631e.png)

처치효과를 추정하기 위해 각 층에는 충분한 데이터가 있어야 하며 층의 개수 또한 중요합니다. 보편적으로 100개 이하의 data points 를 갖는 경우 5개의 층, 10,000 ~ 1,000,000 혹은 그 이상의 data points 를 갖는 경우 100 ~ 1000개의 층이 적절합니다.

DoWhy 라이브러리의 [propensity_score_stratification_estimator](https://microsoft.github.io/dowhy/_modules/dowhy/causal_estimators/propensity_score_matching_estimator.html#PropensityScoreMatchingEstimator.construct_symbolic_estimator) 모듈에서는 기본 strata 개수(num_strata)가 50 이고 각 strata 에 속한 처치집단 혹은 비교집단의 data points 개수(clipping_threshold)가 10 입니다. 이때 clipping_threshold 에 미달인 strata 는 처치효과 계산 시 제외되는데, 기술적으로는 LATE (Local Average Treatment Effect) 를 계산하는 것과 같습니다.

```python
# dowhy.causal_estimators.propensity_score_stratification_estimator _estimate_effect 함수 코드 일부

# 각 개체를 성향점수 오름차순으로 정렬 후 strata 지정
num_rows = self._data[self._outcome_name].shape[0]
self._data['strata'] = (
    (self._data['propensity_score'].rank(ascending=True) / num_rows) * self.num_strata
).round(0)

self._data['dbar'] = 1 - self._data[self._treatment_name[0]] # 비교집단일 경우 1(True)
self._data['d_y'] = self._data[self._treatment_name[0]] * self._data[self._outcome_name] # 처치집단의 outcome
self._data['dbar_y'] = self._data['dbar'] * self._data[self._outcome_name] # 비교집단의 outcome

# clipping_threshold 보다 적은 처치집단 혹은 비교집단이 있는 strata 는 제외
stratified = self._data.groupby('strata')
clipped = stratified.filter(
    lambda strata: min(strata.loc[strata[self._treatment_name[0]] == 1].shape[0],
                       strata.loc[strata[self._treatment_name[0]] == 0].shape[0]) > self.clipping_threshold
)
self.logger.debug("After using clipping_threshold={0}, here are the number of data points in each strata:\n {1}".format(self.clipping_threshold, clipped.groupby(['strata',self._treatment_name[0]])[self._outcome_name].count()))
if clipped.empty:
    raise ValueError("Method requires strata with number of data points per treatment > clipping_threshold (={0}). No such strata exists. Consider decreasing 'num_strata' or 'clipping_threshold' parameters.".format(self.clipping_threshold))

# 각 strata 별로 처치집단 혹은 비교집단의 outcome 가중합 (비교집단 개체 수에 대해)
weighted_outcomes = clipped.groupby('strata').agg({
    self._treatment_name[0]: ['sum'],
    'dbar': ['sum'],
    'd_y': ['sum'],
    'dbar_y': ['sum']
})
weighted_outcomes.columns = ["_".join(x) for x in weighted_outcomes.columns.ravel()]
treatment_sum_name = self._treatment_name[0] + "_sum"
control_sum_name = "dbar_sum"

weighted_outcomes['d_y_mean'] = weighted_outcomes['d_y_sum'] / weighted_outcomes[treatment_sum_name] # 처치집단의 평균 outcome
weighted_outcomes['dbar_y_mean'] = weighted_outcomes['dbar_y_sum'] / weighted_outcomes['dbar_sum'] # 비교집단의 평균 outcome
weighted_outcomes['effect'] = weighted_outcomes['d_y_mean'] - weighted_outcomes['dbar_y_mean'] # 처치집단의 평균 outcome - 비교집단의 평균 outcome

total_treatment_population = weighted_outcomes[treatment_sum_name].sum() # 처치집단 개체 수
total_control_population = weighted_outcomes[control_sum_name].sum() # 비교집단 개체 수
total_population = total_treatment_population + total_control_population # 전체 개체 수
self.logger.debug("Total number of data points is {0}, including {1} from treatment and {2} from control.". format(total_population, total_treatment_population, total_control_population))

if self._target_units=="att": # ATT 계산
    est = (weighted_outcomes['effect'] * weighted_outcomes[treatment_sum_name]).sum() / total_treatment_population
elif self._target_units=="atc": # ATC 계산
    est = (weighted_outcomes['effect'] * weighted_outcomes[control_sum_name]).sum() / total_control_population
elif self._target_units == "ate": # ATE 계산
    est = (weighted_outcomes['effect'] * (weighted_outcomes[control_sum_name]+weighted_outcomes[treatment_sum_name])).sum() / total_population
else:
    raise ValueError("Target units string value not supported")
```

성향점수 층화 방법은 비슷한 공변량 분포를 갖는 처치집단과 비교집단의 outcome 을 비교해 처치효과를 계산하므로 관찰되지 않은 공변량, 즉 혼동요인에 의한 효과가 의미 있게 감소하는 것으로 알려져 있습니다. 

또한 층에 따른 차별적인 처치효과를 알 수 있으므로 보다 많은 정보를 확인할 수 있는 장점이 있으나 이를 위해서는 각 층에 충분히 많은 개체 수가 필요합니다.
