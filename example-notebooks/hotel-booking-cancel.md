# DoWhy - The Causal Story Behind Hotel Booking Cancellations
### 다른 호텔 방에 배정되는 것이, 고객의 호텔 예약 취소에 어떤 영향을 줄까?
[원문](https://microsoft.github.io/dowhy/example_notebooks/DoWhy-The%20Causal%20Story%20Behind%20Hotel%20Booking%20Cancellations.html)

![Screenshot%20from%202020-09-29%2019-08-50.png](attachment:Screenshot%20from%202020-09-29%2019-08-50.png)

호텔 예약의 취소를 야기하는 요소들이 무엇인지에 대해서 고려합니다. 이 분석은 [Antonio, Almeida and Nunes (2019)](https://www.sciencedirect.com/science/article/pii/S2352340918315191) 의 호텔 예약 데이터셋을 활용했습니다. GitHub 에서는 이 링크에서 확인하실 수 있습니다 : [rfordatascience/tidytuesday](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-02-11/readme.md). 

호텔 예약이 취소되기 까지는 다양한 이유들이 존재합니다. 
> A customer may have requested something that was not available (e.g., car parking), a customer may have found later that the hotel did not meet their requirements, or a customer may have simply cancelled their entire trip. Some of these like car parking are actionable by the hotel whereas others like trip cancellation are outside the hotel's control. In any case, we would like to better understand which of these factors cause booking cancellations. 

이러한 상황에서, 호텔 예약을 취소하는 원인을 찾기 위해서는 각 고객들이 랜덤하게 두 가지의 카테고리에 속하는 방식의 *Randomized Controlled Trials* 와 같은 실험이 가장 Golden Standard인데요. 이러한 실험은 특정 상황에서는 너무 비용이 크거나, 윤리적이지 않습니다. 예를 들어 호텔이 고객들에게 서비스의 차등을 준다는 점을 깨닫게 된다면, 호텔의 명성에 손해가 갈 수 있습니다.
> each customer is either assigned a car parking or not. However, such an experiment can be too costly and also unethical in some cases (for example, a hotel would start losing its reputation if people learn that its randomly assigning people to different level of service). 

관측 데이터를 통해서, 이러한 질문에 대답할 수는 없을까요?



```python
%reload_ext autoreload
%autoreload 2
```


```python
# Config dict to set the logging level : 로깅의 수준을 세팅하기 위한 config 라이브러리

import logging.config
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'level': 'INFO',
        },
    }
}

logging.config.dictConfig(DEFAULT_LOGGING)
# Disabling warnings output
import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings('ignore')

#!pip install dowhy : dowhy 설치 되지 않았을 경우
import dowhy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
dataset = pd.read_csv('https://raw.githubusercontent.com/Sid-darthvader/DoWhy-The-Causal-Story-Behind-Hotel-Booking-Cancellations/master/hotel_bookings.csv')
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>304.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>240.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>2015-07-03</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
dataset.columns
```




    Index(['hotel', 'is_canceled', 'lead_time', 'arrival_date_year',
           'arrival_date_month', 'arrival_date_week_number',
           'arrival_date_day_of_month', 'stays_in_weekend_nights',
           'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
           'country', 'market_segment', 'distribution_channel',
           'is_repeated_guest', 'previous_cancellations',
           'previous_bookings_not_canceled', 'reserved_room_type',
           'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
           'company', 'days_in_waiting_list', 'customer_type', 'adr',
           'required_car_parking_spaces', 'total_of_special_requests',
           'reservation_status', 'reservation_status_date'],
          dtype='object')



## Data Description
변수와 그에 대한 설명을 확인하기 위해서는 이 링크를 참조해주세요. 
https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-02-11/readme.md



## Feature Engineering

의미 있는 변수들을 만들고, 데이터 셋의 디멘션을 줄이기 위해서 Feature Engineering 을 진행합니다. 
- **Total Stay** =  stays_in_weekend_nights + stays_in_week_nights
- **Guests** = adults + children + babies
- **Different_room_assigned** = 1 if reserved_room_type & assigned_room_type are different, 0 otherwise.


```python
# Total stay in nights
dataset['total_stay'] = dataset['stays_in_week_nights']+dataset['stays_in_weekend_nights']

# Total number of guests
dataset['guests'] = dataset['adults']+dataset['children'] +dataset['babies']

# Creating the different_room_assigned feature
dataset['different_room_assigned']=0
slice_indices =dataset['reserved_room_type']!=dataset['assigned_room_type']
dataset.loc[slice_indices, 'different_room_assigned'] = 1
```


```python
# Total stay in nights
dataset['total_stay'] = dataset['stays_in_week_nights']+dataset['stays_in_weekend_nights']

# Total number of guests
dataset['guests'] = dataset['adults']+dataset['children'] +dataset['babies']

# Creating the different_room_assigned feature
dataset['different_room_assigned']=0
slice_indices =dataset['reserved_room_type']!=dataset['assigned_room_type']
# Reserved room type 과 Assigned room type 이 다른 인덱스에 대해서, 'different_room_assigned' 컬럼을 1로 채워라
dataset.loc[slice_indices,'different_room_assigned']=1 

# Deleting older features : 가공에 사용되어서 불필요한 피쳐는 삭제합니다.
dataset = dataset.drop(['stays_in_week_nights','stays_in_weekend_nights','adults','children','babies'
                        ,'reserved_room_type','assigned_room_type'],axis=1)

dataset.columns
```




    Index(['hotel', 'is_canceled', 'lead_time', 'arrival_date_year',
           'arrival_date_month', 'arrival_date_week_number',
           'arrival_date_day_of_month', 'meal', 'country', 'market_segment',
           'distribution_channel', 'is_repeated_guest', 'previous_cancellations',
           'previous_bookings_not_canceled', 'booking_changes', 'deposit_type',
           'agent', 'company', 'days_in_waiting_list', 'customer_type', 'adr',
           'required_car_parking_spaces', 'total_of_special_requests',
           'reservation_status', 'reservation_status_date', 'total_stay', 'guests',
           'different_room_assigned'],
          dtype='object')



NULL 값이 있거나, 유저 단위에 1:1 대응이 될 정도로 유니크한 값이 많은 컬럼은 삭제합니다. 또한 `country`의 결측치는 최빈값으로 메웁니다. `distribution_channel` 이라는 컬럼은 `market_segment` 와 겹치는 부분이 많기 때문에 삭제됩니다.


```python
dataset.isnull().sum() # Country,Agent,Company contain 488,16340,112593 missing entries 
```




    hotel                                  0
    is_canceled                            0
    lead_time                              0
    arrival_date_year                      0
    arrival_date_month                     0
    arrival_date_week_number               0
    arrival_date_day_of_month              0
    meal                                   0
    country                              488
    market_segment                         0
    distribution_channel                   0
    is_repeated_guest                      0
    previous_cancellations                 0
    previous_bookings_not_canceled         0
    booking_changes                        0
    deposit_type                           0
    agent                              16340
    company                           112593
    days_in_waiting_list                   0
    customer_type                          0
    adr                                    0
    required_car_parking_spaces            0
    total_of_special_requests              0
    reservation_status                     0
    reservation_status_date                0
    total_stay                             0
    guests                                 4
    different_room_assigned                0
    dtype: int64




```python
# 국가 컬럼의 최빈값 확인
dataset['country'].mode()
```




    0    PRT
    dtype: object




```python
dataset = dataset.drop(['agent','company'],axis=1)

# Replacing missing countries with most freqently occuring countries
dataset['country']= dataset['country'].fillna(dataset['country'].mode()[0])
```

불필요한 컬럼을 판단해 삭제하고, `different_room_assigned` 와 `is_canceled` 라는 컬럼을 여부에 따라 1,0으로 대체합니다. 


```python
dataset = dataset.drop(['reservation_status','reservation_status_date','arrival_date_day_of_month'],axis=1)
dataset = dataset.drop(['arrival_date_year'],axis=1)
dataset = dataset.drop(['distribution_channel'], axis=1)
```


```python
# Replacing 1 by True and 0 by False for the experiment and outcome variables

dataset['different_room_assigned']= dataset['different_room_assigned'].replace(1,True)
dataset['different_room_assigned']= dataset['different_room_assigned'].replace(0,False)
dataset['is_canceled']= dataset['is_canceled'].replace(1,True)
dataset['is_canceled']= dataset['is_canceled'].replace(0,False)
dataset.dropna(inplace=True)
print(dataset.columns)
dataset.iloc[:, 5:20].head(100)
```

    Index(['hotel', 'is_canceled', 'lead_time', 'arrival_date_month',
           'arrival_date_week_number', 'meal', 'country', 'market_segment',
           'is_repeated_guest', 'previous_cancellations',
           'previous_bookings_not_canceled', 'booking_changes', 'deposit_type',
           'days_in_waiting_list', 'customer_type', 'adr',
           'required_car_parking_spaces', 'total_of_special_requests',
           'total_stay', 'guests', 'different_room_assigned'],
          dtype='object')





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meal</th>
      <th>country</th>
      <th>market_segment</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>total_stay</th>
      <th>guests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BB</td>
      <td>GBR</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BB</td>
      <td>GBR</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BB</td>
      <td>GBR</td>
      <td>Online TA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.00</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>BB</td>
      <td>PRT</td>
      <td>Online TA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>73.80</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>BB</td>
      <td>PRT</td>
      <td>Online TA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>117.00</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>HB</td>
      <td>ESP</td>
      <td>Offline TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>196.54</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>BB</td>
      <td>PRT</td>
      <td>Online TA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>99.30</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>BB</td>
      <td>DEU</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>90.95</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 15 columns</p>
</div>



데이터셋에서, No Deposit 으로 예약했으면서 취소한 비율에 대해서 미리 확인해봅니다. 총 104,637건 중에서 29,690건이 취소되었습니다.


```python
dataset = dataset[dataset.deposit_type=="No Deposit"]
dataset.groupby(['deposit_type','is_canceled']).count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>hotel</th>
      <th>lead_time</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>meal</th>
      <th>country</th>
      <th>market_segment</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>booking_changes</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>total_stay</th>
      <th>guests</th>
      <th>different_room_assigned</th>
    </tr>
    <tr>
      <th>deposit_type</th>
      <th>is_canceled</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">No Deposit</th>
      <th>False</th>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
    </tr>
    <tr>
      <th>True</th>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
    </tr>
  </tbody>
</table>
</div>



판다스에는 데이터프레임의 복사본을 만들어주는 pandas.DataFrame.copy가 있고 a = b와는 다른 방식의 복사입니다.
a = b는 원본 데이터가 변하면 똑같이 변하는 얕은 복사인 반면, pandas.DataFrame.copy는 복사 당시의 데이터프레임 상태만 복사되는 깊은 복사입니다.
얕은 복사를 하면 복사본은 원본과 데이터/index를 공유하지만, 깊은 복사는 복사본이 자신만의 데이터/index를 갖게합니다.

`dataframe.copy(deep = defalut is True)`


```python
dataset_copy = dataset.copy(deep=True) 
```

## Calculating Expected Counts : Confounder 를 유추할 수 있는 간단한 방법


예약 취소의 개수와, 다른 방에 배정되는 케이스의 개수가 심각하게 불균형적이기 때문에, 랜덤으로 최초 1000개의 관측치를 선택하며 `is_cancelled` & `different_room_assigned` 가 같은 값을 달성하는지를 확인합니다. 이 전체적인 프로세스는 10000회 반복되고, 그 중에서 예상되는 달성 횟수는 50%에 가깝습니다. 그 이유는 두개의 변수가 같은 값을 랜덤하게 가질 수 있을 확률은 50% 이기 때문입니다.

따라서 통계적으로 이야기하면, 이 단계에서 유한한 결론이 있는 것은 아닙니다. 따라서, 고객이 예약한 방과 다른 방을 배정하는 상황은 그 고객이 예약을 취소하게 할수도, 아닐수도 있습니다. 

**첫번째 시나리오**는, 아무 조건 가정 없이 샘플링을 진행합니다.


```python
counts_sum=0
for i in range(1,10000):
        counts_i = 0
        # 데이터셋에서 1000개의 행을 랜덤 샘플링합니다.
        rdf = dataset.sample(1000)
        # is_canceled FALSE, different_room_assigned FALSE
        # is_canceled TRUE, different_room_assigned TRUE 인 행들의 개수를 셉니다.
        counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
        # 이 과정을 10000회 반복하면서, 총 몇개의 행들이 같았는지를 더합니다.
        counts_sum+= counts_i

# 10000회 반복 시행을 종료했을 때, 평균적으로 몇개의 행들이 같았는지를 파악합니다.
counts_sum/10000
```




$\displaystyle 588.7015$



**두번째 시나리오**는, 예약에 변경이 없었을 것이라는 조건을 가정한 후에 샘플링을 진행합니다.


```python
# Expected Count when there are no booking changes : booking_changes = 0 이라는 조건을 걸어 계층화할 경우

counts_sum=0
for i in range(1,10000):
        counts_i = 0
        # 조건을 건 후 그 중에서 1000개의 행을 랜덤 샘플링합니다.
        rdf = dataset[dataset["booking_changes"]==0].sample(1000)
        counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
        counts_sum+= counts_i
        
counts_sum/10000
```




$\displaystyle 572.608$



**세번째 시나리오**는 예약에 변경이 있었을 것이라는 조건(>0)을 가정한 후에 샘플링을 진행합니다. 


```python
# Expected Count when there are booking changes = 66.4%
counts_sum=0
for i in range(1,10000):
        counts_i = 0
        # 조건을 건 후 그 중에서 1000개의 행을 랜덤 샘플링합니다.
        rdf = dataset[dataset["booking_changes"]>0].sample(1000)
        counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
        counts_sum+= counts_i
        
counts_sum/10000
```




$\displaystyle 666.015$



확실히 예약에 변경이 있었을 것이라는 조건을 통해서, 무언가 숫자에 변화를 볼 수 있습니다. `booking_changes` 변수가 곧 교란변수가 될 수 있다는 힌트가 됩니다. 

하지만 `booking_changes` 변수가 유일한 교란변수일까요? 만일 관측되지 않은 교란변수가 있고, 그것이 데이터셋에 변수 형태로 포함되지 않은 정보라면, 우리는 이 `booking_changes` 변수가 곧 교란변수가 될 수 있다는 주장을 계속할 수 있을까요?

<font size="6"> *DoWhy* 의 시작</font>

## Step-1. Create a Causal Graph : 인과 그래프 생성하기

예측 모델링 문제에 대한 사전지식을, 인과 그래프 형태로 가정을 사용해서 표현합니다. 전체 그래프를 이 단계에서 표현하지 않아도, 일부 그래프만 표현하여도 충분합니다. 나머지는 DoWhy 를 통해서 찾을 수 있습니다.

아래 내용이 인과 모형으로 해석된 몇 개의 가정들입니다.

- `Market Segment` 변수 : TA(Travel Agents), TO(Tour Operators) 라는 2가지 값으로 구성되며, `Lead Time` 변수에 영향을 줍니다.
- `Country` 변수 : 한 사람이 일찍 호텔을 예약할지, 아닐지에 대해 좌우하는 역할을 할 수 있으며, 궁극적으로 일찍 호텔을 예약하는 행동은 `Lead Time` 변수에 영향을 줍니다. 그 외로도, `Meal` 변수에도 영향을 줍니다.
- `Lead Time` 변수 : `Days in Waitlist` 변수에 확실하게 영향을 줍니다. 늦게 예약할수록, 예약을 할 수 있는 기회가 적어지기 때문입니다. 추가적으로 높은 `Lead Time` 은 `Cancellations` 변수를 높아지게 만들 수  있습니다.
- `Previous Booking Retentions` 변수 : 고객의 `Repeated Guest` 여부를 좌우하고, 이 두개의 변수는 모두 `Cancelled` 에도 영향을 줄 수 있습니다. 리텐션이 높았던 고객일수록 취소할 확률이 낮고, 리텐션이 낮았던 즉 늘 취소했던 고객이라면 취소할 확률이 높습닌다.
- `Booking Changes` 변수 : 예약을 변경하는 것은 `Different room assigned` 변수를 좌우하고, `Cancellation` 에도 영향을 줄 수 있습니다.


정리하면, `Booking Changes` 변수가 Treatment, Outcome 에 영향을 주는 가장 유일한 교란변수일 확률은 낮습니다. 현재 변수는 물론, 데이터셋에 고려되지 않은 정보들까지 포함하여 다른 관측되지 않은 교란변수가 많을 것입니다.

### Pygraphviz 활용

pygraphviz 설치에서 에러가 발생할 경우, 터미널에서 아래와 같이 해결할 수 있습니다.

`brew install graphviz` 
`pip install graphviz`
`pip install pygraphviz`

Diagraph 란, Edge 와 Node 로 구성되고 추가적인 데이터나 특성을 포함합니다. Self loops 는 가능하지만, Multiple edge 는 허용되지 않습니다. 노드는 주로 임의의 해싱가능한 Python 객체입니다. Edge는 노드간의 링크입니다. [설명](https://networkx.org/documentation/stable/reference/classes/digraph.html#)



```python
import pygraphviz

# 변수명[label = 그림에 표기될 이름]
# 변수명(label 없으면 변수명 그대로 그림에 표기됨)
# U = Unobserved confounder
# 변수명 -> 변수명 : 영향을 주는 관계
# 변수명 -> {변수명, 변수명} : 다수에 영향을 주는 변수 관계

causal_graph = """digraph {
    different_room_assigned[label="Different Room Assigned"];
    is_canceled[label="Booking Cancelled"];
    booking_changes[label="Booking Changes"];
    previous_bookings_not_canceled[label="Previous Booking Retentions"];
    days_in_waiting_list[label="Days in Waitlist"];
    lead_time[label="Lead Time"];
    market_segment[label="Market Segment"];
    country[label="Country"];
    U[label="Unobserved Confounders"];
    
    is_repeated_guest;
    total_stay;
    guests;
    meal;
    hotel;
    
    U->different_room_assigned; 
    U->is_canceled;
    U->required_car_parking_spaces;
    
    market_segment -> lead_time;
    lead_time -> is_canceled; 
    country -> lead_time;
    different_room_assigned -> is_canceled;
    country -> meal;
    lead_time -> days_in_waiting_list;
    days_in_waiting_list -> is_canceled;
    previous_bookings_not_canceled -> is_canceled;
    previous_bookings_not_canceled -> is_repeated_guest;
    is_repeated_guest -> is_canceled;
    total_stay -> is_canceled;
    guests -> is_canceled;
    booking_changes -> different_room_assigned; 
    booking_changes -> is_canceled; 
    hotel -> is_canceled;
    required_car_parking_spaces -> is_canceled;
    total_of_special_requests -> is_canceled;
    
    country -> {hotel, required_car_parking_spaces,total_of_special_requests,is_canceled};
    market_segment -> {hotel, required_car_parking_spaces,total_of_special_requests,is_canceled};
    }"""
```

여기서, Treatment 는 고객이 예약한 방을 그대로 배정해주는 것입니다. Outcome은 예약이 취소되었는지에 대한 여부입니다.
- Common Causes : 우리의 인과 그래프에 따라 Treatment, Outcome 에 영향을 줄 수 있는 변수들을 의미합니다.
- 앞서 세운 인과적 가정에 따르면 이 Common Causes 가 될 수 있는 조건을 만족하는 변수는 `Booking Changes` 그리고 `Unobserved Confounders` 변수 2가지 입니다. 


따라서, 만일 그래프를 명시적으로 작성하지 않을 경우에는 아래 함수에 따라 파라미터들을 제공할 수 있습니다.
>So if we are not specifying the graph explicitly (Not Recommended!), one can also provide these as parameters in the function mentioned below.


```python
model= dowhy.CausalModel(
        data = dataset,
        graph = causal_graph.replace("\n", " "), # causal_graph 라는 문자열에서 띄어쓰기를 space 로 대체해주는 역할
        treatment = 'different_room_assigned',
        outcome = 'is_canceled') 
model.view_model()

from IPython.display import Image, display
display(Image(filename = "causal_model.png"))
```

    INFO:dowhy.causal_model:Model to find the causal effect of treatment ['different_room_assigned'] on outcome ['is_canceled']



![png](HotelBooking_files/HotelBooking_33_1.png)


## Step-2. Identify Causal Effect : 인과 효과 식별하기

우리는 모든 다른 것들을 유지한 채로 Treatment 를 변화시킬 때, Outcome 에 변화가 있다면 Treatment가 Outcome의 원인이 된다라고 이야기합니다.
따라서 이 단계에서는, 인과 그래프의 특성들을 활용해서 추정하고자 하는 인과 효과를 식별합니다.

dowhy.CausalModel() 클래스를 활용해서 모델을 입력하게 되면, 아래 함수들을 사용할 수 있습니다. [모듈 코드](https://microsoft.github.io/dowhy/_modules/dowhy/causal_model.html)

- identify_effect()
   - Identified estimand 를 리턴하는 주요한 메소드. 만일 Estimand 의 타입이 non-parametric ATE라면, 주어진 인과 모형에서 Identified estimand가 있는지를 확인하기 위해서 Backdoor / Instrumental Variable / Frontdoor Identification 메소드들을 사용합니다. 
- estimate_effect()
    - Step-3 에서 이어짐
- do()
- refute_estimate()
- view_model()
- interpret()
- summary()


### Estimand
예를 들어, 7년간의 월별 수익이 Gaussian 분포를 따른다고 가정합니다. Gaussian 분포는 평균, 분산을 Parameter 로 가지고 있어, 이 경우에 개별 월 수입은 Random variable 이라고 할 수 있습니다. [출처](https://fullfu.tistory.com/2)

![image.png](attachment:image.png)

- Estimand : (예시) Gaussian의 Parameter 가 되는 평균, 분산
- Estimate : Random variable 로 월별 수익을 추정한 값
    - 평균의 Estimate : 100만원 +- standard error
    - 표준 편차의 Estimate : 10만원
- Estimator : Random variable 로부터 Estimate를 얻어내는 함수
    - 평균의 Estimator : mu = sigma(월 수입) / 총 월 수
    - 표준 편차의 Estimate : s = sqrt(sigma(월 수입 - 평균 월 수입) / (총 월 수 -1))
    
> 인과효과 연구에서 관심 모집단의 특성과 연관지어 분석 결과를 해석하고 추론하는 것은
중요하며, 이에 연구자는 통계적 추론(inference)를 위한 관심 모수(estimand; parameter of interest)를 정의하고 이에 대한 추정치(estimates)를 산출하게 된다.

### Non-parametric ATE
Machine Learning: A Probabilistic Perspective (Adaptive computation and Machine Learning Series) by Kevin P. Myrphy (Author). Chapter 1. 에 따르면, [출처](https://process-mining.tistory.com/131)

- Parametric model : The model has a fixed number of parameters. 
   - 데이터가 특정 분포를 따른다고 가정하여, 학습해야 하는 모델의 파라미터의 종류와 수가 정해져 있다.
   - 우선 모델의 형태를 정하고, 이 모델의 파라미터를 학습을 통해서 발전시켜나가는 방식으로 알고리즘이 진행된다.
- Non-parametric model : The number of parameters grow with the amount of training data. 
    - 데이터가 특정 분포를 따른다는 가정을 하지 않아, 학습해야 하는 파라미터의 종류와 수가 학습 데이터의 형태와 크기에 따라 달라진다. 
    - 데이터에 대한 사전 지식이 없을 때 유용하게 사용된다.
    - 선형성 안 보일 때 (OLS 사용시 선형성, 정규성 가정이 만족 되지 않을 때) 
    - PSM도 OLS를 쓸 수 없음. 
    - SEM(structural equation modeling)은 선형성, 분포 가정이 많이 들어가지만 DAG 형태로 생긴 인과 모델들이 Non-parametric이 많다 (분포 형태를 가정하지 않음)
    - ATE를 구할건데 Parametric 한 것은 아님.




```python
import statsmodels

# Identify the causal effect : 아까 위에서 명시한 model 을 활용합니다.
identified_estimand = model.identify_effect(proceed_when_unidentifiable = True)
print(identified_estimand)
```

    WARNING:dowhy.causal_identifier:If this is observed data (not from a randomized experiment), there might always be missing confounders. Causal effect cannot be identified perfectly.
    INFO:dowhy.causal_identifier:Continuing by ignoring these unobserved confounders because proceed_when_unidentifiable flag is True.
    INFO:dowhy.causal_identifier:Instrumental variables for treatment and outcome:[]
    INFO:dowhy.causal_identifier:Frontdoor variables for treatment and outcome:[]


    Estimand type: nonparametric-ate
    
    ### Estimand : 1
    Estimand name: backdoor1 (Default)
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,le
                                                                                  
    
                                                                                  
    ad_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_
                                                                                  
    
                        
    car_parking_spaces))
                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 2
    Estimand name: backdoor2
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,le
                                                                                  
    
                                                                                  
    ad_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking
                                                                                  
    
             
    _spaces))
             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 3
    Estimand name: backdoor3
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,le
                                                                                  
    
                                                                                 
    ad_time,meal,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 4
    Estimand name: backdoor4
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,le
                                                                                  
    
                                                                                  
    ad_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_p
                                                                                  
    
                   
    arking_spaces))
                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 5
    Estimand name: backdoor5
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,me
                                                                                  
    
                                                                                  
    al,days_in_waiting_list,country,booking_changes,total_stay,required_car_parkin
                                                                                  
    
              
    g_spaces))
              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 6
    Estimand name: backdoor6
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time
                                                                                  
    
                                                                                  
    ,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_par
                                                                                  
    
                 
    king_spaces))
                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 7
    Estimand name: backdoor7
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,days_
                                                                                  
    
                                                                                  
    in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces
                                                                                  
    
      
    ))
      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 8
    Estimand name: backdoor8
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiti
                                                                                  
    
                                                                            
    ng_list,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 9
    Estimand name: backdoor9
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_tim
                                                                                  
    
                                                                                  
    e,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_pa
                                                                                  
    
                  
    rking_spaces))
                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 10
    Estimand name: backdoor10
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_li
                                                                                  
    
                                                                       
    st,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 11
    Estimand name: backdoor11
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,le
                                                                                  
    
                                                                      
    ad_time,meal,country,booking_changes,required_car_parking_spaces))
                                                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 12
    Estimand name: backdoor12
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,le
                                                                                  
    
                                                                                  
    ad_time,days_in_waiting_list,country,booking_changes,required_car_parking_spac
                                                                                  
    
        
    es))
        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 13
    Estimand name: backdoor13
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,le
                                                                                  
    
                                                                            
    ad_time,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 14
    Estimand name: backdoor14
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,me
                                                                                  
    
                                                                                 
    al,days_in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 15
    Estimand name: backdoor15
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,me
                                                                                  
    
                                                                       
    al,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 16
    Estimand name: backdoor16
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,da
                                                                                  
    
                                                                                  
    ys_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spa
                                                                                  
    
         
    ces))
         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 17
    Estimand name: backdoor17
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time
                                                                                  
    
                                                                                  
    ,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces
                                                                                  
    
      
    ))
      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 18
    Estimand name: backdoor18
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time
                                                                                  
    
                                                                          
    ,meal,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 19
    Estimand name: backdoor19
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time
                                                                                  
    
                                                                                  
    ,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_
                                                                                  
    
            
    spaces))
            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 20
    Estimand name: backdoor20
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,days
                                                                                  
    
                                                                                  
    _in_waiting_list,country,booking_changes,total_stay,required_car_parking_space
                                                                                  
    
       
    s))
       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 21
    Estimand name: backdoor21
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,days_
                                                                                  
    
                                                                         
    in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 22
    Estimand name: backdoor22
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,count
                                                                                  
    
                                                               
    ry,booking_changes,total_stay,required_car_parking_spaces))
                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 23
    Estimand name: backdoor23
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,lead_time,days_in_wa
                                                                                  
    
                                                                               
    iting_list,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 24
    Estimand name: backdoor24
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,meal,days_in_waiting
                                                                                  
    
                                                                          
    _list,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 25
    Estimand name: backdoor25
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,lead_time,meal,days_in_wait
                                                                                  
    
                                                                             
    ing_list,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 26
    Estimand name: backdoor26
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiti
                                                                                  
    
                                                                 
    ng_list,country,booking_changes,required_car_parking_spaces))
                                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 27
    Estimand name: backdoor27
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,country,booki
                                                                                  
    
                                                       
    ng_changes,total_stay,required_car_parking_spaces))
                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 28
    Estimand name: backdoor28
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_li
                                                                                  
    
                                                                       
    st,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 29
    Estimand name: backdoor29
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,co
                                                                                  
    
                                                                  
    untry,booking_changes,total_stay,required_car_parking_spaces))
                                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 30
    Estimand name: backdoor30
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list
                                                                                  
    
                                                                     
    ,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 31
    Estimand name: backdoor31
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,lead_time,meal,days_in_waiting_list,country,bo
                                                                                  
    
                                                          
    oking_changes,total_stay,required_car_parking_spaces))
                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 32
    Estimand name: backdoor32
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_tim
                                                                                  
    
                                                                                  
    e,meal,days_in_waiting_list,country,booking_changes,required_car_parking_space
                                                                                  
    
       
    s))
       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 33
    Estimand name: backdoor33
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_tim
                                                                                  
    
                                                                           
    e,meal,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 34
    Estimand name: backdoor34
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_tim
                                                                                  
    
                                                                                  
    e,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking
                                                                                  
    
             
    _spaces))
             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 35
    Estimand name: backdoor35
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,day
                                                                                  
    
                                                                                  
    s_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spac
                                                                                  
    
        
    es))
        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 36
    Estimand name: backdoor36
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,
                                                                                  
    
                                                                                  
    days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_s
                                                                                  
    
           
    paces))
           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 37
    Estimand name: backdoor37
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_wai
                                                                                  
    
                                                                              
    ting_list,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 38
    Estimand name: backdoor38
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_lis
                                                                                  
    
                                                                      
    t,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 39
    Estimand name: backdoor39
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_li
                                                                                  
    
                                                            
    st,country,booking_changes,required_car_parking_spaces))
                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 40
    Estimand name: backdoor40
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_ch
                                                                                  
    
                                                  
    anges,total_stay,required_car_parking_spaces))
                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 41
    Estimand name: backdoor41
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,co
                                                                                  
    
                                                                  
    untry,booking_changes,total_stay,required_car_parking_spaces))
                                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 42
    Estimand name: backdoor42
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country
                                                                                  
    
                                                             
    ,booking_changes,total_stay,required_car_parking_spaces))
                                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 43
    Estimand name: backdoor43
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,coun
                                                                                  
    
                                                                
    try,booking_changes,total_stay,required_car_parking_spaces))
                                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 44
    Estimand name: backdoor44
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking
                                                                                  
    
                                                     
    _changes,total_stay,required_car_parking_spaces))
                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 45
    Estimand name: backdoor45
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes
                                                                                  
    
                                             
    ,total_stay,required_car_parking_spaces))
                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 46
    Estimand name: backdoor46
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,cou
                                                                                  
    
                                                                 
    ntry,booking_changes,total_stay,required_car_parking_spaces))
                                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 47
    Estimand name: backdoor47
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,le
                                                                                  
    
                                                                 
    ad_time,country,booking_changes,required_car_parking_spaces))
                                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 48
    Estimand name: backdoor48
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,me
                                                                                  
    
                                                            
    al,country,booking_changes,required_car_parking_spaces))
                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 49
    Estimand name: backdoor49
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,da
                                                                                  
    
                                                                            
    ys_in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 50
    Estimand name: backdoor50
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,co
                                                                                  
    
                                                                  
    untry,booking_changes,total_stay,required_car_parking_spaces))
                                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 51
    Estimand name: backdoor51
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time
                                                                                  
    
                                                               
    ,meal,country,booking_changes,required_car_parking_spaces))
                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 52
    Estimand name: backdoor52
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time
                                                                                  
    
                                                                               
    ,days_in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 53
    Estimand name: backdoor53
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time
                                                                                  
    
                                                                     
    ,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 54
    Estimand name: backdoor54
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,days
                                                                                  
    
                                                                          
    _in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 55
    Estimand name: backdoor55
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,coun
                                                                                  
    
                                                                
    try,booking_changes,total_stay,required_car_parking_spaces))
                                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 56
    Estimand name: backdoor56
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,days_in_w
                                                                                  
    
                                                                                
    aiting_list,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 57
    Estimand name: backdoor57
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,count
                                                                                  
    
                                                    
    ry,booking_changes,required_car_parking_spaces))
                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 58
    Estimand name: backdoor58
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,lead_time,days_in_wa
                                                                                  
    
                                                                    
    iting_list,country,booking_changes,required_car_parking_spaces))
                                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 59
    Estimand name: backdoor59
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,lead_time,country,bo
                                                                                  
    
                                                          
    oking_changes,total_stay,required_car_parking_spaces))
                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 60
    Estimand name: backdoor60
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,meal,days_in_waiting
                                                                                  
    
                                                               
    _list,country,booking_changes,required_car_parking_spaces))
                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 61
    Estimand name: backdoor61
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,meal,country,booking
                                                                                  
    
                                                     
    _changes,total_stay,required_car_parking_spaces))
                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 62
    Estimand name: backdoor62
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,days_in_waiting_list
                                                                                  
    
                                                                     
    ,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 63
    Estimand name: backdoor63
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,lead_time,meal,days_in_wait
                                                                                  
    
                                                                  
    ing_list,country,booking_changes,required_car_parking_spaces))
                                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 64
    Estimand name: backdoor64
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,lead_time,meal,country,book
                                                                                  
    
                                                        
    ing_changes,total_stay,required_car_parking_spaces))
                                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 65
    Estimand name: backdoor65
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,lead_time,days_in_waiting_l
                                                                                  
    
                                                                        
    ist,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 66
    Estimand name: backdoor66
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,meal,days_in_waiting_list,c
                                                                                  
    
                                                                   
    ountry,booking_changes,total_stay,required_car_parking_spaces))
                                                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 67
    Estimand name: backdoor67
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,country,booki
                                                                                  
    
                                            
    ng_changes,required_car_parking_spaces))
                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 68
    Estimand name: backdoor68
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_li
                                                                                  
    
                                                            
    st,country,booking_changes,required_car_parking_spaces))
                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 69
    Estimand name: backdoor69
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,lead_time,country,booking_ch
                                                                                  
    
                                                  
    anges,total_stay,required_car_parking_spaces))
                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 70
    Estimand name: backdoor70
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,co
                                                                                  
    
                                                       
    untry,booking_changes,required_car_parking_spaces))
                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 71
    Estimand name: backdoor71
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,meal,country,booking_changes
                                                                                  
    
                                             
    ,total_stay,required_car_parking_spaces))
                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 72
    Estimand name: backdoor72
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,days_in_waiting_list,country
                                                                                  
    
                                                             
    ,booking_changes,total_stay,required_car_parking_spaces))
                                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 73
    Estimand name: backdoor73
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list
                                                                                  
    
                                                          
    ,country,booking_changes,required_car_parking_spaces))
                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 74
    Estimand name: backdoor74
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,lead_time,meal,country,booking_chan
                                                                                  
    
                                                
    ges,total_stay,required_car_parking_spaces))
                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 75
    Estimand name: backdoor75
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,coun
                                                                                  
    
                                                                
    try,booking_changes,total_stay,required_car_parking_spaces))
                                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 76
    Estimand name: backdoor76
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,b
                                                                                  
    
                                                           
    ooking_changes,total_stay,required_car_parking_spaces))
                                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 77
    Estimand name: backdoor77
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,lead_time,meal,days_in_waiting_list,country,bo
                                                                                  
    
                                               
    oking_changes,required_car_parking_spaces))
                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 78
    Estimand name: backdoor78
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,lead_time,meal,country,booking_changes,total_s
                                                                                  
    
                                     
    tay,required_car_parking_spaces))
                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 79
    Estimand name: backdoor79
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,lead_time,days_in_waiting_list,country,booking
                                                                                  
    
                                                     
    _changes,total_stay,required_car_parking_spaces))
                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 80
    Estimand name: backdoor80
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,meal,days_in_waiting_list,country,booking_chan
                                                                                  
    
                                                
    ges,total_stay,required_car_parking_spaces))
                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 81
    Estimand name: backdoor81
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,lead_time,meal,days_in_waiting_list,country,booking_c
                                                                                  
    
                                                   
    hanges,total_stay,required_car_parking_spaces))
                                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 82
    Estimand name: backdoor82
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_tim
                                                                                  
    
                                                                
    e,meal,country,booking_changes,required_car_parking_spaces))
                                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 83
    Estimand name: backdoor83
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_tim
                                                                                  
    
                                                                                
    e,days_in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 84
    Estimand name: backdoor84
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_tim
                                                                                  
    
                                                                      
    e,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 85
    Estimand name: backdoor85
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,day
                                                                                  
    
                                                                           
    s_in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 86
    Estimand name: backdoor86
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,cou
                                                                                  
    
                                                                 
    ntry,booking_changes,total_stay,required_car_parking_spaces))
                                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 87
    Estimand name: backdoor87
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_
                                                                                  
    
                                                                                 
    waiting_list,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 88
    Estimand name: backdoor88
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,
                                                                                  
    
                                                                              
    days_in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 89
    Estimand name: backdoor89
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,
                                                                                  
    
                                                                    
    country,booking_changes,total_stay,required_car_parking_spaces))
                                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 90
    Estimand name: backdoor90
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_
                                                                                  
    
                                                                                  
    in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces
                                                                                  
    
      
    ))
      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 91
    Estimand name: backdoor91
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_wa
                                                                                  
    
                                                                               
    iting_list,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 92
    Estimand name: backdoor92
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_wai
                                                                                  
    
                                                                   
    ting_list,country,booking_changes,required_car_parking_spaces))
                                                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 93
    Estimand name: backdoor93
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,lead_time,meal,country,boo
                                                                                  
    
                                                         
    king_changes,total_stay,required_car_parking_spaces))
                                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 94
    Estimand name: backdoor94
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_
                                                                                  
    
                                                                         
    list,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 95
    Estimand name: backdoor95
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,
                                                                                  
    
                                                                    
    country,booking_changes,total_stay,required_car_parking_spaces))
                                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 96
    Estimand name: backdoor96
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_li
                                                                                  
    
                                                                       
    st,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 97
    Estimand name: backdoor97
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_lis
                                                                                  
    
                                                           
    t,country,booking_changes,required_car_parking_spaces))
                                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 98
    Estimand name: backdoor98
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_cha
                                                                                  
    
                                                 
    nges,total_stay,required_car_parking_spaces))
                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 99
    Estimand name: backdoor99
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,cou
                                                                                  
    
                                                                 
    ntry,booking_changes,total_stay,required_car_parking_spaces))
                                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 100
    Estimand name: backdoor100
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,
                                                                                  
    
                                                            
    booking_changes,total_stay,required_car_parking_spaces))
                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 101
    Estimand name: backdoor101
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,count
                                                                                  
    
                                                               
    ry,booking_changes,total_stay,required_car_parking_spaces))
                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 102
    Estimand name: backdoor102
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_
                                                                                  
    
                                                    
    changes,total_stay,required_car_parking_spaces))
                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 103
    Estimand name: backdoor103
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_ch
                                                                                  
    
                                       
    anges,required_car_parking_spaces))
                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 104
    Estimand name: backdoor104
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,co
                                                                                  
    
                                                       
    untry,booking_changes,required_car_parking_spaces))
                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 105
    Estimand name: backdoor105
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes
                                                                                  
    
                                             
    ,total_stay,required_car_parking_spaces))
                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 106
    Estimand name: backdoor106
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country
                                                                                  
    
                                                  
    ,booking_changes,required_car_parking_spaces))
                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 107
    Estimand name: backdoor107
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,tota
                                                                                  
    
                                        
    l_stay,required_car_parking_spaces))
                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 108
    Estimand name: backdoor108
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,book
                                                                                  
    
                                                        
    ing_changes,total_stay,required_car_parking_spaces))
                                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 109
    Estimand name: backdoor109
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,coun
                                                                                  
    
                                                     
    try,booking_changes,required_car_parking_spaces))
                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 110
    Estimand name: backdoor110
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,t
                                                                                  
    
                                           
    otal_stay,required_car_parking_spaces))
                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 111
    Estimand name: backdoor111
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,b
                                                                                  
    
                                                           
    ooking_changes,total_stay,required_car_parking_spaces))
                                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 112
    Estimand name: backdoor112
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,bookin
                                                                                  
    
                                                      
    g_changes,total_stay,required_car_parking_spaces))
                                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 113
    Estimand name: backdoor113
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking
                                                                                  
    
                                          
    _changes,required_car_parking_spaces))
                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 114
    Estimand name: backdoor114
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,r
                                                                                  
    
                                
    equired_car_parking_spaces))
                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 115
    Estimand name: backdoor115
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_chan
                                                                                  
    
                                                
    ges,total_stay,required_car_parking_spaces))
                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 116
    Estimand name: backdoor116
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,t
                                                                                  
    
                                           
    otal_stay,required_car_parking_spaces))
                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 117
    Estimand name: backdoor117
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_change
                                                                                  
    
                                              
    s,total_stay,required_car_parking_spaces))
                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 118
    Estimand name: backdoor118
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes
                                                                                  
    
                                  
    ,required_car_parking_spaces))
                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 119
    Estimand name: backdoor119
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_
                                                                                  
    
                        
    car_parking_spaces))
                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 120
    Estimand name: backdoor120
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,tota
                                                                                  
    
                                        
    l_stay,required_car_parking_spaces))
                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 121
    Estimand name: backdoor121
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_sta
                                                                                  
    
                                   
    y,required_car_parking_spaces))
                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 122
    Estimand name: backdoor122
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_
                                                                                  
    
                                      
    stay,required_car_parking_spaces))
                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 123
    Estimand name: backdoor123
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                                  
    ,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,requir
                                                                                  
    
                           
    ed_car_parking_spaces))
                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 124
    Estimand name: backdoor124
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,cou
                                                                                  
    
                                                      
    ntry,booking_changes,required_car_parking_spaces))
                                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 125
    Estimand name: backdoor125
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,
                                                                                  
    
                                            
    total_stay,required_car_parking_spaces))
                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 126
    Estimand name: backdoor126
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,
                                                                                  
    
                                                            
    booking_changes,total_stay,required_car_parking_spaces))
                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 127
    Estimand name: backdoor127
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booki
                                                                                  
    
                                                       
    ng_changes,total_stay,required_car_parking_spaces))
                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 128
    Estimand name: backdoor128
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,bo
                                                                                  
    
                                                          
    oking_changes,total_stay,required_car_parking_spaces))
                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 129
    Estimand name: backdoor129
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_chang
                                                                                  
    
                                               
    es,total_stay,required_car_parking_spaces))
                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 130
    Estimand name: backdoor130
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total
                                                                                  
    
                                       
    _stay,required_car_parking_spaces))
                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 131
    Estimand name: backdoor131
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,co
                                                                                  
    
                                                       
    untry,booking_changes,required_car_parking_spaces))
                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 132
    Estimand name: backdoor132
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time
                                                                                  
    
                                                          
    ,country,booking_changes,required_car_parking_spaces))
                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 133
    Estimand name: backdoor133
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,coun
                                                                                  
    
                                                     
    try,booking_changes,required_car_parking_spaces))
                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 134
    Estimand name: backdoor134
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,days_in_w
                                                                                  
    
                                                                     
    aiting_list,country,booking_changes,required_car_parking_spaces))
                                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 135
    Estimand name: backdoor135
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,country,b
                                                                                  
    
                                                           
    ooking_changes,total_stay,required_car_parking_spaces))
                                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 136
    Estimand name: backdoor136
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,lead_time,country,bo
                                                                                  
    
                                               
    oking_changes,required_car_parking_spaces))
                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 137
    Estimand name: backdoor137
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,meal,country,booking
                                                                                  
    
                                          
    _changes,required_car_parking_spaces))
                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 138
    Estimand name: backdoor138
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,days_in_waiting_list
                                                                                  
    
                                                          
    ,country,booking_changes,required_car_parking_spaces))
                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 139
    Estimand name: backdoor139
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,country,booking_chan
                                                                                  
    
                                                
    ges,total_stay,required_car_parking_spaces))
                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 140
    Estimand name: backdoor140
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,lead_time,meal,country,book
                                                                                  
    
                                             
    ing_changes,required_car_parking_spaces))
                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 141
    Estimand name: backdoor141
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,lead_time,days_in_waiting_l
                                                                                  
    
                                                             
    ist,country,booking_changes,required_car_parking_spaces))
                                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 142
    Estimand name: backdoor142
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,lead_time,country,booking_c
                                                                                  
    
                                                   
    hanges,total_stay,required_car_parking_spaces))
                                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 143
    Estimand name: backdoor143
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,meal,days_in_waiting_list,c
                                                                                  
    
                                                        
    ountry,booking_changes,required_car_parking_spaces))
                                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 144
    Estimand name: backdoor144
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,meal,country,booking_change
                                                                                  
    
                                              
    s,total_stay,required_car_parking_spaces))
                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 145
    Estimand name: backdoor145
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,days_in_waiting_list,countr
                                                                                  
    
                                                              
    y,booking_changes,total_stay,required_car_parking_spaces))
                                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 146
    Estimand name: backdoor146
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,lead_time,country,booking_ch
                                                                                  
    
                                       
    anges,required_car_parking_spaces))
                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 147
    Estimand name: backdoor147
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,meal,country,booking_changes
                                                                                  
    
                                  
    ,required_car_parking_spaces))
                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 148
    Estimand name: backdoor148
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,days_in_waiting_list,country
                                                                                  
    
                                                  
    ,booking_changes,required_car_parking_spaces))
                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 149
    Estimand name: backdoor149
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,country,booking_changes,tota
                                                                                  
    
                                        
    l_stay,required_car_parking_spaces))
                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 150
    Estimand name: backdoor150
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,lead_time,meal,country,booking_chan
                                                                                  
    
                                     
    ges,required_car_parking_spaces))
                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 151
    Estimand name: backdoor151
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,coun
                                                                                  
    
                                                     
    try,booking_changes,required_car_parking_spaces))
                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 152
    Estimand name: backdoor152
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,lead_time,country,booking_changes,t
                                                                                  
    
                                           
    otal_stay,required_car_parking_spaces))
                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 153
    Estimand name: backdoor153
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,b
                                                                                  
    
                                                
    ooking_changes,required_car_parking_spaces))
                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 154
    Estimand name: backdoor154
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,meal,country,booking_changes,total_
                                                                                  
    
                                      
    stay,required_car_parking_spaces))
                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 155
    Estimand name: backdoor155
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,days_in_waiting_list,country,bookin
                                                                                  
    
                                                      
    g_changes,total_stay,required_car_parking_spaces))
                                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 156
    Estimand name: backdoor156
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,lead_time,meal,country,booking_changes,require
                                                                                  
    
                          
    d_car_parking_spaces))
                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 157
    Estimand name: backdoor157
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,lead_time,days_in_waiting_list,country,booking
                                                                                  
    
                                          
    _changes,required_car_parking_spaces))
                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 158
    Estimand name: backdoor158
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,lead_time,country,booking_changes,total_stay,r
                                                                                  
    
                                
    equired_car_parking_spaces))
                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 159
    Estimand name: backdoor159
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,meal,days_in_waiting_list,country,booking_chan
                                                                                  
    
                                     
    ges,required_car_parking_spaces))
                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 160
    Estimand name: backdoor160
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,meal,country,booking_changes,total_stay,requir
                                                                                  
    
                           
    ed_car_parking_spaces))
                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 161
    Estimand name: backdoor161
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,days_in_waiting_list,country,booking_changes,t
                                                                                  
    
                                           
    otal_stay,required_car_parking_spaces))
                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 162
    Estimand name: backdoor162
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,lead_time,meal,days_in_waiting_list,country,booking_c
                                                                                  
    
                                        
    hanges,required_car_parking_spaces))
                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 163
    Estimand name: backdoor163
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,lead_time,meal,country,booking_changes,total_stay,req
                                                                                  
    
                              
    uired_car_parking_spaces))
                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 164
    Estimand name: backdoor164
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,lead_time,days_in_waiting_list,country,booking_change
                                                                                  
    
                                              
    s,total_stay,required_car_parking_spaces))
                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 165
    Estimand name: backdoor165
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,meal,days_in_waiting_list,country,booking_changes,tot
                                                                                  
    
                                         
    al_stay,required_car_parking_spaces))
                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 166
    Estimand name: backdoor166
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_tim
                                                                                  
    
                                                           
    e,country,booking_changes,required_car_parking_spaces))
                                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 167
    Estimand name: backdoor167
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,cou
                                                                                  
    
                                                      
    ntry,booking_changes,required_car_parking_spaces))
                                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 168
    Estimand name: backdoor168
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_
                                                                                  
    
                                                                      
    waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 169
    Estimand name: backdoor169
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,country,
                                                                                  
    
                                                            
    booking_changes,total_stay,required_car_parking_spaces))
                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 170
    Estimand name: backdoor170
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,
                                                                                  
    
                                                         
    country,booking_changes,required_car_parking_spaces))
                                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 171
    Estimand name: backdoor171
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_
                                                                                  
    
                                                                         
    in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 172
    Estimand name: backdoor172
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,lead_time,count
                                                                                  
    
                                                               
    ry,booking_changes,total_stay,required_car_parking_spaces))
                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 173
    Estimand name: backdoor173
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_wa
                                                                                  
    
                                                                    
    iting_list,country,booking_changes,required_car_parking_spaces))
                                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 174
    Estimand name: backdoor174
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,meal,country,bo
                                                                                  
    
                                                          
    oking_changes,total_stay,required_car_parking_spaces))
                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 175
    Estimand name: backdoor175
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting
                                                                                  
    
                                                                          
    _list,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 176
    Estimand name: backdoor176
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,lead_time,meal,country,boo
                                                                                  
    
                                              
    king_changes,required_car_parking_spaces))
                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 177
    Estimand name: backdoor177
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_
                                                                                  
    
                                                              
    list,country,booking_changes,required_car_parking_spaces))
                                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 178
    Estimand name: backdoor178
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,lead_time,country,booking_
                                                                                  
    
                                                    
    changes,total_stay,required_car_parking_spaces))
                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 179
    Estimand name: backdoor179
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,
                                                                                  
    
                                                         
    country,booking_changes,required_car_parking_spaces))
                                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 180
    Estimand name: backdoor180
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,meal,country,booking_chang
                                                                                  
    
                                               
    es,total_stay,required_car_parking_spaces))
                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 181
    Estimand name: backdoor181
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,days_in_waiting_list,count
                                                                                  
    
                                                               
    ry,booking_changes,total_stay,required_car_parking_spaces))
                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 182
    Estimand name: backdoor182
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_li
                                                                                  
    
                                                            
    st,country,booking_changes,required_car_parking_spaces))
                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 183
    Estimand name: backdoor183
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,lead_time,meal,country,booking_ch
                                                                                  
    
                                                  
    anges,total_stay,required_car_parking_spaces))
                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 184
    Estimand name: backdoor184
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,co
                                                                                  
    
                                                                  
    untry,booking_changes,total_stay,required_car_parking_spaces))
                                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 185
    Estimand name: backdoor185
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,meal,days_in_waiting_list,country
                                                                                  
    
                                                             
    ,booking_changes,total_stay,required_car_parking_spaces))
                                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 186
    Estimand name: backdoor186
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_cha
                                                                                  
    
                                      
    nges,required_car_parking_spaces))
                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 187
    Estimand name: backdoor187
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,cou
                                                                                  
    
                                                      
    ntry,booking_changes,required_car_parking_spaces))
                                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 188
    Estimand name: backdoor188
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,
                                                                                  
    
                                            
    total_stay,required_car_parking_spaces))
                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 189
    Estimand name: backdoor189
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,
                                                                                  
    
                                                 
    booking_changes,required_car_parking_spaces))
                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 190
    Estimand name: backdoor190
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total
                                                                                  
    
                                       
    _stay,required_car_parking_spaces))
                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 191
    Estimand name: backdoor191
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booki
                                                                                  
    
                                                       
    ng_changes,total_stay,required_car_parking_spaces))
                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 192
    Estimand name: backdoor192
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,count
                                                                                  
    
                                                    
    ry,booking_changes,required_car_parking_spaces))
                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 193
    Estimand name: backdoor193
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,to
                                                                                  
    
                                          
    tal_stay,required_car_parking_spaces))
                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 194
    Estimand name: backdoor194
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,bo
                                                                                  
    
                                                          
    oking_changes,total_stay,required_car_parking_spaces))
                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 195
    Estimand name: backdoor195
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking
                                                                                  
    
                                                     
    _changes,total_stay,required_car_parking_spaces))
                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 196
    Estimand name: backdoor196
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_
                                                                                  
    
                                         
    changes,required_car_parking_spaces))
                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 197
    Estimand name: backdoor197
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,re
                                                                                  
    
                               
    quired_car_parking_spaces))
                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 198
    Estimand name: backdoor198
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,lead_time,days_in_waiting_list,country,booking_chang
                                                                                  
    
                                               
    es,total_stay,required_car_parking_spaces))
                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 199
    Estimand name: backdoor199
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,to
                                                                                  
    
                                          
    tal_stay,required_car_parking_spaces))
                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 200
    Estimand name: backdoor200
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes
                                                                                  
    
                                             
    ,total_stay,required_car_parking_spaces))
                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 201
    Estimand name: backdoor201
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes
                                                                                  
    
                                  
    ,required_car_parking_spaces))
                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 202
    Estimand name: backdoor202
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,requ
                                                                                  
    
                             
    ired_car_parking_spaces))
                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 203
    Estimand name: backdoor203
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,book
                                                                                  
    
                                             
    ing_changes,required_car_parking_spaces))
                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 204
    Estimand name: backdoor204
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,country,booking_changes,total_sta
                                                                                  
    
                                   
    y,required_car_parking_spaces))
                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 205
    Estimand name: backdoor205
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,r
                                                                                  
    
                                
    equired_car_parking_spaces))
                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 206
    Estimand name: backdoor206
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,b
                                                                                  
    
                                                
    ooking_changes,required_car_parking_spaces))
                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 207
    Estimand name: backdoor207
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_
                                                                                  
    
                                      
    stay,required_car_parking_spaces))
                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 208
    Estimand name: backdoor208
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,bookin
                                                                                  
    
                                           
    g_changes,required_car_parking_spaces))
                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 209
    Estimand name: backdoor209
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,
                                                                                  
    
                                 
    required_car_parking_spaces))
                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 210
    Estimand name: backdoor210
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_cha
                                                                                  
    
                                                 
    nges,total_stay,required_car_parking_spaces))
                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 211
    Estimand name: backdoor211
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,lead_time,meal,country,booking_changes,required_car
                                                                                  
    
                     
    _parking_spaces))
                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 212
    Estimand name: backdoor212
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_chan
                                                                                  
    
                                     
    ges,required_car_parking_spaces))
                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 213
    Estimand name: backdoor213
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,lead_time,country,booking_changes,total_stay,requir
                                                                                  
    
                           
    ed_car_parking_spaces))
                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 214
    Estimand name: backdoor214
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,r
                                                                                  
    
                                
    equired_car_parking_spaces))
                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 215
    Estimand name: backdoor215
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,meal,country,booking_changes,total_stay,required_ca
                                                                                  
    
                      
    r_parking_spaces))
                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 216
    Estimand name: backdoor216
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,days_in_waiting_list,country,booking_changes,total_
                                                                                  
    
                                      
    stay,required_car_parking_spaces))
                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 217
    Estimand name: backdoor217
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_change
                                                                                  
    
                                   
    s,required_car_parking_spaces))
                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 218
    Estimand name: backdoor218
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,lead_time,meal,country,booking_changes,total_stay,required
                                                                                  
    
                         
    _car_parking_spaces))
                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 219
    Estimand name: backdoor219
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,tot
                                                                                  
    
                                         
    al_stay,required_car_parking_spaces))
                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 220
    Estimand name: backdoor220
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,meal,days_in_waiting_list,country,booking_changes,total_st
                                                                                  
    
                                    
    ay,required_car_parking_spaces))
                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 221
    Estimand name: backdoor221
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking
                                                                                  
    
             
    _spaces))
             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 222
    Estimand name: backdoor222
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,requ
                                                                                  
    
                             
    ired_car_parking_spaces))
                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 223
    Estimand name: backdoor223
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_p
                                                                                  
    
                   
    arking_spaces))
                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 224
    Estimand name: backdoor224
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_
                                                                                  
    
                        
    car_parking_spaces))
                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 225
    Estimand name: backdoor225
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,meal,country,booking_changes,total_stay,required_car_parkin
                                                                                  
    
              
    g_spaces))
              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 226
    Estimand name: backdoor226
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,req
                                                                                  
    
                              
    uired_car_parking_spaces))
                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 227
    Estimand name: backdoor227
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,requir
                                                                                  
    
                           
    ed_car_parking_spaces))
                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 228
    Estimand name: backdoor228
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_par
                                                                                  
    
                 
    king_spaces))
                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 229
    Estimand name: backdoor229
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,
                                                                                  
    
                                 
    required_car_parking_spaces))
                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 230
    Estimand name: backdoor230
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,requi
                                                                                  
    
                            
    red_car_parking_spaces))
                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 231
    Estimand name: backdoor231
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                                  
    ,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_park
                                                                                  
    
                
    ing_spaces))
                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 232
    Estimand name: backdoor232
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                                  
    ,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces
                                                                                  
    
      
    ))
      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 233
    Estimand name: backdoor233
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                                  
    ,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_ca
                                                                                  
    
                      
    r_parking_spaces))
                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 234
    Estimand name: backdoor234
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                                  
    ,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_par
                                                                                  
    
                 
    king_spaces))
                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 235
    Estimand name: backdoor235
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,lead_t
    d[different_room_assigned]                                                    
    
                                                                                  
    ime,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_
                                                                                  
    
                    
    parking_spaces))
                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 236
    Estimand name: backdoor236
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,
                                                                                  
    
                                 
    required_car_parking_spaces))
                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 237
    Estimand name: backdoor237
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,
                                                                                  
    
                                                 
    booking_changes,required_car_parking_spaces))
                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 238
    Estimand name: backdoor238
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total
                                                                                  
    
                                       
    _stay,required_car_parking_spaces))
                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 239
    Estimand name: backdoor239
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booki
                                                                                  
    
                                            
    ng_changes,required_car_parking_spaces))
                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 240
    Estimand name: backdoor240
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay
                                                                                  
    
                                  
    ,required_car_parking_spaces))
                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 241
    Estimand name: backdoor241
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_ch
                                                                                  
    
                                                  
    anges,total_stay,required_car_parking_spaces))
                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 242
    Estimand name: backdoor242
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,bo
                                                                                  
    
                                               
    oking_changes,required_car_parking_spaces))
                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 243
    Estimand name: backdoor243
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_s
                                                                                  
    
                                     
    tay,required_car_parking_spaces))
                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 244
    Estimand name: backdoor244
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking
                                                                                  
    
                                                     
    _changes,total_stay,required_car_parking_spaces))
                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 245
    Estimand name: backdoor245
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_chan
                                                                                  
    
                                                
    ges,total_stay,required_car_parking_spaces))
                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 246
    Estimand name: backdoor246
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_chang
                                                                                  
    
                                    
    es,required_car_parking_spaces))
                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 247
    Estimand name: backdoor247
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,lead_time,meal,country,booking_changes,total_stay,require
                                                                                  
    
                          
    d_car_parking_spaces))
                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 248
    Estimand name: backdoor248
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,to
                                                                                  
    
                                          
    tal_stay,required_car_parking_spaces))
                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 249
    Estimand name: backdoor249
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_s
                                                                                  
    
                                     
    tay,required_car_parking_spaces))
                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 250
    Estimand name: backdoor250
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,tota
                                                                                  
    
                                        
    l_stay,required_car_parking_spaces))
                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 251
    Estimand name: backdoor251
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,requi
                                                                                  
    
                            
    red_car_parking_spaces))
                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 252
    Estimand name: backdoor252
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_pa
                                                                                  
    
                  
    rking_spaces))
                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 253
    Estimand name: backdoor253
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay
                                                                                  
    
                                  
    ,required_car_parking_spaces))
                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 254
    Estimand name: backdoor254
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,requ
                                                                                  
    
                             
    ired_car_parking_spaces))
                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 255
    Estimand name: backdoor255
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,r
                                                                                  
    
                                
    equired_car_parking_spaces))
                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 256
    Estimand name: backdoor256
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,lead_
    d[different_room_assigned]                                                    
    
                                                                                  
    time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car
                                                                                  
    
                     
    _parking_spaces))
                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 257
    Estimand name: backdoor257
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,is_repeated_guest,country,b
                                                                                  
    
                                                
    ooking_changes,required_car_parking_spaces))
                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 258
    Estimand name: backdoor258
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,guests,country,booking_chan
                                                                                  
    
                                     
    ges,required_car_parking_spaces))
                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 259
    Estimand name: backdoor259
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,lead_time,country,booking_c
                                                                                  
    
                                        
    hanges,required_car_parking_spaces))
                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 260
    Estimand name: backdoor260
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,meal,country,booking_change
                                                                                  
    
                                   
    s,required_car_parking_spaces))
                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 261
    Estimand name: backdoor261
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,days_in_waiting_list,countr
                                                                                  
    
                                                   
    y,booking_changes,required_car_parking_spaces))
                                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 262
    Estimand name: backdoor262
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,country,booking_changes,tot
                                                                                  
    
                                         
    al_stay,required_car_parking_spaces))
                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 263
    Estimand name: backdoor263
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,guests,country,booking_changes,requ
                                                                                  
    
                             
    ired_car_parking_spaces))
                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 264
    Estimand name: backdoor264
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,lead_time,country,booking_changes,r
                                                                                  
    
                                
    equired_car_parking_spaces))
                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 265
    Estimand name: backdoor265
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,meal,country,booking_changes,requir
                                                                                  
    
                           
    ed_car_parking_spaces))
                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 266
    Estimand name: backdoor266
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,days_in_waiting_list,country,bookin
                                                                                  
    
                                           
    g_changes,required_car_parking_spaces))
                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 267
    Estimand name: backdoor267
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,country,booking_changes,total_stay,
                                                                                  
    
                                 
    required_car_parking_spaces))
                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 268
    Estimand name: backdoor268
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,lead_time,country,booking_changes,required_car
                                                                                  
    
                     
    _parking_spaces))
                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 269
    Estimand name: backdoor269
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,meal,country,booking_changes,required_car_park
                                                                                  
    
                
    ing_spaces))
                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 270
    Estimand name: backdoor270
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,days_in_waiting_list,country,booking_changes,r
                                                                                  
    
                                
    equired_car_parking_spaces))
                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 271
    Estimand name: backdoor271
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,country,booking_changes,total_stay,required_ca
                                                                                  
    
                      
    r_parking_spaces))
                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 272
    Estimand name: backdoor272
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,lead_time,meal,country,booking_changes,required_car_p
                                                                                  
    
                   
    arking_spaces))
                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 273
    Estimand name: backdoor273
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,lead_time,days_in_waiting_list,country,booking_change
                                                                                  
    
                                   
    s,required_car_parking_spaces))
                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 274
    Estimand name: backdoor274
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,lead_time,country,booking_changes,total_stay,required
                                                                                  
    
                         
    _car_parking_spaces))
                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 275
    Estimand name: backdoor275
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,meal,days_in_waiting_list,country,booking_changes,req
                                                                                  
    
                              
    uired_car_parking_spaces))
                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 276
    Estimand name: backdoor276
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,meal,country,booking_changes,total_stay,required_car_
                                                                                  
    
                    
    parking_spaces))
                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 277
    Estimand name: backdoor277
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,days_in_waiting_list,country,booking_changes,total_st
                                                                                  
    
                                    
    ay,required_car_parking_spaces))
                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 278
    Estimand name: backdoor278
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,guests,country,
                                                                                  
    
                                                 
    booking_changes,required_car_parking_spaces))
                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 279
    Estimand name: backdoor279
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,lead_time,count
                                                                                  
    
                                                    
    ry,booking_changes,required_car_parking_spaces))
                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 280
    Estimand name: backdoor280
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,meal,country,bo
                                                                                  
    
                                               
    oking_changes,required_car_parking_spaces))
                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 281
    Estimand name: backdoor281
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting
                                                                                  
    
                                                               
    _list,country,booking_changes,required_car_parking_spaces))
                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 282
    Estimand name: backdoor282
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,country,booking
                                                                                  
    
                                                     
    _changes,total_stay,required_car_parking_spaces))
                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 283
    Estimand name: backdoor283
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,lead_time,country,booking_
                                                                                  
    
                                         
    changes,required_car_parking_spaces))
                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 284
    Estimand name: backdoor284
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,meal,country,booking_chang
                                                                                  
    
                                    
    es,required_car_parking_spaces))
                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 285
    Estimand name: backdoor285
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,days_in_waiting_list,count
                                                                                  
    
                                                    
    ry,booking_changes,required_car_parking_spaces))
                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 286
    Estimand name: backdoor286
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,country,booking_changes,to
                                                                                  
    
                                          
    tal_stay,required_car_parking_spaces))
                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 287
    Estimand name: backdoor287
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,lead_time,meal,country,booking_ch
                                                                                  
    
                                       
    anges,required_car_parking_spaces))
                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 288
    Estimand name: backdoor288
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,co
                                                                                  
    
                                                       
    untry,booking_changes,required_car_parking_spaces))
                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 289
    Estimand name: backdoor289
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,lead_time,country,booking_changes
                                                                                  
    
                                             
    ,total_stay,required_car_parking_spaces))
                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 290
    Estimand name: backdoor290
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,meal,days_in_waiting_list,country
                                                                                  
    
                                                  
    ,booking_changes,required_car_parking_spaces))
                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 291
    Estimand name: backdoor291
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,meal,country,booking_changes,tota
                                                                                  
    
                                        
    l_stay,required_car_parking_spaces))
                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 292
    Estimand name: backdoor292
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,days_in_waiting_list,country,book
                                                                                  
    
                                                        
    ing_changes,total_stay,required_car_parking_spaces))
                                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 293
    Estimand name: backdoor293
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,
                                                                                  
    
                                 
    required_car_parking_spaces))
                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 294
    Estimand name: backdoor294
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,meal,country,booking_changes,requi
                                                                                  
    
                            
    red_car_parking_spaces))
                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 295
    Estimand name: backdoor295
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booki
                                                                                  
    
                                            
    ng_changes,required_car_parking_spaces))
                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 296
    Estimand name: backdoor296
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay
                                                                                  
    
                                  
    ,required_car_parking_spaces))
                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 297
    Estimand name: backdoor297
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,re
                                                                                  
    
                               
    quired_car_parking_spaces))
                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 298
    Estimand name: backdoor298
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,bo
                                                                                  
    
                                               
    oking_changes,required_car_parking_spaces))
                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 299
    Estimand name: backdoor299
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_s
                                                                                  
    
                                     
    tay,required_car_parking_spaces))
                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 300
    Estimand name: backdoor300
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking
                                                                                  
    
                                          
    _changes,required_car_parking_spaces))
                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 301
    Estimand name: backdoor301
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,r
                                                                                  
    
                                
    equired_car_parking_spaces))
                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 302
    Estimand name: backdoor302
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_chan
                                                                                  
    
                                                
    ges,total_stay,required_car_parking_spaces))
                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 303
    Estimand name: backdoor303
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,lead_time,meal,country,booking_changes,required_car_
                                                                                  
    
                    
    parking_spaces))
                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 304
    Estimand name: backdoor304
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,lead_time,days_in_waiting_list,country,booking_chang
                                                                                  
    
                                    
    es,required_car_parking_spaces))
                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 305
    Estimand name: backdoor305
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,lead_time,country,booking_changes,total_stay,require
                                                                                  
    
                          
    d_car_parking_spaces))
                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 306
    Estimand name: backdoor306
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,re
                                                                                  
    
                               
    quired_car_parking_spaces))
                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 307
    Estimand name: backdoor307
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,meal,country,booking_changes,total_stay,required_car
                                                                                  
    
                     
    _parking_spaces))
                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 308
    Estimand name: backdoor308
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,days_in_waiting_list,country,booking_changes,total_s
                                                                                  
    
                                     
    tay,required_car_parking_spaces))
                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 309
    Estimand name: backdoor309
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes
                                                                                  
    
                                  
    ,required_car_parking_spaces))
                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 310
    Estimand name: backdoor310
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,lead_time,meal,country,booking_changes,total_stay,required_
                                                                                  
    
                        
    car_parking_spaces))
                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 311
    Estimand name: backdoor311
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,lead_time,days_in_waiting_list,country,booking_changes,tota
                                                                                  
    
                                        
    l_stay,required_car_parking_spaces))
                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 312
    Estimand name: backdoor312
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,meal,days_in_waiting_list,country,booking_changes,total_sta
                                                                                  
    
                                   
    y,required_car_parking_spaces))
                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 313
    Estimand name: backdoor313
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,guests,country,booking_changes,required_
                                                                                  
    
                        
    car_parking_spaces))
                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 314
    Estimand name: backdoor314
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,lead_time,country,booking_changes,requir
                                                                                  
    
                           
    ed_car_parking_spaces))
                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 315
    Estimand name: backdoor315
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,meal,country,booking_changes,required_ca
                                                                                  
    
                      
    r_parking_spaces))
                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 316
    Estimand name: backdoor316
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_cha
                                                                                  
    
                                      
    nges,required_car_parking_spaces))
                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 317
    Estimand name: backdoor317
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,country,booking_changes,total_stay,requi
                                                                                  
    
                            
    red_car_parking_spaces))
                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 318
    Estimand name: backdoor318
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,lead_time,country,booking_changes,required_car_park
                                                                                  
    
                
    ing_spaces))
                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 319
    Estimand name: backdoor319
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,meal,country,booking_changes,required_car_parking_s
                                                                                  
    
           
    paces))
           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 320
    Estimand name: backdoor320
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,days_in_waiting_list,country,booking_changes,requir
                                                                                  
    
                           
    ed_car_parking_spaces))
                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 321
    Estimand name: backdoor321
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,country,booking_changes,total_stay,required_car_par
                                                                                  
    
                 
    king_spaces))
                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 322
    Estimand name: backdoor322
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,lead_time,meal,country,booking_changes,required_car_parkin
                                                                                  
    
              
    g_spaces))
              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 323
    Estimand name: backdoor323
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,req
                                                                                  
    
                              
    uired_car_parking_spaces))
                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 324
    Estimand name: backdoor324
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,lead_time,country,booking_changes,total_stay,required_car_
                                                                                  
    
                    
    parking_spaces))
                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 325
    Estimand name: backdoor325
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,meal,days_in_waiting_list,country,booking_changes,required
                                                                                  
    
                         
    _car_parking_spaces))
                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 326
    Estimand name: backdoor326
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,meal,country,booking_changes,total_stay,required_car_parki
                                                                                  
    
               
    ng_spaces))
               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 327
    Estimand name: backdoor327
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,days_in_waiting_list,country,booking_changes,total_stay,re
                                                                                  
    
                               
    quired_car_parking_spaces))
                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 328
    Estimand name: backdoor328
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,lead_time,country,booking_changes,required_car_parking_spac
                                                                                  
    
        
    es))
        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 329
    Estimand name: backdoor329
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                 
    eated_guest,guests,meal,country,booking_changes,required_car_parking_spaces))
                                                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 330
    Estimand name: backdoor330
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_p
                                                                                  
    
                   
    arking_spaces))
                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 331
    Estimand name: backdoor331
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,guests,country,booking_changes,total_stay,required_car_parking_spa
                                                                                  
    
         
    ces))
         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 332
    Estimand name: backdoor332
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces
                                                                                  
    
      
    ))
      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 333
    Estimand name: backdoor333
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_ca
                                                                                  
    
                      
    r_parking_spaces))
                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 334
    Estimand name: backdoor334
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_
                                                                                  
    
            
    spaces))
            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 335
    Estimand name: backdoor335
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_par
                                                                                  
    
                 
    king_spaces))
                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 336
    Estimand name: backdoor336
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,meal,country,booking_changes,total_stay,required_car_parking_space
                                                                                  
    
       
    s))
       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 337
    Estimand name: backdoor337
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_c
                                                                                  
    
                       
    ar_parking_spaces))
                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 338
    Estimand name: backdoor338
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                         
    ,lead_time,meal,country,booking_changes,required_car_parking_spaces))
                                                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 339
    Estimand name: backdoor339
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                                  
    ,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_s
                                                                                  
    
           
    paces))
           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 340
    Estimand name: backdoor340
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                               
    ,lead_time,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 341
    Estimand name: backdoor341
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                                  
    ,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces
                                                                                  
    
      
    ))
      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 342
    Estimand name: backdoor342
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                          
    ,meal,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 343
    Estimand name: backdoor343
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                                  
    ,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_
                                                                                  
    
            
    spaces))
            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 344
    Estimand name: backdoor344
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,lead_t
    d[different_room_assigned]                                                    
    
                                                                                  
    ime,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spa
                                                                                  
    
         
    ces))
         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 345
    Estimand name: backdoor345
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,lead_t
    d[different_room_assigned]                                                    
    
                                                                             
    ime,meal,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 346
    Estimand name: backdoor346
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,lead_t
    d[different_room_assigned]                                                    
    
                                                                                  
    ime,days_in_waiting_list,country,booking_changes,total_stay,required_car_parki
                                                                                  
    
               
    ng_spaces))
               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 347
    Estimand name: backdoor347
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,meal,d
    d[different_room_assigned]                                                    
    
                                                                                  
    ays_in_waiting_list,country,booking_changes,total_stay,required_car_parking_sp
                                                                                  
    
          
    aces))
          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 348
    Estimand name: backdoor348
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,lead_time,country,booking_changes,requi
                                                                                  
    
                            
    red_car_parking_spaces))
                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 349
    Estimand name: backdoor349
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,meal,country,booking_changes,required_c
                                                                                  
    
                       
    ar_parking_spaces))
                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 350
    Estimand name: backdoor350
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_ch
                                                                                  
    
                                       
    anges,required_car_parking_spaces))
                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 351
    Estimand name: backdoor351
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,country,booking_changes,total_stay,requ
                                                                                  
    
                             
    ired_car_parking_spaces))
                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 352
    Estimand name: backdoor352
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,lead_time,meal,country,booking_changes,require
                                                                                  
    
                          
    d_car_parking_spaces))
                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 353
    Estimand name: backdoor353
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking
                                                                                  
    
                                          
    _changes,required_car_parking_spaces))
                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 354
    Estimand name: backdoor354
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,r
                                                                                  
    
                                
    equired_car_parking_spaces))
                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 355
    Estimand name: backdoor355
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_chan
                                                                                  
    
                                     
    ges,required_car_parking_spaces))
                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 356
    Estimand name: backdoor356
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,meal,country,booking_changes,total_stay,requir
                                                                                  
    
                           
    ed_car_parking_spaces))
                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 357
    Estimand name: backdoor357
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,t
                                                                                  
    
                                           
    otal_stay,required_car_parking_spaces))
                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 358
    Estimand name: backdoor358
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,lead_time,meal,country,booking_changes,required_car_parki
                                                                                  
    
               
    ng_spaces))
               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 359
    Estimand name: backdoor359
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,re
                                                                                  
    
                               
    quired_car_parking_spaces))
                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 360
    Estimand name: backdoor360
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,lead_time,country,booking_changes,total_stay,required_car
                                                                                  
    
                     
    _parking_spaces))
                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 361
    Estimand name: backdoor361
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,meal,days_in_waiting_list,country,booking_changes,require
                                                                                  
    
                          
    d_car_parking_spaces))
                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 362
    Estimand name: backdoor362
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,meal,country,booking_changes,total_stay,required_car_park
                                                                                  
    
                
    ing_spaces))
                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 363
    Estimand name: backdoor363
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,r
                                                                                  
    
                                
    equired_car_parking_spaces))
                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 364
    Estimand name: backdoor364
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,requ
                                                                                  
    
                             
    ired_car_parking_spaces))
                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 365
    Estimand name: backdoor365
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,lead_time,meal,country,booking_changes,total_stay,required_car_p
                                                                                  
    
                   
    arking_spaces))
                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 366
    Estimand name: backdoor366
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,lead_time,days_in_waiting_list,country,booking_changes,total_sta
                                                                                  
    
                                   
    y,required_car_parking_spaces))
                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 367
    Estimand name: backdoor367
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,req
                                                                                  
    
                              
    uired_car_parking_spaces))
                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 368
    Estimand name: backdoor368
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,guests,lead_time,meal,country,booking_changes,required_car_parking_space
                                                                                  
    
       
    s))
       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 369
    Estimand name: backdoor369
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_c
                                                                                  
    
                       
    ar_parking_spaces))
                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 370
    Estimand name: backdoor370
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking
                                                                                  
    
             
    _spaces))
             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 371
    Estimand name: backdoor371
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_pa
                                                                                  
    
                  
    rking_spaces))
                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 372
    Estimand name: backdoor372
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spac
                                                                                  
    
        
    es))
        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 373
    Estimand name: backdoor373
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_
                                                                                  
    
                        
    car_parking_spaces))
                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 374
    Estimand name: backdoor374
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car
                                                                                  
    
                     
    _parking_spaces))
                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 375
    Estimand name: backdoor375
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_s
                                                                                  
    
           
    paces))
           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 376
    Estimand name: backdoor376
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,requir
                                                                                  
    
                           
    ed_car_parking_spaces))
                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 377
    Estimand name: backdoor377
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_ca
                                                                                  
    
                      
    r_parking_spaces))
                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 378
    Estimand name: backdoor378
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,lead_
    d[different_room_assigned]                                                    
    
                                                                                  
    time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_sp
                                                                                  
    
          
    aces))
          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 379
    Estimand name: backdoor379
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,lead_
    d[different_room_assigned]                                                    
    
                                                                              
    time,meal,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 380
    Estimand name: backdoor380
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,lead_
    d[different_room_assigned]                                                    
    
                                                                                  
    time,days_in_waiting_list,country,booking_changes,total_stay,required_car_park
                                                                                  
    
                
    ing_spaces))
                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 381
    Estimand name: backdoor381
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,meal,
    d[different_room_assigned]                                                    
    
                                                                                  
    days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_s
                                                                                  
    
           
    paces))
           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 382
    Estimand name: backdoor382
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,lead_time,me
    d[different_room_assigned]                                                    
    
                                                                                  
    al,days_in_waiting_list,country,booking_changes,total_stay,required_car_parkin
                                                                                  
    
              
    g_spaces))
              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 383
    Estimand name: backdoor383
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,total_of_special_requests,market_segment,country,booking_changes,req
                                                                                  
    
                              
    uired_car_parking_spaces))
                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,total_of_special_requests,market_segment,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 384
    Estimand name: backdoor384
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,is_repeated_guest,country,booking_changes,required_ca
                                                                                  
    
                      
    r_parking_spaces))
                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 385
    Estimand name: backdoor385
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,guests,country,booking_changes,required_car_parking_s
                                                                                  
    
           
    paces))
           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 386
    Estimand name: backdoor386
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,lead_time,country,booking_changes,required_car_parkin
                                                                                  
    
              
    g_spaces))
              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 387
    Estimand name: backdoor387
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,meal,country,booking_changes,required_car_parking_spa
                                                                                  
    
         
    ces))
         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 388
    Estimand name: backdoor388
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,days_in_waiting_list,country,booking_changes,required
                                                                                  
    
                         
    _car_parking_spaces))
                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 389
    Estimand name: backdoor389
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,country,booking_changes,total_stay,required_car_parki
                                                                                  
    
               
    ng_spaces))
               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 390
    Estimand name: backdoor390
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,is_repeated_guest,country,booking
                                                                                  
    
                                          
    _changes,required_car_parking_spaces))
                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 391
    Estimand name: backdoor391
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,guests,country,booking_changes,re
                                                                                  
    
                               
    quired_car_parking_spaces))
                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 392
    Estimand name: backdoor392
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,lead_time,country,booking_changes
                                                                                  
    
                                  
    ,required_car_parking_spaces))
                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 393
    Estimand name: backdoor393
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,meal,country,booking_changes,requ
                                                                                  
    
                             
    ired_car_parking_spaces))
                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 394
    Estimand name: backdoor394
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,days_in_waiting_list,country,book
                                                                                  
    
                                             
    ing_changes,required_car_parking_spaces))
                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 395
    Estimand name: backdoor395
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,country,booking_changes,total_sta
                                                                                  
    
                                   
    y,required_car_parking_spaces))
                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 396
    Estimand name: backdoor396
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,guests,country,booking_changes,required_c
                                                                                  
    
                       
    ar_parking_spaces))
                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 397
    Estimand name: backdoor397
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,lead_time,country,booking_changes,require
                                                                                  
    
                          
    d_car_parking_spaces))
                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 398
    Estimand name: backdoor398
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,meal,country,booking_changes,required_car
                                                                                  
    
                     
    _parking_spaces))
                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 399
    Estimand name: backdoor399
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_chan
                                                                                  
    
                                     
    ges,required_car_parking_spaces))
                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 400
    Estimand name: backdoor400
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,country,booking_changes,total_stay,requir
                                                                                  
    
                           
    ed_car_parking_spaces))
                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 401
    Estimand name: backdoor401
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,lead_time,country,booking_changes,required_car_parki
                                                                                  
    
               
    ng_spaces))
               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 402
    Estimand name: backdoor402
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,meal,country,booking_changes,required_car_parking_sp
                                                                                  
    
          
    aces))
          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 403
    Estimand name: backdoor403
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,days_in_waiting_list,country,booking_changes,require
                                                                                  
    
                          
    d_car_parking_spaces))
                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 404
    Estimand name: backdoor404
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,country,booking_changes,total_stay,required_car_park
                                                                                  
    
                
    ing_spaces))
                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 405
    Estimand name: backdoor405
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,lead_time,meal,country,booking_changes,required_car_parking
                                                                                  
    
             
    _spaces))
             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 406
    Estimand name: backdoor406
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,lead_time,days_in_waiting_list,country,booking_changes,requ
                                                                                  
    
                             
    ired_car_parking_spaces))
                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 407
    Estimand name: backdoor407
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,lead_time,country,booking_changes,total_stay,required_car_p
                                                                                  
    
                   
    arking_spaces))
                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 408
    Estimand name: backdoor408
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,meal,days_in_waiting_list,country,booking_changes,required_
                                                                                  
    
                        
    car_parking_spaces))
                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 409
    Estimand name: backdoor409
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,meal,country,booking_changes,total_stay,required_car_parkin
                                                                                  
    
              
    g_spaces))
              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 410
    Estimand name: backdoor410
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,days_in_waiting_list,country,booking_changes,total_stay,req
                                                                                  
    
                              
    uired_car_parking_spaces))
                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 411
    Estimand name: backdoor411
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,is_repeated_guest,country,booking_changes,required_car_par
                                                                                  
    
                 
    king_spaces))
                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 412
    Estimand name: backdoor412
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,guests,country,booking_changes,required_car_parking_spaces
                                                                                  
    
      
    ))
      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 413
    Estimand name: backdoor413
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,lead_time,country,booking_changes,required_car_parking_spa
                                                                                  
    
         
    ces))
         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 414
    Estimand name: backdoor414
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,meal,country,booking_changes,required_car_parking_spaces))
                                                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 415
    Estimand name: backdoor415
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,days_in_waiting_list,country,booking_changes,required_car_
                                                                                  
    
                    
    parking_spaces))
                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 416
    Estimand name: backdoor416
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                                  
    ests,market_segment,country,booking_changes,total_stay,required_car_parking_sp
                                                                                  
    
          
    aces))
          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 417
    Estimand name: backdoor417
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                            
    eated_guest,guests,country,booking_changes,required_car_parking_spaces))
                                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 418
    Estimand name: backdoor418
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                               
    eated_guest,lead_time,country,booking_changes,required_car_parking_spaces))
                                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 419
    Estimand name: backdoor419
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                          
    eated_guest,meal,country,booking_changes,required_car_parking_spaces))
                                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 420
    Estimand name: backdoor420
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                  
    eated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_
                                                                                  
    
            
    spaces))
            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 421
    Estimand name: backdoor421
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                                
    eated_guest,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 422
    Estimand name: backdoor422
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                    
    ,lead_time,country,booking_changes,required_car_parking_spaces))
                                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 423
    Estimand name: backdoor423
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                               
    ,meal,country,booking_changes,required_car_parking_spaces))
                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 424
    Estimand name: backdoor424
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                               
    ,days_in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 425
    Estimand name: backdoor425
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                                     
    ,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 426
    Estimand name: backdoor426
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,lead_t
    d[different_room_assigned]                                                    
    
                                                                  
    ime,meal,country,booking_changes,required_car_parking_spaces))
                                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 427
    Estimand name: backdoor427
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,lead_t
    d[different_room_assigned]                                                    
    
                                                                                  
    ime,days_in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 428
    Estimand name: backdoor428
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,lead_t
    d[different_room_assigned]                                                    
    
                                                                        
    ime,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 429
    Estimand name: backdoor429
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,meal,d
    d[different_room_assigned]                                                    
    
                                                                             
    ays_in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 430
    Estimand name: backdoor430
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,meal,c
    d[different_room_assigned]                                                    
    
                                                                   
    ountry,booking_changes,total_stay,required_car_parking_spaces))
                                                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 431
    Estimand name: backdoor431
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,days_i
    d[different_room_assigned]                                                    
    
                                                                                  
    n_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
                                                                                  
    
     
    )
     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 432
    Estimand name: backdoor432
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,guests,country,booking_changes,required_car_pa
                                                                                  
    
                  
    rking_spaces))
                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 433
    Estimand name: backdoor433
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,lead_time,country,booking_changes,required_car
                                                                                  
    
                     
    _parking_spaces))
                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 434
    Estimand name: backdoor434
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,meal,country,booking_changes,required_car_park
                                                                                  
    
                
    ing_spaces))
                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 435
    Estimand name: backdoor435
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,r
                                                                                  
    
                                
    equired_car_parking_spaces))
                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 436
    Estimand name: backdoor436
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,country,booking_changes,total_stay,required_ca
                                                                                  
    
                      
    r_parking_spaces))
                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 437
    Estimand name: backdoor437
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,lead_time,country,booking_changes,required_car_parking_sp
                                                                                  
    
          
    aces))
          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 438
    Estimand name: backdoor438
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,meal,country,booking_changes,required_car_parking_spaces)
                                                                                  
    
     
    )
     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 439
    Estimand name: backdoor439
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,days_in_waiting_list,country,booking_changes,required_car
                                                                                  
    
                     
    _parking_spaces))
                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 440
    Estimand name: backdoor440
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,guests,country,booking_changes,total_stay,required_car_parking_s
                                                                                  
    
           
    paces))
           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 441
    Estimand name: backdoor441
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,lead_time,meal,country,booking_changes,required_car_parking_spac
                                                                                  
    
        
    es))
        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 442
    Estimand name: backdoor442
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,lead_time,days_in_waiting_list,country,booking_changes,required_
                                                                                  
    
                        
    car_parking_spaces))
                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 443
    Estimand name: backdoor443
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,lead_time,country,booking_changes,total_stay,required_car_parkin
                                                                                  
    
              
    g_spaces))
              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 444
    Estimand name: backdoor444
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,meal,days_in_waiting_list,country,booking_changes,required_car_p
                                                                                  
    
                   
    arking_spaces))
                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 445
    Estimand name: backdoor445
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,meal,country,booking_changes,total_stay,required_car_parking_spa
                                                                                  
    
         
    ces))
         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 446
    Estimand name: backdoor446
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,days_in_waiting_list,country,booking_changes,total_stay,required
                                                                                  
    
                         
    _car_parking_spaces))
                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 447
    Estimand name: backdoor447
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                
    guest,guests,lead_time,country,booking_changes,required_car_parking_spaces))
                                                                                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 448
    Estimand name: backdoor448
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                           
    guest,guests,meal,country,booking_changes,required_car_parking_spaces))
                                                                           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 449
    Estimand name: backdoor449
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking
                                                                                  
    
             
    _spaces))
             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 450
    Estimand name: backdoor450
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                 
    guest,guests,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 451
    Estimand name: backdoor451
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                              
    guest,lead_time,meal,country,booking_changes,required_car_parking_spaces))
                                                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 452
    Estimand name: backdoor452
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_park
                                                                                  
    
                
    ing_spaces))
                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 453
    Estimand name: backdoor453
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces
                                                                                  
    
      
    ))
      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 454
    Estimand name: backdoor454
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_s
                                                                                  
    
           
    paces))
           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 455
    Estimand name: backdoor455
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                               
    guest,meal,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 456
    Estimand name: backdoor456
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_par
                                                                                  
    
                 
    king_spaces))
                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 457
    Estimand name: backdoor457
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,lead_
    d[different_room_assigned]                                                    
    
                                                                   
    time,meal,country,booking_changes,required_car_parking_spaces))
                                                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 458
    Estimand name: backdoor458
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,lead_
    d[different_room_assigned]                                                    
    
                                                                                  
    time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
                                                                                  
    
     
    )
     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 459
    Estimand name: backdoor459
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,lead_
    d[different_room_assigned]                                                    
    
                                                                         
    time,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 460
    Estimand name: backdoor460
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,meal,
    d[different_room_assigned]                                                    
    
                                                                              
    days_in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 461
    Estimand name: backdoor461
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,meal,
    d[different_room_assigned]                                                    
    
                                                                    
    country,booking_changes,total_stay,required_car_parking_spaces))
                                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 462
    Estimand name: backdoor462
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,days_
    d[different_room_assigned]                                                    
    
                                                                                  
    in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces
                                                                                  
    
      
    ))
      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 463
    Estimand name: backdoor463
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,lead_time,me
    d[different_room_assigned]                                                    
    
                                                                                 
    al,days_in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,lead_time,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 464
    Estimand name: backdoor464
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,lead_time,me
    d[different_room_assigned]                                                    
    
                                                                       
    al,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,lead_time,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 465
    Estimand name: backdoor465
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,lead_time,da
    d[different_room_assigned]                                                    
    
                                                                                  
    ys_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spa
                                                                                  
    
         
    ces))
         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,lead_time,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 466
    Estimand name: backdoor466
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,meal,days_in
    d[different_room_assigned]                                                    
    
                                                                                  
    _waiting_list,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,meal,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 467
    Estimand name: backdoor467
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,hotel,market_segment,country,booking_changes,required_car_parking_spaces))
                                                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,hotel,market_segment,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 468
    Estimand name: backdoor468
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,total_of_special_requests,market_segment,country,booking_changes,required_
                                                                                  
    
                        
    car_parking_spaces))
                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,total_of_special_requests,market_segment,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 469
    Estimand name: backdoor469
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,is_repeated_guest,country,booking_changes,required_car_park
                                                                                  
    
                
    ing_spaces))
                
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 470
    Estimand name: backdoor470
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,guests,country,booking_changes,required_car_parking_spaces)
                                                                                  
    
     
    )
     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 471
    Estimand name: backdoor471
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,lead_time,country,booking_changes,required_car_parking_spac
                                                                                  
    
        
    es))
        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 472
    Estimand name: backdoor472
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                 
    led,market_segment,meal,country,booking_changes,required_car_parking_spaces))
                                                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 473
    Estimand name: backdoor473
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,days_in_waiting_list,country,booking_changes,required_car_p
                                                                                  
    
                   
    arking_spaces))
                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 474
    Estimand name: backdoor474
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                                  
    led,market_segment,country,booking_changes,total_stay,required_car_parking_spa
                                                                                  
    
         
    ces))
         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 475
    Estimand name: backdoor475
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,total_of_special_requ
    d[different_room_assigned]                                                    
    
                                                                             
    ests,market_segment,country,booking_changes,required_car_parking_spaces))
                                                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,total_of_special_requests,market_segment,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 476
    Estimand name: backdoor476
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,is_rep
    d[different_room_assigned]                                                    
    
                                                                     
    eated_guest,country,booking_changes,required_car_parking_spaces))
                                                                     
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 477
    Estimand name: backdoor477
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,guests
    d[different_room_assigned]                                                    
    
                                                          
    ,country,booking_changes,required_car_parking_spaces))
                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 478
    Estimand name: backdoor478
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,lead_t
    d[different_room_assigned]                                                    
    
                                                             
    ime,country,booking_changes,required_car_parking_spaces))
                                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 479
    Estimand name: backdoor479
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,meal,c
    d[different_room_assigned]                                                    
    
                                                        
    ountry,booking_changes,required_car_parking_spaces))
                                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 480
    Estimand name: backdoor480
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,days_i
    d[different_room_assigned]                                                    
    
                                                                        
    n_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 481
    Estimand name: backdoor481
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,countr
    d[different_room_assigned]                                                    
    
                                                              
    y,booking_changes,total_stay,required_car_parking_spaces))
                                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 482
    Estimand name: backdoor482
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,is_repeated_guest,country,booking_changes,required_car_parking_s
                                                                                  
    
           
    paces))
           
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 483
    Estimand name: backdoor483
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                              
    arket_segment,guests,country,booking_changes,required_car_parking_spaces))
                                                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 484
    Estimand name: backdoor484
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                 
    arket_segment,lead_time,country,booking_changes,required_car_parking_spaces))
                                                                                 
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 485
    Estimand name: backdoor485
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                            
    arket_segment,meal,country,booking_changes,required_car_parking_spaces))
                                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 486
    Estimand name: backdoor486
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,days_in_waiting_list,country,booking_changes,required_car_parkin
                                                                                  
    
              
    g_spaces))
              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 487
    Estimand name: backdoor487
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                                  
    arket_segment,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 488
    Estimand name: backdoor488
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                      
    guest,guests,country,booking_changes,required_car_parking_spaces))
                                                                      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 489
    Estimand name: backdoor489
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                         
    guest,lead_time,country,booking_changes,required_car_parking_spaces))
                                                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 490
    Estimand name: backdoor490
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                    
    guest,meal,country,booking_changes,required_car_parking_spaces))
                                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 491
    Estimand name: backdoor491
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                                  
    guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces
                                                                                  
    
      
    ))
      
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 492
    Estimand name: backdoor492
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                                          
    guest,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                          
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 493
    Estimand name: backdoor493
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,lead_
    d[different_room_assigned]                                                    
    
                                                              
    time,country,booking_changes,required_car_parking_spaces))
                                                              
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 494
    Estimand name: backdoor494
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,meal,
    d[different_room_assigned]                                                    
    
                                                         
    country,booking_changes,required_car_parking_spaces))
                                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 495
    Estimand name: backdoor495
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,days_
    d[different_room_assigned]                                                    
    
                                                                         
    in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                         
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 496
    Estimand name: backdoor496
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,count
    d[different_room_assigned]                                                    
    
                                                               
    ry,booking_changes,total_stay,required_car_parking_spaces))
                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 497
    Estimand name: backdoor497
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,lead_time,me
    d[different_room_assigned]                                                    
    
                                                            
    al,country,booking_changes,required_car_parking_spaces))
                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,lead_time,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 498
    Estimand name: backdoor498
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,lead_time,da
    d[different_room_assigned]                                                    
    
                                                                            
    ys_in_waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,lead_time,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 499
    Estimand name: backdoor499
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,lead_time,co
    d[different_room_assigned]                                                    
    
                                                                  
    untry,booking_changes,total_stay,required_car_parking_spaces))
                                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,lead_time,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 500
    Estimand name: backdoor500
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,meal,days_in
    d[different_room_assigned]                                                    
    
                                                                       
    _waiting_list,country,booking_changes,required_car_parking_spaces))
                                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,meal,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 501
    Estimand name: backdoor501
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,meal,country
    d[different_room_assigned]                                                    
    
                                                             
    ,booking_changes,total_stay,required_car_parking_spaces))
                                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,meal,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 502
    Estimand name: backdoor502
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,days_in_wait
    d[different_room_assigned]                                                    
    
                                                                             
    ing_list,country,booking_changes,total_stay,required_car_parking_spaces))
                                                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,days_in_waiting_list,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 503
    Estimand name: backdoor503
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|previous_bookings_not_cance
    d[different_room_assigned]                                                    
    
                                                                            
    led,market_segment,country,booking_changes,required_car_parking_spaces))
                                                                            
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,previous_bookings_not_canceled,market_segment,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 504
    Estimand name: backdoor504
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,market_segment,countr
    d[different_room_assigned]                                                    
    
                                                   
    y,booking_changes,required_car_parking_spaces))
                                                   
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,market_segment,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,hotel,market_segment,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 505
    Estimand name: backdoor505
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|total_of_special_requests,m
    d[different_room_assigned]                                                    
    
                                                                       
    arket_segment,country,booking_changes,required_car_parking_spaces))
                                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,total_of_special_requests,market_segment,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 506
    Estimand name: backdoor506
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,is_repeated_
    d[different_room_assigned]                                                    
    
                                                               
    guest,country,booking_changes,required_car_parking_spaces))
                                                               
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,is_repeated_guest,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 507
    Estimand name: backdoor507
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,guests,count
    d[different_room_assigned]                                                    
    
                                                    
    ry,booking_changes,required_car_parking_spaces))
                                                    
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,guests,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,guests,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 508
    Estimand name: backdoor508
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,lead_time,co
    d[different_room_assigned]                                                    
    
                                                       
    untry,booking_changes,required_car_parking_spaces))
                                                       
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,lead_time,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,lead_time,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 509
    Estimand name: backdoor509
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,meal,country
    d[different_room_assigned]                                                    
    
                                                  
    ,booking_changes,required_car_parking_spaces))
                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,meal,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,meal,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 510
    Estimand name: backdoor510
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,days_in_wait
    d[different_room_assigned]                                                    
    
                                                                  
    ing_list,country,booking_changes,required_car_parking_spaces))
                                                                  
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,days_in_waiting_list,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 511
    Estimand name: backdoor511
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,country,book
    d[different_room_assigned]                                                    
    
                                                        
    ing_changes,total_stay,required_car_parking_spaces))
                                                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,country,booking_changes,total_stay,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,country,booking_changes,total_stay,required_car_parking_spaces)
    
    ### Estimand : 512
    Estimand name: backdoor512
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|market_segment,country,book
    d[different_room_assigned]                                                    
    
                                             
    ing_changes,required_car_parking_spaces))
                                             
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,market_segment,country,booking_changes,required_car_parking_spaces,U) = P(is_canceled|different_room_assigned,market_segment,country,booking_changes,required_car_parking_spaces)
    
    ### Estimand : 513
    Estimand name: iv
    No such variable found!
    
    ### Estimand : 514
    Estimand name: frontdoor
    No such variable found!
    


#### TODO: Backdoor estimand 가 무수히 많이 생겨나는 그 로직에 대해서 추가적인 공부

## Step-3. Estimate Identified Estimand : 

dowhy.CausalModel() 클래스를 활용해서 모델을 입력하게 되면, 아래 함수들을 사용할 수 있습니다. [모듈 코드](https://microsoft.github.io/dowhy/_modules/dowhy/causal_model.html)

- identify_effect()
- estimate_effect()
    - method_name 파라미터 값으로 아래 종류들이 있습니다. 
        - Propensity Score Matching: “backdoor.propensity_score_matching”

        - Propensity Score Stratification: “backdoor.propensity_score_stratification”

        - Propensity Score-based Inverse Weighting: “backdoor.propensity_score_weighting”

        - Linear Regression: “backdoor.linear_regression”

        - Generalized Linear Models (e.g., logistic regression): “backdoor.generalized_linear_model”

        - Instrumental Variables: “iv.instrumental_variable”

        - Regression Discontinuity: “iv.regression_discontinuity”

- do()
- refute_estimate()
- view_model()
- interpret()
- summary()


```python
estimate = model.estimate_effect(identified_estimand, 
                                 method_name="backdoor.propensity_score_stratification",target_units="ate")

# target_units can be these three : 
# ATE = Average Treatment Effect
# ATT = Average Treatment Effect on Treated (i.e. those who were assigned a different room)
# ATC = Average Treatment Effect on Control (i.e. those who were not assigned a different room)

print(estimate)
```

    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+previous_bookings_not_canceled+hotel+total_of_special_requests+market_segment+is_repeated_guest+guests+lead_time+meal+days_in_waiting_list+country+booking_changes+total_stay+required_car_parking_spaces


    *** Causal Estimate ***
    
    ## Identified estimand
    Estimand type: nonparametric-ate
    
    ## Realized estimand
    b: is_canceled~different_room_assigned+previous_bookings_not_canceled+hotel+total_of_special_requests+market_segment+is_repeated_guest+guests+lead_time+meal+days_in_waiting_list+country+booking_changes+total_stay+required_car_parking_spaces
    Target units: ate
    
    ## Estimate
    Mean value: -0.2508732026233869
    


#### TODO: 참고로, method_name 에서 바로 EconML의 메소드를 call 할 수도 있습니다. 다른 메소드로 IV를 사용해보고 결과를 비교하려 했으나 IV가 식별되지 않아서 불가능했습니다. 또한 다른 backdoor 메소드 적용할 경우 모두 에러가 발생했습니다. 



```python
# 이 함수는 특별한 써머리가 없습니다.
model.interpret()
```

    INFO:dowhy.causal_model:Model to find the causal effect of treatment ['different_room_assigned'] on outcome ['is_canceled']


    Model to find the causal effect of treatment ['different_room_assigned'] on outcome ['is_canceled']


###### 결과는 꽤나 놀라웠습니다. 다른 방에 배정받는게 취소할 확률을 낮춰준다는 결과를 의미하기 떄문입니다. (-0.251만큼) 여기서, 이게 정말 맞는 인과 효과일까요? 

다른 메커니즘이 작용했을 수 있습니다. 다른 방에 배정받는 시간적인 상황이 체크인 때만 발생한다면? 이미 호텔에 와있기 때문에 취소할 수 있는 확률이 당연히 낮습니다. 이러한 케이스라면, 이 예약 취소가 언제 발생했는지에 대한 시각 정보가 주요한 변수가 됩니다. 예를 들어, `different_room_assigned` 변수가 예약을 한 당일에 주로 발생한다면 또 어떨까요? 이런 변수를 알게 됨으로써 그래프와 분석을 향상시킬 수 있습니다.

앞서 진행했던 연관성 분석이, `is_canceled` 와 `different_room_assigned` 사이의 양의 상관관계를 보였는데요. DoWhy를 통해서 인과 효과를 예측하는 것은 그 반대로, 아예 다른 그림을 보였습니다. 이는 곧, 다른 방에 배정하는 현상의 수를 줄이는 정책은 곧 호텔에게 비생산적인 방향의 정책일 수 있다는 점을 암시합니다.

## Step-4. Refute results


주의해야 할 점은 인과효과는 데이터셋으로부터 발견되는 것이 아니라, Identification 으로 이끄는 연구자의 가정(Assumptions)을 통해서 발견 됩니다. 데이터는 단순히 통계적인 추정(Estimation)에만 사용됩니다. 즉, 가정이 맞았는지 아닌지에 대해서 증명하는 것이 정말 중요합니다.

- 다른 Common cause 가 존재한다면?
- Treatment 자체가 Placebo 효과라면?

### Method-1
**Add Random Common Cause:**

- 랜덤한 독립 변수를 Common cause로 데이터셋에 추가했을 때, 현재의 추정 메소드가 다른 추정치를 돌려줄까요? 
- 데이터에서 랜덤하게 공변량 변수들을 Draw하고, 동일한 분석을 재 진행하여 인과 추정치가 변화하는지를 봅니다. (CNN 돌릴 때 랜덤 값 조정하여 강건성 확인하듯이)
- 정규 분포 내에 랜덤하게 빼고 더하는 형태로 변수 추가 
- 가정이 본래 옳았다면, 인과 추정치는 크게 변화해서는 안됩니다.



```python
refute1_results = model.refute_estimate(identified_estimand, estimate,
        method_name = "random_common_cause")

print(refute1_results)
```

    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest+w_random


    Refute: Add a Random Common Cause
    Estimated effect:-0.2509269875427757
    New effect:-0.2484279296616035
    


### Method-2
**Placebo Treatment Refuter:**

- 실제 Treatment 변수 `different_room_assigned` 를 랜덤한 독립 변수로 대체한다면, 추정된 인과 효과는 어떻게 변화할까요?
- 데이터에서 랜덤하게 공변량 변수를 하나 Draw하고 Treatment로 대체하여 동일한 분석을 재 진행하여 인과 추정치가 변화하는지를 봅니다.
- 가정이 본래 옳았다면, 새로운 추정치는 0에 가까워야 합니다.
- p-value 는 New effect 가 통계적으로 유의하게 0과 다른지를 검정합니다.
- p-value < 0.05 라면 New effect가 0과 다르기 때문에 인과 추정치가 문제가 있다는 것을 의미합니다. 


```python
refute2_results=model.refute_estimate(identified_estimand, estimate,
        method_name="placebo_treatment_refuter")

print(refute2_results)
```

    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Refutation over 100 simulated datasets of Random Data treatment
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Using a Binomial Distribution with 1 trials and 0.5 probability of success
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~placebo+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.placebo_treatment_refuter:Making use of Bootstrap as we have more than 100 examples.
                     Note: The greater the number of examples, the more accurate are the confidence estimates


    Refute: Use a Placebo Treatment
    Estimated effect:-0.2509269875427757
    New effect:7.121213729569751e-05
    p value:0.5
    


### Method-3
**Data Subset Refuter:**

- 주어진 전체 데이터 셋을, 랜덤하게 선택한 데이터셋 일부로 대체한다면 인과 추정치가 변화할까요?
- Cross-validation과 유사한 방식으로 데이터셋의 subset 을 생성합니다. 인과 추정치가 subset들에 걸쳐서 차이가 나는지를 확인합니다.
     - Making use of Bootstrap as we have more than 100 examples.The greater the number of examples, the more accurate are the confidence estimates
     - 정해진 subset을 추출하는 trial 수가 있고, 각 subset 마다의 결과는 보여지지 않고 결과량은 하나.이 값은 어떻게 추출되었는지?
- 가정이 본래 옳았다면, 분산이 크지 않습니다.
- p-value 는 New effect 가 통계적으로 유의하게 Estimated effect 와 다른지를 검정합니다.
- p-value < 0.05 라면 두 Effect가 다르기 때문에 인과 추정치가 문제가 있다는 것을 의미합니다. [출처](https://issueexplorer.com/issue/microsoft/dowhy/312)


```python
refute3_results=model.refute_estimate(identified_estimand, estimate,
        method_name="data_subset_refuter")
print(refute3_results)
```

    INFO:dowhy.causal_refuters.data_subset_refuter:Refutation over 0.8 simulated datasets of size 83709.6 each
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_estimator:INFO: Using Propensity Score Stratification Estimator
    INFO:dowhy.causal_estimator:b: is_canceled~different_room_assigned+total_of_special_requests+previous_bookings_not_canceled+meal+total_stay+guests+days_in_waiting_list+required_car_parking_spaces+booking_changes+country+hotel+lead_time+market_segment+is_repeated_guest
    INFO:dowhy.causal_refuters.data_subset_refuter:Making use of Bootstrap as we have more than 100 examples.
                     Note: The greater the number of examples, the more accurate are the confidence estimates


    Refute: Use a subset of data
    Estimated effect:-0.2509269875427757
    New effect:-0.2495950577205681
    p value:0.28
    


우리의 추정치가 세 가지의 Refutation test를 통과하는 것을 확인했습니다. 이는 추정치의 정확함을 증명하는 것은 아니지만, 추정치의 신뢰도를 높여줍니다.

## 소감

예로 다양한 refutation 을 거친 후에 인과 분석 결과에 대한 높은 확신을 얻었다고 합시다.

호텔의 관리자에게, 인과 분석 내용을 어떻게 쉽게 잘 설명해야 직관과 반대되는 정책을 실제로 집행하게 이끌 수 있을까? 하는 의문이 들었습니다. 작은 정책 집행부터 시작해서, 수차례 인과 분석을 통해 실제 비즈니스에서 지표 상승과 같은 긍정적인 경험을 거친 후 -> 큰 정책 집행에 있어서 분석 결과를 신뢰할 수 있겠다는 생각이 들었습니다. 

인과 분석을 실제 비즈니스에 적용시키기 위해서, 이해관계자 및 의사 결정권자들이 절차에 있고 또 그 분들의 인과 분석 이해도가 낮다면 분석 그 자체보다도 더 큰 챌린지가 되겠다는 생각이 들었습니다.

그와 별개로, DoWhy 에서 [Open Issues](https://github.com/microsoft/dowhy/issues) 는 물론 다른 웹사이트들에서도 다양한 질의답변이 있어서 혼자 참고하면서 공부하기 좋은 것 같습니다.
