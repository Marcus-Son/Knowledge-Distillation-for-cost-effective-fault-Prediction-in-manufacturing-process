#### Advance inspection ratio에 따른 Test data의 AUC 구하는 함수 만들기
import pandas as pd 
from sklearn.metrics import roc_auc_score


#### Advance inspection ratio에 따른 Test data의 AUC 구하는 함수 만들기

def split_and_predict_uncertainty(test_data, random_forest_ad, test_y, split_ratio):
    
# split_ratio가 0일 경우: basic_model 데이터 전체에 대해 예측을 바로 사용하여 AUC 계산
    
    if split_ratio == 0:
        test_X_last = test_data.drop(['uncertainty_scores', '0.1'], axis=1)
        predictions = test_X_last['predictions']
        auc_score = roc_auc_score(test_y, predictions)
        return auc_score

    if split_ratio == 1:
        uncertainty_100 = test_data.drop(['predictions', 'uncertainty_scores', '0.1'], axis=1)
        predictions = random_forest_ad.predict_proba(uncertainty_100)[:, 1]
        uncertainty_100['predictions'] = predictions
        predictions = uncertainty_100['predictions']
        auc_score = roc_auc_score(test_y, predictions)
        return auc_score

    split_index = int(len(test_data) * split_ratio)
    uncertainty_part = test_data.iloc[:split_index].drop(['predictions', 'uncertainty_scores', '0.1'], axis=1)
    remaining_part = test_data.iloc[split_index:].drop(['uncertainty_scores'], axis=1)
    predictions = random_forest_ad.predict_proba(uncertainty_part)[:, 1]
    uncertainty_part['predictions'] = predictions
    result = pd.concat([uncertainty_part, remaining_part], axis=0)
    predictions = result['predictions']
    auc_score = roc_auc_score(test_y, predictions)
    return auc_score