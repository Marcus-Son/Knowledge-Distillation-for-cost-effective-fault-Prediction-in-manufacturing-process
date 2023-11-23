import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def split_and_predict(test_data, random_forest_ad, test_y, split_ratio, random_state):
    # advance inspection ratio의 비율이 0인 basic model(student model)의 성능 
    if split_ratio == 0:
        test_X_last = test_data.drop(['0.1'], axis=1)
        predictions = test_X_last['predictions']
        auc_score = roc_auc_score(test_y, predictions)
        return auc_score
    
    # advance inspection ratio의 비율이 1인 advance model(Teacher model)의 성능
    if split_ratio == 1:
        randomsampling_100 = test_data.drop(['predictions', '0.1'], axis=1)
        predictions = random_forest_ad.predict_proba(randomsampling_100)[:, 1]
        randomsampling_100['predictions'] = predictions
        predictions = randomsampling_100['predictions']
        auc_score = roc_auc_score(test_y, predictions)
        return auc_score
    
    random_selection_data =test_data.sample(n=int(len(test_data)*split_ratio),replace=False,random_state=random_state) # randomsampling 진행
    random_selection_data_y=random_selection_data['0.1'] 
    random_selection_part=random_selection_data.drop(['predictions','0.1'],axis=1) # radom sampling 후 advanced에서 이용할 부분

    remaining_data = test_data.drop(random_selection_data.index)
    remaining_data_y=remaining_data['0.1']
    remaining_part = remaining_data.drop(['0.1'],axis=1)
    
    predictions = random_forest_ad.predict_proba(random_selection_part)[:, 1]
    random_selection_part['predictions'] = predictions
    result = pd.concat([random_selection_part, remaining_part], axis=0)
    predictions = result['predictions']
    test_y=pd.concat([random_selection_data_y,remaining_data_y],axis=0)
    auc_score = roc_auc_score(test_y, predictions)
    return auc_score
