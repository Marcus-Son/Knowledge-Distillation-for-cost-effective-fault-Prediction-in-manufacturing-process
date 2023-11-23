#### Advance inspection ratio에 따른 Test data의 AUC 구하는 함수 만들기

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import warnings 
import numpy as np 
import pandas as pd 
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset,DataLoader

#### Advance inspection ratio에 따른 Test data의 AUC 구하는 함수 만들기

def split_and_predict_uncertainty(test_data, best_teacher_model, test_y, split_ratio,scaler,batch_size):
    
# split_ratio가 0일 경우: basic_model 데이터 전체에 대해 예측을 바로 사용하여 AUC 계산
    
    if split_ratio == 0:
        test_X_last = test_data.drop(['uncertainty_scores', '0.1'], axis=1) # 불필요한 열 제거
        predictions = test_X_last['predictions'] # 예측값 추출
        auc_score = roc_auc_score(test_y, predictions) # AUC 계산
        return auc_score

# split_ratio가 1일 경우: Advanced model 사용해 테스트 데이터 전체에 대해 예측 후 AUC 계산
    if split_ratio == 1:
        uncertainty_100 = test_data.drop(['predictions', 'uncertainty_scores', '0.1'], axis=1) # 불필요한 열 제거
        uncertainty_100_scaler=scaler.transform(uncertainty_100)
        uncertainty_100_tensor=torch.tensor(uncertainty_100_scaler,dtype=torch.float32)
        test_y_tensor=torch.tensor(np.array(test_y),dtype=torch.long)
        test_tensorboard=TensorDataset(uncertainty_100_tensor,test_y_tensor)
        test_loader=DataLoader(test_tensorboard,batch_size=batch_size,shuffle=False)
        best_teacher_model.eval()
        
        test_preds=[]
        test_labels=[]
        
        with torch.no_grad():
            for xx, yy in test_loader:
                outputs_test=best_teacher_model(xx)
                outputs_test_softmax=F.softmax(outputs_test,dim=1)
                test_preds.extend(outputs_test_softmax[:,1].numpy())
                test_labels.extend(yy.numpy())
        auc_score = roc_auc_score(test_labels, test_preds,multi_class='ovr')
        return auc_score
    
#split_ratio가 0 또는 1이 아닌 다른 값일 경우
    
    split_index = int(len(test_data) * split_ratio) # 분할 인덱스 계산
    uncertainty_part = test_data.iloc[:split_index].drop(['predictions', 'uncertainty_scores', '0.1'], axis=1) # uncertainty score를 기준으로 advanced model로 예측할 부분
    uncertainty_part_label=test_data.iloc[:split_index]['0.1']
    remaining_part = test_data.iloc[split_index:].drop(['uncertainty_scores'], axis=1) # uncertainty score를 기준으로 advanced model로 예측할 부분
    uncertainty_part_scaler=scaler.transform(uncertainty_part)
    uncertainty_part_tensor=torch.tensor(uncertainty_part_scaler,dtype=torch.float32)
    uncertainty_part_label_tensor=torch.tensor(np.array(uncertainty_part_label),dtype=torch.long)    
    test_y_tensor=torch.tensor(np.array(test_y),dtype=torch.long)
    test_tensorboard=TensorDataset(uncertainty_part_tensor,uncertainty_part_label_tensor)
    test_loader=DataLoader(test_tensorboard,batch_size=batch_size,shuffle=False)
    
    test_preds=[] # Test data 예측값들의 리스트
    test_labels=[] # Test data의 실제값들의 리스트
    best_teacher_model.eval() # 모델 예측
    with torch.no_grad():
        for xx, yy in test_loader:
            outputs_test=best_teacher_model(xx)
            outputs_test_softmax=F.softmax(outputs_test,dim=1)
            test_preds.extend(outputs_test_softmax[:,1].numpy())
            test_labels.extend(yy.numpy())
            test_y_list=test_y_tensor.tolist()  
    uncertainty_part['predictions'] = np.array(test_preds)
    result = pd.concat([uncertainty_part, remaining_part], axis=0)
    predictions = result['predictions']
    auc_score = roc_auc_score(np.array(test_y_list), predictions)
    return auc_score