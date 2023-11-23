import pandas as pd


    # Test data에 있는 피처 값들을 생성하기 위해 회귀 모델 만들기

def virtual_features(train_X,test_X,num_features):    
    predicted_features=pd.DataFrame()
    test_stage_features=[0,1,2,3,4,5,6,7]
    train_X_new=train_X[test_stage_features]
    ET_regressor_dict={}
    for new_feature in range(8,num_features):
        from sklearn.ensemble import ExtraTreesRegressor
        # ExtraTreeRegressor 모델 객체를 생성합니다.
        ET_regressor=ExtraTreesRegressor(n_estimators=50,min_samples_split=10,random_state=42)
        train_vm=train_X[(new_feature)]
        # 모델을 훈련 데이터에 적합시키기
        ET_regressor.fit(train_X_new,train_vm)
        
        # 전체 데이터셋에 대해 새로운 feature를 예측합니다.
        predictions=ET_regressor.predict(train_X_new)
        
        # 예측된 새로운 feature를 사용하여 데이터를 확장합니다.
        train_X_new[(new_feature)]=predictions
        ET_regressor_dict[(new_feature)]=ET_regressor
        
    # test data virtual metrology
        
    test_X_new=test_X[test_stage_features]
    for new_feature in range(8,num_features):
        # Select the current feature from the test data 
        ET_regressor=ET_regressor_dict[new_feature]
        test_predictions=ET_regressor.predict(test_X_new)
        test_X_new[new_feature]=test_predictions
    
    return test_X_new