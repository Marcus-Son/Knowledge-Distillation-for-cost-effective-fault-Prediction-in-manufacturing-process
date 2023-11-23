import torch 
import numpy as np
import torch.nn.functional as F 


def test_predict(model,model_name,test_loader):
    
    if model_name=="best_teacher_model":
        test_preds=[]
        test_labels=[]
        model.eval()
        with torch.no_grad():
            for xx, yy in test_loader:
                test_outputs=model(xx)
                test_outputs_softmax=F.softmax(test_outputs,dim=1)
                test_preds.extend(test_outputs_softmax[:,1].numpy())
                test_labels.extend(yy.numpy())       
        y_predictions=np.array(test_preds)
        return y_predictions
    
    
    if model_name=="best_student_model":
        # 가장 좋은 basic model로 Test data AUC 계산
        test_preds=[] # 테스트 데이터에 대한 예측값을 저장하기 위한 리스트
        test_labels=[] # 테스트 데이터의 실제 라벨을 저장하기 위한 리스트
        
        model.eval()
        # 그래디언트 계산이 필요하지 않기 때문에 torch.no_grad()내에서 실행
        with torch.no_grad():
            # 테스트 데이터으 각 배치에 대해 반복
            for xx, yy in test_loader:
                # xx에서 첫 8개의 컬럼만 사용
                xx_s=xx[:,:8]
                # 학생 모델을 통해 예측값을 생성
                outputs_test=model(xx_s)
                # 소프트 맥스를 사용하여 확률값을 계산
                outputs_test_softmax=F.softmax(outputs_test,dim=1)
                # 확률값과 라벨을 저장
                test_preds.extend(outputs_test_softmax[:,1].numpy())
                test_labels.extend(yy.numpy())
        # 예측값을 numpy 배열로 변환    
        y_predictions=np.array(test_preds)
        return y_predictions
    
    