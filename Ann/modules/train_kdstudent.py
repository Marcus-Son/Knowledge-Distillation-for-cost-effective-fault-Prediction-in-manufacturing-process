import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F 
import torch.nn as nn


def train_kdstudent(best_teacher_model,student_model,optimizer_KD_t1,criterion,train_loader,val_loader,num_epochs,temperature,alpha):
    
    # 이제 T=1일 때 KD Basic model 구하기   
    T=temperature
    train_losses=[]
    val_losses=[]
    train_auces=[]
    val_auces=[]
    train_preds=[]
    train_labels=[]
    val_preds=[]
    val_labels=[]
    best_val_auc=0
    best_val_loss=float('inf')
    early_stopping_patience=10
    early_stopping_counter=0
    best_student_score=0
    best_student_model=None 
    num_epochs=200
    
    ### 클래스 weight 설정하기

    # KD - loss를 distillation loss 사용.

    for epoch in range(num_epochs):
        student_model.train()
        batch_loss=0
        for xx, yy in train_loader:
            optimizer_KD_t1.zero_grad()
            xx_s=xx[:,:8]

            # distillation loss 구할 때 가장 좋은 advance model 사용하기
            # 가장 좋은 teacher model 모델을 사용하여 출력 생성
            outputs_teacher=best_teacher_model(xx)

            # student model 모델을 사용하여 출력 생성
            outputs_kdstudent=student_model(xx_s)
            outputs_kdstudent_softmax=F.softmax(outputs_kdstudent,dim=1)
            yy=yy.squeeze(dim=-1)
            
            #distillation loss 함수 짜기 --  alpha*T*T*distillation_loss+(1-alpha)*cross_entropy
            #지식 전이 loss 계산
            distillation_loss=nn.KLDivLoss()(F.log_softmax(outputs_kdstudent/T), F.softmax(outputs_teacher/T)) * (T*T*alpha)+criterion(outputs_kdstudent,yy)*(1-alpha)
            
            # 예측값과 라벨을 리스트에 추가
            train_preds.extend(outputs_kdstudent_softmax[:,1].detach().numpy())
            train_labels.extend(yy.numpy())
           
            ### 그래디언트 계산
            distillation_loss.backward()
            
            # 최적화
            optimizer_KD_t1.step()
            batch_loss+=distillation_loss.item() 
                
        # student model 검증
        student_model.eval()
        val_batch_loss=0
        
        with torch.no_grad():
            for xx, yy in val_loader:
                xx_s=xx[:,:8]
                outputs_val_teacher=best_teacher_model(xx)
                outputs_val_student=student_model(xx_s)
                outputs_val_softmax=F.softmax(outputs_val_student,dim=1)
                yy=yy.squeeze(dim=-1)
                val_distillation_loss=nn.KLDivLoss()(F.log_softmax(outputs_val_student/T),F.softmax(outputs_val_teacher/T))*(T*T*alpha)+criterion(outputs_val_student,yy)*(1-alpha)
                val_batch_loss+=val_distillation_loss
                val_preds.extend(outputs_val_softmax[:,1].detach().numpy())
                val_labels.extend(yy.numpy())
        
        val_losses.append(val_batch_loss/len(val_loader))
        val_loss=val_batch_loss/len(val_loader)
        val_auc=roc_auc_score(val_labels,val_preds,multi_class='ovr')
        val_auces.append(val_auc)
            
        # Check for early stopping 

        if val_auc > best_val_auc:
            best_val_auc=val_auc 
            best_student_model=student_model
            early_stopping_counter=0
        else:
            early_stopping_counter+=1
            
        if early_stopping_counter >=early_stopping_patience:
            print('KD Basic model Early stopping')
            break
    return best_student_model