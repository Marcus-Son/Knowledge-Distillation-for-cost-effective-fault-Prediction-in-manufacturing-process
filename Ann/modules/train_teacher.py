import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F 
import torch.nn as nn

def train_teacher(teacher_model,optimizer_advance,criterion,train_loader,val_loader,num_epochs):
    
    # Teacher model (advanced model 구하기)    
    # loss function - crossentropy()함수

    train_losses = []
    train_auces = []
    val_losses = []
    val_auces = []
    train_preds=[]
    train_labels=[]
    best_val_auc=0
    best_val_loss = float('inf')
    early_stopping_patience=10
    early_stopping_counter=0
    best_model = None
    
    # train data로 200회 에포크 반복 실험
     
    for epoch in range(num_epochs):
        #Forward pass
        batch_loss=0
        # 학습 모드 설정
        teacher_model.train()
        # 학습 데이터의 각 배치에 대해 반복
        for xx, yy in train_loader:
            # 그래디언트를 0으로 초기화.
            optimizer_advance.zero_grad()
            # 모델의 출력값을 계산
            outputs_teacher=teacher_model(xx)
            # 소프트맥스를 사용하여 확률값을 계산
            outputs_teacher_softmax=F.softmax(outputs_teacher,dim=1)
            # 라벨의 차원을 조절
            yy=yy.squeeze(dim=-1)
            # 손실함수를 계산
            train_loss=criterion(outputs_teacher,yy)
            # 확률값과 라벨을 저장
            train_preds.extend(outputs_teacher_softmax[:,1].detach().numpy())
            train_labels.extend(yy.numpy())
            # 역전파를 통해 그래디언트를 계산
            train_loss.backward()
            optimizer_advance.step()
            batch_loss+=train_loss.item()
        
        train_losses.append(batch_loss/len(train_loader))
        train_auc=roc_auc_score(train_labels,train_preds,multi_class='ovr')
        train_auces.append(train_auc)
        
        # Validation data 평가하기 
        
        teacher_model.eval()
        val_batch_loss=0
        val_preds=[]
        val_labels=[]
        val_bestloss=None 
        with torch.no_grad():
            for xx, yy in val_loader:
                # 모델의 출력값을 계산 
                outputs_val=teacher_model(xx)
                # 소프트맥스를 사용하여 확률값을 계산
                outputs_val_softmax=F.softmax(outputs_val,dim=1)
                # 라벨의 차원을 조절 
                yy=yy.squeeze(dim=-1)
                # 손실을 계산하고 누적
                val_batch_loss+=criterion(outputs_val,yy).item()
                # 확률값과 라벨을 저장
                val_preds.extend(outputs_val_softmax[:,1].detach().numpy())
                val_labels.extend(yy.numpy())
        
        # 평균 검증 손실을 저장        
        val_losses.append(val_batch_loss/len(val_loader))
        val_loss=val_batch_loss/len(val_loader)
        # 검증 데이터의 AUC를 계산                        
        val_auc=roc_auc_score(val_labels,val_preds,multi_class='ovr')
        val_auces.append(val_auc)

        # validation loss가 10회 이상 나아지지 않으면 학습 중단하기
        
        if val_auc > best_val_auc:
            best_val_auc=val_auc 
            best_teacher_model=teacher_model
            early_stopping_counter=0
        else:
            early_stopping_counter+=1
        
        # 조기종료 판단
        if early_stopping_counter >=early_stopping_patience:
            print('Advance model Early stopping')
            break
    return best_teacher_model
