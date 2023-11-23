import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F 
import torch.nn as nn

def train_student(student_model,optimizer_basic,criterion,train_loader,val_loader,num_epochs):
    
    # 이제 Student model(basic model) 구하기 - advance model과 같은 방식으로 학습 및 검증
  
    train_losses=[]
    val_losses=[]
    train_auces=[]
    val_auces=[]
    train_preds=[]
    train_labels=[]
    val_preds=[]
    val_labels=[]
    best_val_loss=float('inf')
    best_val_auc=0
    early_stopping_patience=10
    early_stopping_counter=0
    best_student_score=0
    best_student_model=None 
    num_epochs=200
    for epoch in range(num_epochs):
        student_model.train()
        batch_loss=0
        for xx, yy in train_loader:
            optimizer_basic.zero_grad()
            xx_s=xx[:,:8]
            outputs_student=student_model(xx_s)
            outputs_student_softmax=F.softmax(outputs_student,dim=1)
            yy=yy.squeeze(dim=-1)
            student_losses =  criterion(outputs_student, yy)
            train_preds.extend(outputs_student_softmax[:,1].detach().numpy())
            train_labels.extend(yy.numpy())
            student_losses.backward()
            optimizer_basic.step()
            batch_loss+=student_losses.item() 
        
        train_losses.append(batch_loss/len(train_loader))
        train_auc=roc_auc_score(train_labels,train_preds,multi_class='ovr')
        train_auces.append(train_auc)
        
        student_model.eval()
        val_batch_loss=0
        val_preds=[]
        val_labels=[]
        
        # Validation data 평가하기
        
        with torch.no_grad():
            for xx, yy in val_loader:
                xx_s=xx[:,:8]
                outputs_val=student_model(xx_s)
                outputs_val_softmax=F.softmax(outputs_val,dim=1)
                yy=yy.squeeze(dim=-1)
                val_batch_loss+=criterion(outputs_val,yy).item()
                val_preds.extend(outputs_val_softmax[:,1].detach().numpy())
                val_labels.extend(yy.numpy())
        
        val_losses.append(val_batch_loss/len(val_loader))
        val_loss=val_batch_loss/len(val_loader)
        val_auc=roc_auc_score(val_labels,val_preds,multi_class='ovr')
        val_auces.append(val_auc)
            
        # Validation loss가 10회 이상 나아지지 않으면 학습 중단하기 그리고 가장 좋은 basice model 저장하기
        

        if val_auc > best_val_auc:
            best_val_auc=val_auc 
            best_student_model=student_model
            early_stopping_counter=0
        else:
            early_stopping_counter+=1
            
        if early_stopping_counter >=early_stopping_patience:
            print('Basic model Early stopping')
            break
    return best_student_model

