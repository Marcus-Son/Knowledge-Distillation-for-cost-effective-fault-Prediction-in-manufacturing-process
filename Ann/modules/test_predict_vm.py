import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as F 

def test_predict_vm(best_teacher_model,test_X_new_scaled,test_y_tensor,batch_size):
    test_preds=[]
    test_labels=[]
    best_teacher_model.eval()
    with torch.no_grad():
        test_X_new_tensor=torch.tensor(test_X_new_scaled,dtype=torch.float32)
        test_new_tensorboard=TensorDataset(test_X_new_tensor,test_y_tensor)
        test_new_loader=DataLoader(test_new_tensorboard,batch_size=batch_size,shuffle=False)
        for xx, yy in test_new_loader:
            test_outputs=best_teacher_model(xx)
            test_outputs_softmax=F.softmax(test_outputs,dim=1)
            test_preds.extend(test_outputs_softmax[:,1].numpy())
            test_labels.extend(yy.numpy())       
    y_predictions=np.array(test_preds)
    return y_predictions

