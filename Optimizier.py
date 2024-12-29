import torch
from Calculate_Gram import cal_gram
#calculate MSE Loss,input: A(matrix),B(matrix)  return: loss(float)
#the shape of A/B: [H,W] (as for gram_matrix,H=W)  
def cal_MSE(A,B):
    assert A.shape == B.shape

    return torch.mean((A - B) ** 2)

def loss_fn(syn_features,target_features,selected_layers=["conv1_1","conv1_2","pool1","conv2_1","conv2_2","pool2",
                                                          "conv3_1","conv3_2","conv3_3","conv3_4","pool3","conv4_1",
                                                          "conv4_2","conv4_3","conv4_4","pool4","conv5_1","conv5_2",
                                                          "conv5_3","conv5_4","pool5"]):
    loss=0.0

    for layer in selected_layers:
        syn_feature=syn_features[layer]
        target_feature=target_features[layer]
        syn_gram=cal_gram(syn_feature)
        target_gram=cal_gram(target_feature)
        loss+=cal_MSE(syn_gram,target_gram)

    return loss
    