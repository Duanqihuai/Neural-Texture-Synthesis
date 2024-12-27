import torch
#calculate MSE Loss,input: A(matrix),B(matrix)  return: loss(float)
#the shape of A/B: [H,W] (as for gram_matrix,H=W)  
def cal_MSE(A,B):
    assert A.shape == B.shape

    return torch.mean((A - B) ** 2)