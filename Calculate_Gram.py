import torch
#input:feature from VGG with size of [batch_size,channels,H,W]
def cal_gram(input):
    assert len(input.shape)==4

    batch_size,channels,H,W=input.shape
    feature=input.view(batch_size*channels,H*W)
    G = torch.matmul(feature, feature.t())
    gram_matrix=G.div(H*W)
    return gram_matrix

