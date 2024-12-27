from VGG19 import get_vgg19
import torch
from Calculate_Gram import cal_gram
from Optimizier import cal_MSE

#example code about caculate gram and mse
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=get_vgg19().to(device)
    img1 = torch.rand(1,3,224,224).to(device)
    img2 = torch.rand(1,3,224,224).to(device)
    model(img1)
    feature_maps1 = model.features_maps
    model(img2)
    feature_maps2 = model.features_maps
    Gram_matrix1=[]
    Gram_matrix2=[]
    for key,value in feature_maps1.items():   
        Gram_matrix1.append(cal_gram(value))  #G=[G1,G2,G3,G4......Gn]
    '''  a approch to choose a few layer to calculate gram and mse
    Chosen_Layers=["conv1_1","conv1_2"]
    for key,value in feature_maps1.items():   
        if key in Chosen_Layers:
            Gram_matrix1.append(cal_gram(value)) 
    '''
    for key,value in feature_maps2.items():
        Gram_matrix2.append(cal_gram(value))  #G=[G1,G2,G3,G4......Gn]
    Gram_MSE=[] #MSE for every layer...if needed can be summed
    for i in range(len(Gram_matrix1)):
        Gram_MSE.append(cal_MSE(Gram_matrix1[i],Gram_matrix2[i]))

if __name__ == '__main__':
    main()