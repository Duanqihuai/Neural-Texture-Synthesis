import torch
import cv2
import numpy as np
import argparse
import torchvision.transforms.functional as TF
from torchvision import transforms
from tqdm import tqdm
from VGG19 import get_vgg19, rescale_weights
from Optimizier import loss_fn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean=torch.tensor([0.485,0.456,0.406])
std=torch.tensor([0.229,0.224,0.225])
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    # transforms.Normalize(mean,std)
])

def load_image(image_path):
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=transform(image)
    image=image.unsqueeze(0)
    # print(image.shape)
    return image

def save_image(image,path):
    path='output/'+path
    image=image.squeeze(0)
    image=image.permute(1,2,0)
    # m=mean.to(device)
    # s=std.to(device)
    # image=image*s+m
    # image=image.clamp(0,1)
    image=image.detach().cpu()
    image=image.numpy()
    image=(image*255).astype(np.uint8)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite(path,image)

def texture_synthesis(model,target_img,args):
    syn_img=torch.rand(1,3,224,224)
    syn_img=syn_img.to(device).requires_grad_()
    model=model.to(device)
    target_img=target_img.to(device)
    

    model(target_img)
    target_features=model.features_maps

    def closure():
        optimizer.zero_grad()
        model(syn_img)
        syn_features=model.features_maps
        loss=loss_fn(syn_features,target_features,args.layer_list)
        # print(f'Loss={loss.item()}',end=',')
        loss.backward(retain_graph=True)
        return loss

    optimizer=torch.optim.LBFGS([syn_img],lr=args.lr)
    for i in tqdm(range(args.epochs)):
        optimizer.step(closure)
        syn_img.data.clamp_(0,1)
        # print(f'Epoch={i+1}')
        if i%100==0:
            save_image(syn_img,f'{i+1}_{args.output_path}')

    save_image(syn_img,args.output_path)
    return syn_img




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural texture synthesis')
    parser.add_argument('--model', type=str, default='vgg19',help='Model to use')
    parser.add_argument('--pooling', type=str, default='avg',help='Pooling method')
    parser.add_argument('--rescale', type=bool, default=True,help='Rescale the weights or not')
    parser.add_argument('--lr', type=float, default=0.1,help='Learning rate')
    parser.add_argument('--image_path', type=str, default='images/pebbles.jpg',help='Path to the source image')
    parser.add_argument('--output_path', type=str, default='output.jpg',help='Path to save the synthesized image')
    parser.add_argument('--epochs', type=int, default=100,help='Number of epochs')
    parser.add_argument('--layer_list',nargs='+', default=['conv1_1','conv2_1','conv3_1','conv4_1'],help='List of layers to use')
    args = parser.parse_args()
    print(args.layer_list)

    model=get_vgg19(args.pooling)
    target_img=load_image(args.image_path)
    if args.rescale:
        model=rescale_weights(model,[target_img])

    texture_synthesis(model,target_img,args)
