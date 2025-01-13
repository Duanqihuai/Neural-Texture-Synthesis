import torch
import cv2
import numpy as np
import argparse
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from VGG19 import get_vgg19, rescale_weights
from Optimizier import loss_fn
from CNNMRF import Loss_forward
from orientation import DirectionFeatureExtractor

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
    path='output_mrf/fruit/'+path
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

def mrf_synthesis(model, target_img, args, orientations):
    syn_img=torch.rand(1,3, 224, 224)
    syn_img=syn_img.to(device).requires_grad_()
    model=model.to(device)
    target_img=target_img.to(device)

    model(target_img)
    target_features=model.features_maps

    loss_forward = Loss_forward(
        h=args.h, 
        patch_size=args.patch_size, 
        # progression_weight=args.lambda_progression, 
        orientation_weight=args.lambda_orientation,
        occurrence_weight=args.lambda_occurrence,
        ).to(device)
    def get_loss(target_features, refer_features, progressions, orientations, layers=[]):
        loss = 0
        for layer in layers:
            loss += loss_forward(
                target_features[layer], refer_features[layer],
                progressions, orientations,
            )
        return loss
    optimizer = torch.optim.Adam([syn_img],lr=0.01)
    for i in tqdm(range(args.epochs)):
        optimizer.zero_grad()
        syn_img.data.clamp_(0,1)
        model(syn_img)
        syn_features=model.features_maps
        loss = get_loss(
            syn_features, target_features,
            None,
            orientations,
            layers=args.layer_list
        )

        loss.backward(retain_graph=True)
        optimizer.step()

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
    parser.add_argument('--epochs', type=int, default=1000,help='Number of epochs')
    parser.add_argument('--layer_list', type=list, default=['conv1_1','conv2_1','conv3_1','conv4_1'],help='List of layers to use')
    parser.add_argument('--method', type=str, default='gram', help='The method for texture synthesis')
    parser.add_argument('--h', type=float, default=0.5, help='h lambdas')
    parser.add_argument('--patch_size', type=int, default=7, help='patch size')
    parser.add_argument('--lambda_progression', type=float, default=0, help='progression lambdas')
    parser.add_argument('--lambda_orientation', type=float, default=0, help='orientation lambdas')
    parser.add_argument('--lambda_occurrence', type=float, default=0.05, help='occurrence lambdas')
    parser.add_argument('--target_orientation_file', type=str, default='', help='target orientation')
    args = parser.parse_args()

    model=get_vgg19(args.pooling)
    target_img=load_image(args.image_path)
    if args.rescale:
        model=rescale_weights(model,[target_img])

    with torch.no_grad():
        # Reference orientation
        refer_orientation = None
        if args.lambda_orientation > 0:
            hogExtractor =DirectionFeatureExtractor().to(device)
            refer_orientation = hogExtractor(target_img.to(device))[0]
        # Target orientation
        target_orientation = None
        if args.lambda_orientation > 0:
            # target  --------------------
            target_orientation = np.load(args.target_orientation_file)
            target_orientation = torch.from_numpy(target_orientation).type(torch.float32).to(device)[None]
            target_orientation = F.interpolate(target_orientation, size=[224,224], mode='bilinear', align_corners=True)
            target_orientation = target_orientation / target_orientation.norm(dim=1, keepdim=True)
    orientations = [target_orientation, refer_orientation]

    if args.method == 'gram':
        texture_synthesis(model,target_img,args)
    elif args.method == 'cnnmrf':
        mrf_synthesis(model, target_img, args, orientations)

