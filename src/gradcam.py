import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .models import build_model

def preprocess(img_path, img_size):
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert('RGB')
    tensor = tf(img).unsqueeze(0)
    rgb = np.array(img.resize((img_size, img_size))) / 255.0
    return tensor, rgb

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img', type=str, required=True)
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--class_index', type=int, default=None)
    p.add_argument('--out', type=str, default='docs/gradcam.png')
    args = p.parse_args()

    state = torch.load(args.checkpoint, map_location='cpu')
    classes = state.get('classes', [])
    num_classes = len(classes) if classes else 4
    model = build_model(args.model, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state['state_dict'])
    target_layers = []
    if args.model == 'resnet50':
        target_layers = [model.layer4[-1]]
    elif args.model == 'efficientnet_b0':
        target_layers = [model.features[-1][0]]
    elif args.model == 'inception_v3':
        target_layers = [model.Mixed_7c.branch7x7_3]
    cam = GradCAM(model=model, target_layers=target_layers)
    tensor, rgb = preprocess(args.img, args.img_size)
    targets = None
    if args.class_index is not None:
        targets = [ClassifierOutputTarget(args.class_index)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)
    overlay = show_cam_on_image(rgb, grayscale_cam[0], use_rgb=True)
    Image.fromarray(overlay).save(args.out)

if __name__ == '__main__':
    main()
