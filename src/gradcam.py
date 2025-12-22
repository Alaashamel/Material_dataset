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
    p.add_argument('--img', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--class_index', type=int, default=None)
    p.add_argument('--out', default='gradcam.png')
    args = p.parse_args()

    state = torch.load(args.checkpoint, map_location='cpu')
    classes = state.get('classes', [])
    model = build_model(args.model, len(classes), pretrained=False)
    model.load_state_dict(state['state_dict'])
    model.eval()

    if args.model == 'resnet50':
        target_layers = [model.layer4[-1]]
    else:
        raise ValueError("GradCAM supported only for resnet50 here")

    cam = GradCAM(model, target_layers)
    tensor, rgb = preprocess(args.img, args.img_size)

    targets = [ClassifierOutputTarget(args.class_index)] if args.class_index is not None else None
    grayscale_cam = cam(tensor, targets)[0]
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    Image.fromarray(overlay).save(args.out)


if __name__ == '__main__':
    main()
