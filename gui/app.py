import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
from src.models import build_model

@st.cache_resource
def load_model(model_name: str, ckpt_path: str):
    state = torch.load(ckpt_path, map_location='cpu')
    classes = state.get('classes', [])
    num_classes = len(classes) if classes else 4
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model, classes

def preprocess(img, img_size):
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])
    return tf(img).unsqueeze(0)

def predict(model, tensor):
    with torch.no_grad():
        out = model(tensor)
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        return probs

st.title('Material Classification')
model_name = st.selectbox('Model', ['resnet50','efficientnet_b0','inception_v3'])
ckpt = st.text_input('Checkpoint path', 'models/resnet50_best.pt')
img_file = st.file_uploader('Upload image', type=['jpg','jpeg','png','webp'])
img_size = st.slider('Image size', min_value=128, max_value=512, value=224, step=32)

if img_file and ckpt:
    img = Image.open(img_file).convert('RGB')
    try:
        model, classes = load_model(model_name, ckpt)
        tensor = preprocess(img, img_size)
        probs = predict(model, tensor)
        top3_idx = np.argsort(probs)[-3:][::-1]
        st.image(img, caption='Input', use_column_width=True)
        for i in top3_idx:
            label = classes[i] if classes else str(i)
            st.write(f"{label}: {probs[i]:.3f}")
    except Exception as e:
        st.error(str(e))
