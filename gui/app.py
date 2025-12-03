import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import IMAGENET_MEAN, IMAGENET_STD
from src.models import build_model
import pandas as pd
import requests
from urllib.parse import urlparse

@st.cache_resource
def load_model(model_name: str, ckpt_path: str):
    state = torch.load(ckpt_path, map_location='cpu')
    ckpt_model = state.get('model')
    if ckpt_model and ckpt_model != model_name:
        raise ValueError(f"Checkpoint architecture mismatch: selected '{model_name}' but checkpoint is for '{ckpt_model}'. Choose the matching model or a correct checkpoint.")
    classes = state.get('classes', [])
    num_classes = len(classes) if classes else 4
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    r = model.load_state_dict(state['state_dict'], strict=False)
    if getattr(r, 'missing_keys', []) or getattr(r, 'unexpected_keys', []):
        st.warning(f"State dict loaded with differences. Missing: {len(r.missing_keys)} Unexpected: {len(r.unexpected_keys)}")
    model.eval()
    return model, classes

def preprocess(img, img_size):
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tf(img).unsqueeze(0)

def predict(model, tensor):
    with torch.no_grad():
        out = model(tensor)
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        return probs

def predict_batch(model, batch):
    with torch.no_grad():
        out = model(batch)
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=1).cpu().numpy()
        return probs.mean(axis=0)

def _on_model_change():
    m = st.session_state.get('model_name', 'resnet50')
    st.session_state['ckpt_path'] = f"models/{m}_best.pt"

def _on_model_a_change():
    m = st.session_state.get('m1', 'resnet50')
    st.session_state['ckpt_a'] = f"models/{m}_best.pt"

def _on_model_b_change():
    m = st.session_state.get('m2', 'efficientnet_b0')
    st.session_state['ckpt_b'] = f"models/{m}_best.pt"

st.set_page_config(page_title='Material Classification', layout='wide')
st.title('Material Classification')

# initialize defaults
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = 'resnet50'
if 'ckpt_path' not in st.session_state:
    st.session_state['ckpt_path'] = f"models/{st.session_state['model_name']}_best.pt"

with st.sidebar:
    options = ['resnet50','efficientnet_b0','inception_v3']
    st.selectbox('Model', options, key='model_name', on_change=_on_model_change)
    st.text_input('Checkpoint path', key='ckpt_path')
    ckpt_exists = Path(st.session_state.get('ckpt_path', 'models/resnet50_best.pt')).exists()
    with st.expander('Advanced: manage checkpoint', expanded=not ckpt_exists):
        if not ckpt_exists:
            st.info('Checkpoint not found. Provide a URL or upload a .pt file.')
        ckpt_url = st.text_input('Checkpoint URL (optional)', placeholder='https://raw.githubusercontent.com/Alaashamel/Material_dataset/main/models/resnet50_best.pt')
        ckpt_upload = st.file_uploader('Upload checkpoint (.pt)', type=['pt'], key='ckpt_upload')
        if ckpt_upload is not None:
            try:
                out_path = Path(st.session_state.get('ckpt_path', 'models/resnet50_best.pt'))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'wb') as f:
                    f.write(ckpt_upload.getbuffer())
                st.success(f"Checkpoint uploaded to {out_path}")
            except Exception as e:
                st.error(str(e))
        if st.button('Download checkpoint') and ckpt_url:
            try:
                parsed = urlparse(ckpt_url)
                if parsed.scheme not in ('http', 'https'):
                    raise ValueError('Provide a full https URL (e.g., GitHub raw or Releases)')
                out_path = Path(st.session_state.get('ckpt_path', 'models/resnet50_best.pt'))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                r = requests.get(ckpt_url, stream=True, timeout=60)
                r.raise_for_status()
                with open(out_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.success(f"Checkpoint downloaded to {out_path}")
            except Exception as e:
                st.error(str(e))
    img_size = st.slider('Image size', min_value=128, max_value=512, value=224, step=32)
    show_cam = st.checkbox('Show Grad-CAM', value=False)
    conf_thresh = st.slider('Confidence threshold', 0.0, 1.0, 0.6, 0.05)

model_name = st.session_state.get('model_name', 'resnet50')
ckpt = st.session_state.get('ckpt_path', 'models/resnet50_best.pt')
img_file = st.file_uploader('Upload image', type=['jpg','jpeg','png','webp'])
tabs = st.tabs(["Inference", "Compare", "Confusion Matrix", "Webcam", "Grad-CAM Compare"])

with tabs[0]:
    if img_file and ckpt:
        img = Image.open(img_file).convert('RGB')
        try:
            col1, col2 = st.columns([3,2])
            with col1:
                st.image(img, caption='Input', use_container_width=True)
            model, classes = load_model(model_name, ckpt)
            tensor = preprocess(img, img_size)
            probs = predict(model, tensor)
            idx = np.argsort(probs)[::-1]
            labels = classes if classes else [str(i) for i in range(len(probs))]
            label_map = {"g": "glass", "m": "metal", "p": "plastic", "w": "wood"}
            readable = [label_map.get(labels[i], labels[i]) for i in idx]
            top3_idx = idx[:3]
            with col2:
                for i in top3_idx:
                    st.metric(readable[i], f"{probs[i]:.3f}")
                df = pd.DataFrame({'class': readable, 'prob': probs[idx]})
                df = df.set_index('class')
                st.bar_chart(df)
            if show_cam:
                from pytorch_grad_cam import GradCAM
                from pytorch_grad_cam.utils.image import show_cam_on_image
                rgb = np.array(img.resize((img_size, img_size))) / 255.0
                target_layers = _target_layers_for(model_name, model)
                cam = GradCAM(model=model, target_layers=target_layers)
                grayscale_cam = cam(input_tensor=tensor)
                overlay = show_cam_on_image(rgb, grayscale_cam[0], use_rgb=True)
                st.image(overlay, caption='Grad-CAM', use_container_width=True)
        except Exception as e:
            st.error(str(e))

with tabs[1]:
    models = ['resnet50','efficientnet_b0','inception_v3']
    rows = []
    for m in models:
        p = Path('docs') / f'{m}_metrics.csv'
        if p.exists():
            d = pd.read_csv(p)
            vals = dict(zip(d['metric'], d['value']))
            rows.append({'model': m, **vals})
    if rows:
        df = pd.DataFrame(rows)
        st.subheader('Model Comparison')
        st.dataframe(df, use_container_width=True)
        if {'model','accuracy'} <= set(df.columns):
            st.bar_chart(df.set_index('model')[['accuracy']])
        if {'model','f1'} <= set(df.columns):
            st.bar_chart(df.set_index('model')[['f1']])
    else:
        st.info('No evaluation files found in docs/. Run evaluation to compare models.')

with tabs[2]:
    sel = st.selectbox('Select model', ['resnet50','efficientnet_b0','inception_v3'], index=['resnet50','efficientnet_b0','inception_v3'].index(model_name))
    img_path = Path('docs') / f'{sel}_confusion_matrix.png'
    if img_path.exists():
        st.image(str(img_path), caption='Confusion Matrix', use_container_width=True)
    else:
        st.info('Confusion matrix not found. Run evaluation to generate it.')

with tabs[3]:
    cam_file = st.camera_input('Capture image')
    if cam_file and ckpt:
        img = Image.open(cam_file).convert('RGB')
        with st.expander('Webcam preprocessing'):
            zoom = st.slider('Zoom', 1.0, 2.0, 1.2, 0.05)
            bright = st.slider('Brightness', 0.5, 2.0, 1.0, 0.05)
            contrast = st.slider('Contrast', 0.5, 2.0, 1.0, 0.05)
            blur = st.slider('Background blur (px)', 0, 10, 0, 1)
            robust = st.checkbox('Use robust averaging (flip)', value=True)
        try:
            col1, col2 = st.columns([3,2])
            with col1:
                st.image(img, caption='Captured', use_container_width=True)
            model, classes = load_model(model_name, ckpt)
            def preprocess_webcam(pil_img, size, zoom_f, bright_f, contrast_f, blur_px):
                if blur_px > 0:
                    pil_img = pil_img.filter(ImageFilter.GaussianBlur(blur_px))
                if bright_f != 1.0:
                    pil_img = ImageEnhance.Brightness(pil_img).enhance(bright_f)
                if contrast_f != 1.0:
                    pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast_f)
                if zoom_f > 1.0:
                    w, h = pil_img.size
                    s = min(w, h)
                    crop_s = int(s / zoom_f)
                    left = (w - crop_s) // 2
                    top = (h - crop_s) // 2
                    right = left + crop_s
                    bottom = top + crop_s
                    pil_img = pil_img.crop((left, top, right, bottom))
                return preprocess(pil_img, size)
            tensor = preprocess_webcam(img, img_size, zoom, bright, contrast, blur)
            if robust:
                img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
                tensor_flip = preprocess_webcam(img_flip, img_size, zoom, bright, contrast, blur)
                batch = torch.cat([tensor, tensor_flip], dim=0)
                probs = predict_batch(model, batch)
            else:
                probs = predict(model, tensor)
            idx = np.argsort(probs)[::-1]
            labels = classes if classes else [str(i) for i in range(len(probs))]
            label_map = {"g": "glass", "m": "metal", "p": "plastic", "w": "wood"}
            readable = [label_map.get(labels[i], labels[i]) for i in idx]
            top3_idx = idx[:3]
            with col2:
                for i in top3_idx:
                    st.metric(readable[i], f"{probs[i]:.3f}")
                df = pd.DataFrame({'class': readable, 'prob': probs[idx]})
                df = df.set_index('class')
                st.bar_chart(df)
            top1_conf = float(probs[idx[0]])
            if top1_conf < conf_thresh:
                st.warning('Low confidence prediction. Try: move closer, center the material, reduce background clutter, improve lighting, adjust zoom/brightness/contrast above.')
            else:
                st.success('High confidence prediction. Capture looks good!')
        except Exception as e:
            st.error(str(e))

def _target_layers_for(model_name, model):
    if model_name == 'resnet50':
        return [model.layer4[-1]]
    if model_name == 'efficientnet_b0':
        try:
            return [model.features[-1][0]]
        except Exception:
            return [model.features[-1]]
    # inception_v3: be robust across torchvision versions
    inc = getattr(model, 'Mixed_7c', None)
    if inc is not None:
        for attr in ['branch3x3dbl_3', 'branch3x3_2', 'branch1x1']:
            if hasattr(inc, attr):
                return [getattr(inc, attr)]
        return [inc]
    inc = getattr(model, 'Mixed_7b', None)
    if inc is not None:
        for attr in ['branch3x3dbl_3', 'branch3x3_2', 'branch1x1']:
            if hasattr(inc, attr):
                return [getattr(inc, attr)]
        return [inc]
    return [model]

with tabs[4]:
    if 'm1' not in st.session_state:
        st.session_state['m1'] = model_name
    if 'ckpt_a' not in st.session_state:
        st.session_state['ckpt_a'] = ckpt
    if 'm2' not in st.session_state:
        st.session_state['m2'] = 'inception_v3'
    if 'ckpt_b' not in st.session_state:
        st.session_state['ckpt_b'] = f"models/{st.session_state['m2']}_best.pt"
    st.selectbox('Model A', ['resnet50','efficientnet_b0','inception_v3'], key='m1', on_change=_on_model_a_change)
    st.selectbox('Model B', ['resnet50','efficientnet_b0','inception_v3'], key='m2', on_change=_on_model_b_change)
    st.text_input('Checkpoint A', key='ckpt_a')
    st.text_input('Checkpoint B', key='ckpt_b')
    src_file = st.file_uploader('Grad-CAM image', type=['jpg','jpeg','png','webp'], key='cam_compare')
    if src_file and st.session_state['ckpt_a'] and st.session_state['ckpt_b']:
        img = Image.open(src_file).convert('RGB')
        rgb = np.array(img.resize((img_size, img_size))) / 255.0
        try:
            model_a, _ = load_model(st.session_state['m1'], st.session_state['ckpt_a'])
            model_b, _ = load_model(st.session_state['m2'], st.session_state['ckpt_b'])
            tensor = preprocess(img, img_size)
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            cam_a = GradCAM(model=model_a, target_layers=_target_layers_for(st.session_state['m1'], model_a))
            cam_b = GradCAM(model=model_b, target_layers=_target_layers_for(st.session_state['m2'], model_b))
            ga = cam_a(input_tensor=tensor)
            gb = cam_b(input_tensor=tensor)
            oa = show_cam_on_image(rgb, ga[0], use_rgb=True)
            ob = show_cam_on_image(rgb, gb[0], use_rgb=True)
            col1, col2 = st.columns(2)
            with col1:
                st.image(oa, caption=f"Grad-CAM {st.session_state['m1']}", use_container_width=True)
            with col2:
                st.image(ob, caption=f"Grad-CAM {st.session_state['m2']}", use_container_width=True)
        except Exception as e:
            st.error(str(e))
