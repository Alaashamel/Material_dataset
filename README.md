# Material Classification Project

## Overview
CNN image classifier with transfer learning across multiple architectures and a Streamlit GUI.

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Data
Run split on the provided dataset folder:
```
python split_dataset.py --source_dir JPEGImages --output_dir data_split --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --infer_from_names
```

## Train
```
python -m src.train --data_dir data_split --model resnet50 --epochs 5
python -m src.train --data_dir data_split --model efficientnet_b0 --epochs 5
python -m src.train --data_dir data_split --model inception_v3 --epochs 5
```

## Evaluate
```
python -m src.evaluate --data_dir data_split --model resnet50 --checkpoint models/resnet50_best.pt
```

## GUI
```
streamlit run gui/app.py
```

## Repository
Commit and push after training and evaluation.
