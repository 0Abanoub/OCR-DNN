# OCR Character Classifier

Small project implementing character-level OCR using PyTorch.

## Overview
- Data: local labeled character images (folder-per-class)
- Model: simple CNN baseline for character classification
- Notebook: `OCR.ipynb` contains data loading, preprocessing, training and evaluation
- Goal: build a robust character recognizer and later extend to sequence OCR (CRNN / CTC)

## How to run (locally)
1. Clone repo:
   git clone https://github.com/0Abanoub/OCR-DNN.git
   cd OCR-DNN

2. Create virtual env & install:
   python -m venv venv
   source venv/bin/activate
   --or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   
4. Place your data/ folder (not committed) in the project root:
   data/
     Training set/
     Testing set/
   
5. Open notebooks/OCR.ipynb and run cells.
