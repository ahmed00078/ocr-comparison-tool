# OCR Model Comparison Tool - Requirements and Installation Guide

## Overview
This guide helps you install and set up all dependencies needed for the OCR Model Comparison Tool.

## Requirements
- Python 3.7+ 
- pip package manager

## Installation Steps

### 1. Create a virtual environment (recommended)
```bash
python -m venv ocr-env
source ocr-env/bin/activate  # On Windows: ocr-env\Scripts\activate
```

### 2. Install base dependencies
```bash
pip install numpy matplotlib pandas pillow opencv-python tqdm seaborn requests
```

### 3. Install OCR libraries

#### Tesseract OCR
First, install Tesseract on your system:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki

Then install the Python wrapper:
```bash
pip install pytesseract
```

#### EasyOCR
```bash
pip install easyocr
```

#### Kraken
```bash
pip install kraken
```

#### TrOCR (Transformer-based OCR)
```bash
pip install transformers
pip install torch torchvision
```

#### PaddleOCR (Optional)
```bash
pip install paddlepaddle
pip install paddleocr
```

### 4. Install evaluation metrics
```bash
pip install python-Levenshtein
```

## Testing the Installation

Run a basic test to ensure everything is installed correctly:

```bash
python -c "import cv2; import pytesseract; import easyocr; import kraken; import torch; import numpy; import pandas; print('All dependencies installed successfully!')"
```

## Preparing Test Data

1. Create directories for your test images and ground truth:
```bash
mkdir -p test_data/images
mkdir -p test_data/ground_truth
```

2. Place your test images in `test_data/images/`
3. For each image, create a corresponding text file with the same name (but .txt extension) in `test_data/ground_truth/` containing the expected text

## Running the Tool

Basic usage:
```bash
python ocr_comparator.py --images test_data/images --ground-truth test_data/ground_truth --output results
```

To test specific models only:
```bash
python ocr_comparator.py --models tesseract easyocr --images test_data/images --ground-truth test_data/ground_truth
```

To download sample test data:
```bash
python ocr_comparator.py --download-samples
```

## Troubleshooting

### Common Issues

1. **Tesseract not found error**:
   - Make sure Tesseract is installed on your system
   - Set the Tesseract executable path: `pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'`

2. **CUDA/GPU issues**:
   - Some models like EasyOCR can use GPU acceleration
   - If GPU errors occur, try forcing CPU mode:
     ```python
     reader = easyocr.Reader(['en'], gpu=False)
     ```

3. **Memory errors**:
   - For large images, try resizing them before processing
   - Process fewer images at a time

### Getting Help
If you encounter issues, check the documentation for the specific OCR library:
- [Tesseract Documentation](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [Kraken Documentation](https://github.com/mittagessen/kraken)
- [TrOCR Documentation](https://huggingface.co/microsoft/trocr-base-printed)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)