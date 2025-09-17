# CarciScan

CarciScan is a Python-based project for running machine learning and OCR algorithms on toxin datasets and images.

## Getting Started

Follow these steps to set up your environment and run the Python files:

### 1. Clone the Repository
If you haven't already, clone this repository:
```bash
git clone <repo-url>
cd CarciScan
```

### 2. Install Python
Ensure you have Python 3.8 or newer installed. You can check your version with:
```bash
python3 --version
```

### 3. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 4. Install Dependencies
If a `requirements.txt` file is present, install dependencies with:
```bash
pip install -r requirements.txt
```
If not, install required packages manually (e.g., scikit-learn, xgboost, pandas, numpy, opencv-python):
```bash
pip install scikit-learn xgboost pandas numpy opencv-python
```

### 5. Running Python Scripts
Navigate to the appropriate directory and run the desired script. For example:
```bash
cd test/algo
python randomforest_predict.py
python xgboost_predict.py

cd ../dataset
python classification.py
python validation.py

cd ../ocr
python benchmark.py
python preprocessing.py
```

### 6. Dataset and Images
- Datasets are in the `dataset/` folder.
- OCR images are in `test/ocr/images/`.
- Ground truth for OCR is in `test/ocr/ground_truth.json`.

### 7. Troubleshooting
- If you get `ModuleNotFoundError`, ensure you installed all required packages.
- Activate your virtual environment before running scripts.
- For issues with image processing, ensure OpenCV (`opencv-python`) is installed.

### 8. Deactivating the Environment
When finished, deactivate the virtual environment:
```bash
deactivate
```

---

For more details, check the source code in the `test/` and `dataset/` folders.

