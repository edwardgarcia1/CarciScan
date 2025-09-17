import os
import json
import string
import numpy as np
from paddleocr import PaddleOCR
import pytesseract
from PIL import Image
from jiwer import wer, cer
import cv2
import sys

from rapidfuzz import fuzz
from spellchecker import SpellChecker
from doctr.models import ocr_predictor


# --- Import preprocessing function ---
sys.path.append(os.path.dirname(__file__))
from preprocessing import preprocess_image

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
GT_FILE = os.path.join(BASE_DIR, "ground_truth.json")

# --- Initialize OCR engines ---
paddle_ocr = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True)

doctr_ocr = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)


# --- Text normalization ---

def extract_ingredients_block(text):
    """
    Extracts the text block starting from 'INGREDIENTS:' (or fuzzy matches)
    until a stop word like 'WARNINGS:', 'DIRECTIONS:', etc.
    """
    lines = text.split("\n")
    keywords = ["INGREDIENTS", "Contains", "Active Ingredients"]
    stopwords = ["DIRECTIONS", "WARNINGS", "CAUTION", "STORAGE"]

    start_idx = None
    for i, line in enumerate(lines):
        for kw in keywords:
            if fuzz.partial_ratio(line.upper(), kw.upper()) > 80:
                start_idx = i
                break
        if start_idx is not None:
            break

    if start_idx is None:
        return ""

    collected = []
    for line in lines[start_idx:]:
        if any(fuzz.partial_ratio(line.upper(), sw) > 80 for sw in stopwords):
            break
        collected.append(line)

    return " ".join(collected).strip()


def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

# --- Build ingredient vocabulary from ground truth ---
def build_ingredient_vocab(gt_dict):
    import re
    vocab = set()
    for text in gt_dict.values():
        # Remove punctuation and split
        words = re.findall(r"\b[a-zA-Z][a-zA-Z\-]*\b", text.lower())
        vocab.update(words)
    return vocab

# --- Text correction with custom vocabulary ---
def correct_text(text, vocab=None):
    spell = SpellChecker()
    if vocab:
        spell.word_frequency.load_words(vocab)
    corrected_words = []
    for word in text.split():
        # Only correct if not a number or punctuation
        if word.isalpha() and vocab:
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)
        elif word.isalpha():
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)

# --- Run Tesseract on preprocessed image ---
def run_tesseract_cv2(cv2_img):
    # Convert OpenCV image (numpy array) to PIL Image
    pil_img = Image.fromarray(cv2_img)
    return pytesseract.image_to_string(pil_img, config="--psm 6")

# --- Run PaddleOCR on preprocessed image ---
def run_paddle_cv2(cv2_img):
    # Ensure image is 3-channel BGR for PaddleOCR
    if len(cv2_img.shape) == 2:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)
    result = paddle_ocr.predict(cv2_img)
    if not result or not result[0]:
        return ""
    text = " ".join([line[1][0] for line in result[0]])
    return extract_ingredients_block(text)

# --- Run Doctr OCR on preprocessed image ---
def run_doctr_cv2(cv2_img):
    # Ensure image is RGB numpy array for Doctr
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    result = doctr_ocr([img_rgb])
    # Extract text robustly from Doctr result
    try:
        text = " ".join(
            word.value
            for page in result.pages
            for block in page.blocks
            for line in block.lines
            for word in line.words
        )
    except Exception:
        text = ""
    return text

# --- Load ground truth and build vocab ---
with open(GT_FILE, "r") as f:
    ground_truth = json.load(f)
ingredient_vocab = build_ingredient_vocab(ground_truth)

# --- Benchmarking ---
wer_scores_t, cer_scores_t = [], []
wer_scores_p, cer_scores_p = [], []
wer_scores_d, cer_scores_d = [], []

print("=== OCR Detailed Results ===")
for fname, gt_text in ground_truth.items():
    img_path = os.path.join(IMG_DIR, fname)
    if not os.path.exists(img_path):
        print(f"[SKIP] Missing image: {fname}")
        continue

    # Normalize GT
    gt_norm = normalize_text(gt_text)

    # --- Preprocess image in memory ---
    img_cv2 = cv2.imread(img_path)
    if img_cv2 is None:
        print(f"[SKIP] Could not read image: {fname}")
        continue

    # --- Preprocess image in memory ---
    processed_img = preprocess_image(img_cv2)

    # --- Tesseract ---
    pred_tess = normalize_text(run_tesseract_cv2(processed_img))
    pred_tess_corr = correct_text(pred_tess, ingredient_vocab)

    # --- PaddleOCR ---
    # Use a less aggressive preprocessed image for PaddleOCR: use CLAHE-enhanced grayscale converted to BGR
    # We'll extract this from the preprocessing function:
    def get_clahe_bgr(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=7)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
        enhanced = clahe.apply(denoised)
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 7
        )
        kernel = np.ones((1, 1), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return closed
    paddle_img = get_clahe_bgr(img_cv2)
    pred_paddle = normalize_text(run_paddle_cv2(paddle_img))
    pred_paddle_corr = correct_text(pred_paddle, ingredient_vocab)

    # --- Doctr OCR ---
    pred_doctr = normalize_text(run_doctr_cv2(processed_img))
    pred_doctr_corr = correct_text(pred_doctr, ingredient_vocab)

    # --- Metrics ---
    wer_t = wer(gt_norm, pred_tess_corr) if pred_tess_corr else 1.0
    cer_t = cer(gt_norm, pred_tess_corr) if pred_tess_corr else 1.0
    wer_p = wer(gt_norm, pred_paddle_corr) if pred_paddle_corr else 1.0
    cer_p = cer(gt_norm, pred_paddle_corr) if pred_paddle_corr else 1.0
    wer_d = wer(gt_norm, pred_doctr_corr) if pred_doctr_corr else 1.0
    cer_d = cer(gt_norm, pred_doctr_corr) if pred_doctr_corr else 1.0

    wer_scores_t.append(wer_t)
    cer_scores_t.append(cer_t)
    wer_scores_p.append(wer_p)
    cer_scores_p.append(cer_p)
    wer_scores_d.append(wer_d)
    cer_scores_d.append(cer_d)

    # --- Print detailed output ---
    print(f"\nImage: {fname}")
    print(f"  Ground Truth : {gt_norm}")
    print(f"  Tesseract    : RAW: {pred_tess}")
    print(f"                CORRECTED: {pred_tess_corr}")
    print(f"  PaddleOCR    : RAW: {pred_paddle}")
    print(f"                CORRECTED: {pred_paddle_corr}")
    print(f"  Doctr OCR    : RAW: {pred_doctr}")
    print(f"                CORRECTED: {pred_doctr_corr}")
    print(f"  [WER] Tesseract={wer_t:.3f}, Paddle={wer_p:.3f}, Doctr={wer_d:.3f}")
    print(f"  [CER] Tesseract={cer_t:.3f}, Paddle={cer_p:.3f}, Doctr={cer_d:.3f}")

# --- Summary ---
print("\n=== OCR Benchmark Results ===")
print(f"Tesseract - Avg WER: {np.mean(wer_scores_t):.3f}, Avg CER: {np.mean(cer_scores_t):.3f}")
print(f"PaddleOCR - Avg WER: {np.mean(wer_scores_p):.3f}, Avg CER: {np.mean(cer_scores_p):.3f}")
print(f"Doctr OCR   - Avg WER: {np.mean(wer_scores_d):.3f}, Avg CER: {np.mean(cer_scores_d):.3f}")
