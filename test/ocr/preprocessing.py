import os
import cv2
import numpy as np

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Milder denoising (less detail loss)
    denoised = cv2.fastNlMeansDenoising(gray, h=7)

    # Gentler contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
    enhanced = clahe.apply(denoised)

    # Adaptive thresholding: larger block size, smaller C for less harsh binarization
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, 7
    )

    # Optional: very light morphological closing (smaller kernel)
    kernel = np.ones((1,1), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return closed

def process_all_images():
    for fname in os.listdir(IMAGES_DIR):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(IMAGES_DIR, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            processed = preprocess_image(img)
            out_path = os.path.join(PROCESSED_DIR, fname)
            cv2.imwrite(out_path, processed)
            print(f"Processed: {fname} -> {out_path}")

if __name__ == "__main__":
    process_all_images()
