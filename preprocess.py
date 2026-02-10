import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from shutil import copy2
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_patient_id(filename, label):
    
    basename = os.path.basename(filename)
    if label == 'PNEUMONIA':
        if 'person' in basename:
            return basename.split('_')[0]
    else: # NORMAL
        if 'NORMAL2' in basename:
            return basename.split('-')[0] + '-' + basename.split('-')[1]
        elif 'IM-' in basename:
            return basename.split('-')[0] + '-' + basename.split('-')[1]
    return basename # Fallback

def remove_bad_samples(img):
    
    if img is None: return False
    if np.std(img) < 5: return False 
    if img.shape[0] < 100 or img.shape[1] < 100: return False
    return True

def apply_enhancement(img):
    
    noise_sigma = np.std(cv2.Laplacian(img, cv2.CV_64F))
    
    if noise_sigma > 20: 
        img = cv2.bilateralFilter(img, 9, 75, 75)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    return img


ENABLE_SEGMENTATION = False  

def segment_lungs(img):
    
    if not ENABLE_SEGMENTATION:
        return img
        
    h, w = img.shape
    
    small = cv2.resize(img, (256, 256))
    
    _, body_mask = cv2.threshold(small, 20, 255, cv2.THRESH_BINARY)
  
    lungs = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 101, 2)
    
    lungs = cv2.bitwise_and(lungs, body_mask)
    
    kernel = np.ones((5,5), np.uint8)
    lungs = cv2.morphologyEx(lungs, cv2.MORPH_OPEN, kernel)
    lungs = cv2.morphologyEx(lungs, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(lungs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_mask = np.zeros_like(small)
    relevant_contours = []
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
       
        if area > (256 * 256 * 0.05): #
            if y > (256 * 0.1) and (y + ch) < (256 * 0.9):
                relevant_contours.append(cnt)
    
    if len(relevant_contours) >= 1:
       
        cv2.drawContours(final_mask, relevant_contours, -1, 255, -1)
        
        final_mask = cv2.dilate(final_mask, kernel, iterations=2)
        
        mask = cv2.resize(final_mask, (w, h))
        return cv2.bitwise_and(img, img, mask=mask)
        
    
    logging.warning("Lung segmentation failed to find reasonable contours. Returning full image.")
    return img

def resize_with_padding(img, target_size=384):
    
    h, w = img.shape
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def main(sample_mode=True):
    raw_dir = 'chest_xray'
    output_dir = 'processed_dataset'
    target_size = 384
    
    all_files = []
    for split in ['train', 'test', 'val']:
        for label in ['NORMAL', 'PNEUMONIA']:
            path = os.path.join(raw_dir, split, label, '*.jpeg')
            files = glob(path)
            for f in files:
                all_files.append({'path': f, 'label': label, 'patient_id': get_patient_id(f, label)})
    
    df = pd.DataFrame(all_files)
    if sample_mode:
        logging.info("Running in SAMPLE MODE (subset of data)")
        df = df.sample(min(50, len(df)), random_state=42)
    
    logging.info(f"Processing {len(df)} images.")

    patients = df['patient_id'].unique()
    train_p, test_val_p = train_test_split(patients, test_size=0.2, random_state=42)
    val_p, test_p = train_test_split(test_val_p, test_size=0.5, random_state=42)
    
    split_map = {p: 'train' for p in train_p}
    split_map.update({p: 'val' for p in val_p})
    split_map.update({p: 'test' for p in test_p})
    
    df['split'] = df['patient_id'].map(split_map)

    for _, row in df.iterrows():
        img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
        
        if not remove_bad_samples(img):
            logging.warning(f"Skipping bad sample: {row['path']}")
            continue
 
        img = apply_enhancement(img)
        img = segment_lungs(img)
        img = resize_with_padding(img, target_size=target_size)

        save_path = os.path.join(output_dir, row['split'], row['label'])
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path, os.path.basename(row['path'])), img)

    logging.info("Preprocessing complete!")

if __name__ == "__main__":
    main(sample_mode=False)
