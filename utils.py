import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(img):
    glcm = graycomatrix((img * 255).astype(np.uint8), distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
    return contrast, entropy

def get_stage_from_glcm_features(contrast, entropy, contrast_qs, entropy_qs, cancer_class):
    c_score = np.digitize(contrast, contrast_qs)
    e_score = np.digitize(entropy, entropy_qs)
    stage_score = (c_score + e_score) / 2

    if cancer_class == 0:
        return "Stage 1" if stage_score <= 1 else "Stage 2"
    elif cancer_class == 1:
        return "Stage 3" if stage_score <= 1 else "Stage 4"
    else:
        return "Normal"
