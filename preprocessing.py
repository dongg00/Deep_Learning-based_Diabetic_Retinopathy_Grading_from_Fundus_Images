# preprocessing.py
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def crop_image_to_circle(image):
    if image.ndim == 3: gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else: gray_image = image
    blurred = cv2.GaussianBlur(gray_image, (0,0), 10)
    _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return image
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center, radius = (int(x), int(y)), int(radius)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped_image, mask = image[y:y+h, x:x+w], mask[y:y+h, x:x+w]
    final_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
    return final_image

def ben_graham_preprocessing(image, sigmaX=10):
    return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)

def get_transforms(data_type, img_size):
    if data_type == 'train':
        return A.Compose([
            A.Resize(img_size, img_size), A.RandomRotate90(p=0.5), A.Flip(p=0.5), A.Transpose(p=0.5),
            A.ShiftScaleRotate(p=0.7), A.RandomBrightnessContrast(p=0.7), A.CoarseDropout(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2(),
        ])
    elif data_type == 'valid':
        return A.Compose([ A.Resize(img_size, img_size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2() ])