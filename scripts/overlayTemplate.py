import numpy as np
import cv2
import math

def overlayTemplate(foreground, background):
    # get image dim
    hf, wf, _ = foreground.shape
    hb, wb, _ = background.shape
    # padd background image\
    pad_top = int(math.ceil(hf/2))
    pad_bottom = pad_top
    pad_left = int(math.ceil(wf/2))
    pad_right = pad_left
    background_padded = cv2.copyMakeBorder(background, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])
    # generate random position
    x=np.random.randint(0, wb-1)
    y=np.random.randint(0, hb-1)
    foreground_rand_pos = np.zeros_like(background_padded)
    foreground_rand_pos[y:y+hf,x:x+wf,:]=foreground
    bbox = np.array([x-pad_left, y-pad_top, wf, hf])
    # generate image mask: 1->foreground, 0->background
    mask = np.any(foreground_rand_pos!=0, axis=2).astype(np.float)
    # blur image mask
    mask_blurred = cv2.GaussianBlur(mask, ksize=(0,0), sigmaX=6)
    mask_blurred = (mask_blurred-0.5)/0.5
    mask_blurred = np.clip(mask_blurred, a_min=0, a_max=1.0)
    mask_blurred = np.expand_dims(mask_blurred, axis=2)
    # overlay image using alpha blending
    background_padded = background_padded.astype(np.float)
    foreground_rand_pos = foreground_rand_pos.astype(np.float)
    overlayed_img = background_padded.copy()
    overlayed_img[y:y+hf,x:x+wf,:] = background_padded[y:y+hf,x:x+wf,:]*(1-mask_blurred[y:y+hf,x:x+wf,:])+foreground_rand_pos[y:y+hf,x:x+wf,:]*mask_blurred[y:y+hf,x:x+wf,:]
    overlayed_img = overlayed_img.astype(np.uint8)
    # crop out border
    overlayed_img = overlayed_img[pad_top:pad_top+hb,pad_left:pad_left+wb,:]
    
    return overlayed_img, bbox