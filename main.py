import cv2 as c
import numpy as np
import sys

cap = c.VideoCapture(0)

vid = c.VideoCapture('pup.mp4')

sucess, ref_img = cap.read()
flag = 0

while True:
    sucess, img = cap.read()
    if img is not None:
        img = c.resize(img,(1500,850),interpolation = c.INTER_AREA)
        ref_img = c.resize(ref_img,(1500,850),interpolation = c.INTER_AREA)
        img = c.flip(img, 1)
        sucess, bg = vid.read()
        try:
            bg = c.resize(bg,(1500,850),interpolation = c.INTER_AREA)
        except:
            break
        
        if flag == 0:
            ref_img = img
            
        diff1 = c.subtract(img,ref_img)
        diff2 = c.subtract(img,ref_img)
        diff = diff1 + diff2 
        
        diff[abs(diff)< 25] = 0
        
        gray = c.cvtColor(diff.astype(np.uint8),c.COLOR_BGR2GRAY)
        gray[np.abs(gray)>10] = 10
        
        fgmask = gray.astype(np.uint8)
        fgmask[fgmask > 0] = 255
        
        fgmask_inv = c.bitwise_not(fgmask)
        fgimg = c.bitwise_and(img,img,mask=fgmask)
        bgimg = c.bitwise_and(bg,bg,mask=fgmask_inv)
        
        dst = c.add(bgimg,fgimg)
        c.imshow('Background test', dst)
        key = c.waitKey(1) & 0xFF
        if ord('x') == key:
            break
        elif ord('d') == key:
            flag = 1
        elif ord('r') == key:
            flag = 0
c.destroyAllWindows()
cap.release()
