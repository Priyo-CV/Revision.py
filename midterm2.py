import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("C:\\Users\\USER\\Downloads\\wound.jpg")
grayScale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
    # kernel for morphologyEx
kernel = cv2.getStructuringElement(1,(17,17))
    
    # apply MORPH_BLACKHAT to grayScale image
blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    
    # apply thresholding to blackhat
_,threshold = cv2.threshold(blackhat,5,255,cv2.THRESH_BINARY)
    
    # inpaint with original image and threshold image
final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
    # Applying median blur
final_image = cv2.medianBlur(final_image,9)


Gaussian=cv2.GaussianBlur(final_image,(7,7),0)
img=cv2.subtract(final_image,Gaussian)
img1=cv2.add(final_image,img)
#cv2.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
#cv2.imshow(cv2.cvtColor(Gaussian,cv2.COLOR_BGR2RGB))
#img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
cv2.imshow('original image',image)
cv2.imshow('Final image',final_image)
cv2.imshow('Final 2',img1)

cv2.waitKey(0)
    
