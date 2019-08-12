import cv2
import os

def get_chars(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    
    mser = cv2.MSER_create()
    regions, bboxes = mser.detectRegions(gray)
    
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    
    sqr_hulls = list()    
    
    for hull in hulls:
        
        approx = cv2.approxPolyDP(hull,0.09999*cv2.arcLength(hull,True),True)
        
        if len(approx)==4:
            sqr_hulls.append(hull)
            
    filter_hulls = list()
    filter_area_min = 520
    filter_area_max = 650
    
    for hull in sqr_hulls:
        if cv2.contourArea(hull) > filter_area_min and cv2.contourArea(hull) < filter_area_max:
            filter_hulls.append(hull)
    
    cv2.polylines(vis, filter_hulls, 1, (0,255,0))
     
    poss_chars = list()   
    for i, hull in enumerate(filter_hulls):
        x,y,w,h = cv2.boundingRect(hull) 
        possible_char = img[y:y+h,x:x+w]
        poss_chars.append(possible_char)
    
    return poss_chars