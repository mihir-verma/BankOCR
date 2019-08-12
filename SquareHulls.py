import cv2

def get_square_hulls(img):
    
    #Resize the image so that MSER can work better
    #img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    
    mser = cv2.MSER_create()
    regions, bboxes = mser.detectRegions(gray)
    
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    
    sqr_hulls = list()
    
    for hull in hulls:
        
        approx = cv2.approxPolyDP(hull,0.01*cv2.arcLength(hull,True),True)
        
        if len(approx)==4:
            sqr_hulls.append(hull)
            #cv2.polylines(vis, hull, 1, (0, 255, 0))
    
    cv2.polylines(vis, sqr_hulls, 1, (0, 255, 0))
    
    return sqr_hulls, vis
    
