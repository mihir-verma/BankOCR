import cv2

def find_ac_num(img, hulls):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    vis = img.copy()

    max_w = 0
    
    for i, contour in enumerate(hulls):
        
        x,y,w,h = cv2.boundingRect(contour)
        
        if w > max_w:
            max_x, max_y, max_w, max_h = x,y,w,h
    
    ac_num_img = img[max_y:max_y+max_h,max_x:max_x+max_w]
    cv2.rectangle(vis, (max_x, max_y), (max_x+max_w, max_y+max_h), (0, 255, 0), 2)
    
    return ac_num_img, vis