import cv2

def grouper(iterable, interval=2):
    prev = None
    group = []
    for item in iterable:
        if not prev or abs(item[1] - prev[1]) <= interval:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group
        
def date_gen(img, hulls):
  
    vis = img.copy()
        
    heights = list()
    bboxes_list = list()
    
    for i, contour in enumerate(hulls):
        
        x,y,w,h = cv2.boundingRect(contour)
        
        img_h = img.shape[0]
        img_w = img.shape[1] 
        
        if (x > 3*(img_w/4) and x < img_w) and (y > 0 and y < img_h/4):
            bboxes_list.append([x, y, x + w, y + h])
            heights.append(h)
    
    heights = sorted(heights)
    median_height = heights[int(len(heights) / 2)] / 2
    
    bboxes_list = sorted(bboxes_list, key=lambda k: k[1])  # Sort the bounding boxes based on y1 coordinate ( y of the left-top coordinate )
    
    combined_bboxes = grouper(bboxes_list, median_height)  # Group the bounding boxes

    for group in combined_bboxes:
        x_min = min(group, key=lambda k: k[0])[0]  # Find min of x1
        x_max = max(group, key=lambda k: k[2])[2]  # Find max of x2
        y_min = min(group, key=lambda k: k[1])[1]  # Find min of y1
        y_max = max(group, key=lambda k: k[3])[3]  # Find max of y2
        
        date_img = img[y_min:y_max,x_min:x_max]
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        break #to return just the first box
    
    return date_img, vis