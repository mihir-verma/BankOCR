import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

import AutoCroppy
import DeSkew
import BrtCnt
import SquareHulls
import DateGen
import ACNum
import GetChars
import kNN

def main_process(data, i=0):    
    
    i = i+1 #for starting with 1
    
    #reading image:
    #image = cv2.imread('IIT/test/test01.png')
    image = cv2.imread(project_dir+'/test_data/'+data)
    
    #crop 1/3 if landscape:
    h = image.shape[0]
    w = image.shape[1]
    if h > w:
        crop_img = image[0:int(h/3), 0:w]
    else:
        crop_img = image
    
    #autocropping the black borders for better deskewing:
    ac = AutoCroppy.AutoCropper(crop_img)
    ac.max_border_size = 300 #def300
    ac.safety_margin = 10 #def4
    ac.tolerance = 4 #def4
    autocrop_img = ac.autocrop()

    #deskewing the image
    angle, deskew_img = DeSkew.deskew(autocrop_img)

    #adjusting brightness and contrast:
    brtcnt_img = np.zeros(deskew_img.shape, deskew_img.dtype)
    brtcnt_img = BrtCnt.apply_brightness_contrast(deskew_img, -50, 50)
    
    #getting square convex hulls from image:
    sqr_hulls, hulls_vis = SquareHulls.get_square_hulls(brtcnt_img)

    #getting the date image:
    date_img, date_vis = DateGen.date_gen(brtcnt_img, sqr_hulls)
    
    #once again trying to apply deskew:
    _, deskew_date = DeSkew.deskew(date_img)
    
    #saving the date img:
    cv2.imwrite(project_dir+'/res/dates/date_{}.png'.format(i), deskew_date)
    
    #getting a/c num image:
    ac_num_img, ac_num_vis = ACNum.find_ac_num(brtcnt_img, sqr_hulls)
    
    #once again trying to apply deskew:
    _, deskew_ac_num = DeSkew.deskew(ac_num_img)
    
    #saving the ac num img:
    cv2.imwrite(project_dir+'/res/ac_nums/ac_num_{}.png'.format(i), deskew_ac_num)
    
    #getting possible characters from date:    
    poss_chars_date = GetChars.get_chars(deskew_date)
    
    #getting possible characters from a/c number:    
    poss_chars_ac_num = GetChars.get_chars(deskew_ac_num)
    
    #building local directory for storing chars:
    if 'doc{}'.format(i) not in os.listdir(project_dir+'/res/chars'):
        os.mkdir(project_dir+'/res/chars/doc{}'.format(i))
        os.mkdir(project_dir+'/res/chars/doc{}/date'.format(i))
        os.mkdir(project_dir+'/res/chars/doc{}/ac_num'.format(i))

    #writing down the results:
    f.write('Document {}:\n'.format(i))
    
    #date
    f.write('Date:\n')
    for j, char in enumerate(poss_chars_date):
        cv2.imwrite(project_dir+'/res/chars/doc{0}/date/poss_char{1}.png'.format(i,j), char)        
        result = kNN.get_result(knn, char)
        f.write(result)
    f.write('\n')
    
    f.write('Bank A/C Num:\n')
    for j, char in enumerate(poss_chars_ac_num):
        cv2.imwrite(project_dir+'/res/chars/doc{0}/ac_num/poss_char{1}.png'.format(i,j), char)        
        result = kNN.get_result(knn, char)
        f.write(result)
    f.write('\n\n')

#setting the project directory:
project_dir = input('Please enter the complete directory of \'project\' folder (e.g. C:\downloads <do not end with a backslash>) (otherwise copy the folder into current working directory):\n')
if len(project_dir) == 0 :
    project_dir = os.getcwd() + '/project'
else:
    project_dir = project_dir + '/project'
    
#making a knn model:    
knn = kNN.make_knn(project_dir)

#making/opening the output file:    
f = open(project_dir+'/Out_Data.txt', 'w')

#setting the document to be worked on:
action_doc = input('Give the name of test file with full directory path (leave blank for default data set): \n')
if action_doc == True:
    main_process(action_doc)
else:
    test_data_files = os.listdir(project_dir + '/test_data' )
    for i, data in enumerate(test_data_files):
        main_process(data, i)
        
#closing the output file:
f.close()
