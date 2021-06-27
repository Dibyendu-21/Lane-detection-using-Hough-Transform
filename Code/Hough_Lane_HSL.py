# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:26:10 2021

@author: Sonu
"""

import cv2
import numpy as np

#Reading the images
img = cv2.imread(r'C:\Users\Sonu\Documents\Passion\Perception_Vision\Projects\Hough Transformation\HSL_Lane_Good\Lane_4.png')
img = cv2.resize(img, (960, 540))

#Converting image to HSL format
hsl_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
cv2.imshow("hsl image",hsl_img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite(r'C:\Users\Sonu\Documents\Passion\Perception_Vision\Projects\Hough Transformation\HSL_Lane_Good\Lane_4_hsl_image.png', hsl_img) 
    
#Isolating the yellow pixels and creating a mask out of it
def isolate_yellow_hsl(img):
    low_threshold = np.array([15, 38, 115], dtype=np.uint8)
    high_threshold = np.array([35, 204, 255], dtype=np.uint8)  
    yellow_mask = cv2.inRange(img, low_threshold, high_threshold)
    return yellow_mask

#Applying the isloate yellow mask on image and displaying
isolate_yellow_hsl_image = isolate_yellow_hsl(hsl_img)
cv2.imshow("isolated yellow hsl image", isolate_yellow_hsl_image)
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite(r'C:\Users\Sonu\Documents\Passion\Perception_Vision\Projects\Hough Transformation\HSL_Lane_Good\Lane_4_hsl_isolate_yellow.png', isolate_yellow_hsl_image) 
                            
#Isolating the white pixels and creating a mask out of it
def isolate_white_hsl(img):
    low_threshold = np.array([0, 200, 0], dtype=np.uint8)
    high_threshold = np.array([180, 255, 255], dtype=np.uint8)  
    white_mask = cv2.inRange(img, low_threshold, high_threshold)
    return white_mask

#Applying the isloate white mask on image and displaying
isolate_white_hsl_image = isolate_white_hsl(hsl_img)
cv2.imshow("isolated white hsl image", isolate_white_hsl_image)
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite(r'C:\Users\Sonu\Documents\Passion\Perception_Vision\Projects\Hough Transformation\HSL_Lane_Good\Lane_4_hsl_isolate_white.png', isolate_white_hsl_image) 

#Combining the isloate yellow and white mask and applying it on image
def combine_yw_isolated(img, hsl_img):
    hsl_yellow = isolate_yellow_hsl(hsl_img)
    hsl_white = isolate_white_hsl(hsl_img)
    hsl_mask = cv2.bitwise_or(hsl_yellow, hsl_white)
    return cv2.bitwise_and(img, img, mask=hsl_mask)

isolated_img = combine_yw_isolated(img, hsl_img)
cv2.imshow("combined isolated image",isolated_img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite(r'C:\Users\Sonu\Documents\Passion\Perception_Vision\Projects\Hough Transformation\HSL_Lane_Good\Lane_4_hsl_combined_isolated_image.png', isolated_img) 

#Creating a gray image
gray_isolated_img = cv2.cvtColor(isolated_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray isolated image",gray_isolated_img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite(r'C:\Users\Sonu\Documents\Passion\Perception_Vision\Projects\Hough Transformation\HSL_Lane_Good\Lane_4_gray_isolated_image.png', gray_isolated_img) 

#Blurring the image
def blur(gray_img, method="gaussian", kernel_size=5):
    if method == "gaussian":
      img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0) 
    return img

blur_isolated_img = blur(gray_isolated_img)
cv2.imshow("blur_isolated_img",blur_isolated_img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite(r'C:\Users\Sonu\Documents\Passion\Perception_Vision\Projects\Hough Transformation\HSL_Lane_Good\Lane_4_blur_isolated_image.png', blur_isolated_img) 


canny_isolated_img = cv2.Canny(blur_isolated_img, 50, 150)
cv2.imshow("canny_isolated_img",canny_isolated_img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite(r'C:\Users\Sonu\Documents\Passion\Perception_Vision\Projects\Hough Transformation\HSL_Lane_Good\Lane_4_canny_isolated_image.png', canny_isolated_img) 


def get_vertices_for_img(img):
    imshape = img.shape
    height = imshape[0]
    width = imshape[1]

    vert = None
    
    if (width, height) == (960, 540):
        region_bottom_left = (130 ,imshape[0] - 1)
        region_top_left = (410, 330)
        region_top_right = (650, 350)
        region_bottom_right = (imshape[1] - 30,imshape[0] - 1)
        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
    else:
        region_bottom_left = (200 , 680)
        region_top_left = (600, 450)
        region_top_right = (750, 450)
        region_bottom_right = (1100, 650)
        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)

    return vert

#Detecting the ROI to find the lanes and discarding other edges not part of lane
def region_of_interest(img):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
        
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    vert = get_vertices_for_img(img)    
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vert, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

canny_segmented_images = region_of_interest(canny_isolated_img)
cv2.imshow("canny_segmented_images",canny_segmented_images)
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite(r'C:\Users\Sonu\Documents\Passion\Perception_Vision\Projects\Hough Transformation\HSL_Lane_Good\Lane_4_canny_ROI_image.png', canny_segmented_images) 

#Applying hough transform on image to detect staright lines 
def hough_transform(canny_img,
                    rho=1,
                    theta=(np.pi/180)*1,
                    threshold=15,
                    min_line_len=20,
                    max_line_gap=10):
    return cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

hough_lines = hough_transform(canny_segmented_images)
print(hough_lines.shape)

#Drwaing the lines detected
def draw_lines(img, lines, color=[255, 0, 0], thickness=10, make_copy=True):
    # Copy the passed image
    img_copy = np.copy(img) if make_copy else img
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    
    return img_copy

hough_img = draw_lines(img, hough_lines)
cv2.imshow("hough_lines",hough_img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite(r'C:\Users\Sonu\Documents\Passion\Perception_Vision\Projects\Hough Transformation\HSL_Lane_Good\Lane_4_Hough_Lane_image.png', hough_img) 

#Separating the left and right lanes 
def separate_lines(lines, img):
    img_shape = img.shape
    
    middle_x = float(img_shape[1] / 2)
    
    left_lane_lines = []
    right_lane_lines = []
    c,l,r = 0,0,0
    for line in lines:
        print(line)
        for x1, y1, x2, y2 in line:
            dx = x2 - x1 
            if dx == 0:
                print('dx=0',line)
                #Discarding line since we can't gradient is undefined at this dx
                continue
            dy = y2 - y1
            
            # Similarly, if the y value remains constant as x increases, discard line
            if dy == 0:
                print('dy=0',line)
                continue
            
            slope = dy / dx
            
            # This is pure guess than anything... 
            # but get rid of lines with a small slope as they are likely to be horizontal one
            epsilon = 0.1
            if abs(slope) <= epsilon:
                print('slope less',line)
                c = c+1
                continue
            
            if x1 <= middle_x and x2 <= middle_x:
                # Lane should also be within the left hand side of region of interest
                left_lane_lines.append([[x1, y1, x2, y2]])
                print('left_lane',line)
                l = l+1
            elif x1 > middle_x and x2 > middle_x:
                # Lane should also be within the right hand side of region of interest
                print('right_lane',line)
                right_lane_lines.append([[x1, y1, x2, y2]])
                r = r+1
    print(c,l,r)
    return left_lane_lines, right_lane_lines

left_lane_lines, right_lane_lines = separate_lines(hough_lines, img)
print(np.array(left_lane_lines).shape)
print(np.array(right_lane_lines).shape)

#Coloring the lanes with different colors. The left lane in red, while those belonging to the right lane are in blue.
def color_lanes(img, left_lane_lines, right_lane_lines, left_lane_color=[255, 0, 0], right_lane_color=[0, 0, 255]):
    left_colored_img = draw_lines(img, left_lane_lines, color=left_lane_color, make_copy=False)
    right_colored_img = draw_lines(left_colored_img, right_lane_lines, color=right_lane_color, make_copy=False)
    return right_colored_img

img_different_lane_colors = color_lanes(img, left_lane_lines, right_lane_lines)
cv2.imshow("img_different_lane_colors",img_different_lane_colors)
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite(r'C:\Users\Sonu\Documents\Passion\Perception_Vision\Projects\Hough Transformation\HSL_Lane_Good\Lane_4_Different_Lane_Colors.png', img_different_lane_colors) 

