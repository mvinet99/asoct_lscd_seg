# Mask creation script for the AS-OCT pipeline
import math
import os
import scipy
import random
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line
from skimage.draw import line
from scipy.signal import savgol_filter
from math import atan
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def crop_image(image, left, top, right, bottom):
    """
    Function to crop image to crossline region
    """

    image2 = Image.fromarray(image)
    cropped = image2.crop((left, top, right, bottom)) 

    return np.array(cropped)

def crop_box(image, x, y):
    """
    Function to crop image to crossline region
    """

    image2 = Image.fromarray(image)

    top = y-50
    left = x-75
    bottom = y+100
    right = x+75
    cropped = image2.crop((left, top, right, bottom))

    # Return the image and (bottomleftcorner, bottomrightcorner, toprightcorner, topleftcorner, bottomleftcorner) points of the box
    return cropped, np.array([left, right, right, left, left]), np.array([bottom, bottom, top, top, bottom])

def crop_color(image):
    """
    Function to crop image to crossline region
    """

    image2 = Image.fromarray(image)
  
    top = 143
    left = 440
    bottom = 174
    right = 500
    cropped = image2.crop((left, top, right, bottom))
    return cropped

def findAngle(M1, M2):
    """
    Function to find the angle between two lines given their slopes M1 and M2.
    """

    PI = 3.14159265

    # Store the tan value  of the angle
    angle = abs((M2 - M1) / (1 + M1 * M2))
 
    # Calculate tan inverse of the angle
    ret = atan(angle)
 
    # Convert the angle from
    # radian to degree
    val = (ret * 180) / PI
 
    return (round(val, 4))

def rotate_image(image, angle, image_center):
    """
    Function to rotate image by a given angle around a specified center point.
    """
    
    # Rotate image angle
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    
    # Warp image to correct position
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)

    return result, rot_mat

def get_line(x1, y1, x2, y2):
    """
    Function to get the coordinates of a line between two points using Bresenham's algorithm.
    """

    # Find coordinates of line given start and end coordinates
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax

    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()

    return points

def extend_line(p1, p2, distance):
    """
    Extend a line defined by two points (p1 and p2) by a given distance in both directions.
    """

    # Extend line based on start and end points, given distance
    diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    p3_x = int(p1[0] + distance*np.cos(diff))
    p3_y = int(p1[1] + distance*np.sin(diff))
    p4_x = int(p1[0] - distance*np.cos(diff))
    p4_y = int(p1[1] - distance*np.sin(diff))

    return ((p3_x, p3_y), (p4_x, p4_y))

def line_intersection(line1, line2):
    """
    Function to find the intersection of two lines given their endpoints.
    """
    
    # Find intersection of lines
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def diff_lines(file, data_path):
    """
    Function to find the top and bottom layers from edge detection images
    using edge detection and Hough transform.
    """
    edge_im = cv.imread(data_path+'edge/' + file, cv.IMREAD_GRAYSCALE)  
    edge_im = cv.resize(edge_im, dsize=(2200, 820), interpolation=cv.INTER_CUBIC)
    edge_im[edge_im != 0] = 1

    # Identify the objects
    Zlabeled,Nlabels = ndimage.measurements.label(edge_im)
    label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]

    # Remove the labels
    for label,size in enumerate(label_size):
        if size < 1000:
            edge_im[Zlabeled == label] = 0
    edge_im_bool = edge_im > 0

    edge_im = skeletonize(edge_im)

    # Hough transform - vertical
    tested_angles = np.linspace(-np.pi/32, np.pi/32, 180, endpoint=False)
    lines = probabilistic_hough_line(edge_im, threshold=1, line_length=3,
                                    line_gap=1, theta=tested_angles)

    # Remove the pixels found by Hough transform
    rr_s = []
    rr_c = []

    for coord in lines:
        p0, p1 = coord
        rr, cc = line(p0[1], p0[0], p1[1], p1[0])
        edge_im[rr, cc] = 0
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=2)

        rr_s.append(rr)
        rr_c.append(cc)

    # Hough transform - horizontal
    tested_angles = np.linspace(-np.pi/2, np.pi/2, 180, endpoint=False)

    lines = probabilistic_hough_line(edge_im, threshold=1, line_length=1,
                                    line_gap=1, theta=tested_angles)

    # Remove the pixels found by Hough transform
    rr_s = []
    rr_c = []
    for coord in lines:
        p0, p1 = coord
        rr, cc = line(p0[1], p0[0], p1[1], p1[0])
        edge_im[rr, cc] = 255

        rr_s.append(rr)
        rr_c.append(cc)

    x_list = list(np.unique(np.where(edge_im == 1)[1]))

    # Find the top values at each x-coordinate
    new_im = np.zeros([edge_im.shape[0], edge_im.shape[1]])
    for x_val in x_list:
        y_val = np.min(np.where(edge_im[:,x_val] == 1)[0])
        new_im[y_val, x_val] = 1
    
    # Find the bottom values at each x-coordinate
    y_val2s = []
    for x_val2 in x_list:
        y_val2s.append(np.max(np.where(edge_im[:,x_val2] == 1)[0]))
    bot_pixel_y = y_val2s[int(round(len(y_val2s)/4))]
    bot_pixel_x = x_list[int(round(len(y_val2s)/4))]

    edge_im = new_im

    # Dilate to connect any remaining objects
    kernel = np.ones((10,20), np.uint8)
    edge_im = edge_im.astype(int).astype('uint8')
    edge_im = cv.dilate(edge_im, kernel, iterations=1)

    edge_im = skeletonize(edge_im)

    kernel = np.ones((3,3), np.uint8)
    edge_im = edge_im.astype(int).astype('uint8')
    edge_im = cv.dilate(edge_im, kernel, iterations=2)

    # Identify the objects
    Zlabeled,Nlabels = ndimage.measurements.label(edge_im)
    label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]

    # Remove the labels
    for label,size in enumerate(label_size):
        if size < 750: 
            edge_im[Zlabeled == label] = 0  
    
    # Remove objects again: if any y-value is above 800 in the object, remove the entire object
    # Identify the objects
    Zlabeled,Nlabels = ndimage.measurements.label(edge_im)
    label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
    for label in range(Nlabels+1):
        if np.max(np.unique(np.where(Zlabeled==label)[0])) > 800 and label_size[label] < 2000:
            edge_im[Zlabeled == label] = 0  

    # Remove objects again: if any y-value is below 100 in the object, remove the entire object
    # Identify the objects
    Zlabeled,Nlabels = ndimage.measurements.label(edge_im)
    label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
    for label in range(Nlabels+1):
        if np.min(np.unique(np.where(Zlabeled==label)[0])) < 10:
            edge_im[Zlabeled == label] = 0  

    kernel = np.ones((10,10), np.uint8)
    edge_im = edge_im.astype(int).astype('uint8')
    edge_im = cv.dilate(edge_im, kernel, iterations=1)
    edge_im = cv.erode(edge_im, kernel, iterations=1)

    Zlabeled,Nlabels = ndimage.measurements.label(edge_im)
    label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
    label_mins = []
    label_maxs = []

    # Find the largest object, and the average x-value of each object
    label_avgs = []
    for label in range(Nlabels+1):
        label_mins.append(np.min(np.unique(np.where(Zlabeled==label)[1])))
        label_maxs.append(np.max(np.unique(np.where(Zlabeled==label)[1])))
        label_avgs.append((statistics.mean(np.unique(np.where(Zlabeled==label)[1]))))
    
    label_mins = label_mins[1:len(label_mins)]
    label_maxs = label_maxs[1:len(label_maxs)]
    label_avgs = label_avgs[1:len(label_avgs)]

    if len(label_mins) != 0:
        diff = []
        for i in range(len(label_mins)):
            diff.append(label_maxs[i] - label_mins[i])
        max_idx = diff.index(max(diff))
    else:
        max_idx = 0

    # Find the min and max of the largest object
    lar_ob_min = label_mins[max_idx]
    lar_ob_max = label_maxs[max_idx]

    # For each other object, find the value of the distance (min or max) 
    # that is closest to either the min or max of the largest object
    deletes = []
    for i in range(len(label_mins)):
        com1 = abs(lar_ob_min - label_maxs[i])
        com2 = abs(lar_ob_max - label_mins[i])
        com3 = abs(lar_ob_max - label_maxs[i])
        com4 = abs(lar_ob_min - label_mins[i])
        min_dist = min([com1, com2, com3, com4])

        if min_dist >= 1 and label_avgs[i] >= 1900:
            deletes.append(1)
        elif min_dist >= 1 and label_avgs[i] <= 300:
            deletes.append(1)
        else:
            deletes.append(0)

    deletes = [0] + deletes
    # Delete the object that match the criteria
    i = 0
    for label in range(Nlabels+1):
        if deletes[i] == 1:
            edge_im[Zlabeled == label] = 0
        i = i + 1

    edge_im = skeletonize(edge_im)

    # Find all unique x-coordinates of edge detection layers
    x_list = list(np.unique(np.where(edge_im == 1)[1]))

    # Find the top values at each x-coordinate
    x_positions = []
    y_positions = []
    for x_val in x_list:
        y_val = np.min(np.where(edge_im[:,x_val] == 1)[0])
        x_positions.append(x_val)
        y_positions.append(y_val+1)

    smoothed_y = savgol_filter(y_positions, window_length=300, polyorder=4)

    return [x_positions], [smoothed_y], bot_pixel_x, bot_pixel_y

def LoG_masks(file, data_path, severities, file_names, epi_thicknesses):
    """
    Create the top and bottom masks for the AS-OCT images based on the severity and epithelial thickness.
    """

    file_name = file[0:-6]+'.png'
    
    severity = severities[file_names.index(file_name)]

    epi_thickness = epi_thicknesses[file_names.index(file_name)]
    
    # Ranges based on spreadsheet of values from clinical values
    if severity == 'control' or severity == 'Control': # Control - max 64, min 49.5
        if epi_thickness == ' ':
            thickness = random.randint(49.5, 64)
        else:
            thickness = float(epi_thickness)

    elif severity == 'mild' or severity == 'Mild': # Mild - max 55, min 34
        if epi_thickness == ' ':
            thickness = random.randint(34, 62)
        else:
            thickness = float(epi_thickness)

    elif severity == 'moderate' or severity == 'Moderate': # Moderate - max 41, min (without 0s) - 21
        if epi_thickness == ' ':
            thickness = random.randint(21, 41)
        else:
            thickness = float(epi_thickness)

    elif severity == 'severe' or severity == 'Severe': # Severe 0 max 39, min (without 0s) - 1
        if epi_thickness == ' ':
            result = random.choice([True, False])
            if result == True:
                thickness = random.randint(1, 39)
            else:
                thickness = 0
        else:
            thickness = float(epi_thickness)

    pixel_thick = int(round(thickness*(72/200)))
    
    if pixel_thick < 5:
        tck_pix = range(0,pixel_thick+1)
    else:
        tck_pix = range(pixel_thick-5,pixel_thick+6)

    img = cv.imread(data_path+'/resized/'+file)
    img = cv.resize(img, dsize=(2200, 820), interpolation=cv.INTER_CUBIC)

    # Creation of Laplacian of Gaussian image
    pr_img = scipy.ndimage.gaussian_laplace(img, sigma = 1)

    x_coords, y_coords, bot_pixel_x, bot_pixel_y = diff_lines(file, data_path)

    # For each pixel between the arcs, find the coordinates of the pixels of that arc
    x_coord = x_coords[0]
    y_coord_top = round(min(y_coords[0]))
    y_coord_bot = bot_pixel_y

    y_list = np.ndarray.tolist(np.linspace(y_coord_bot, y_coord_top, num=((y_coord_bot-y_coord_top))))

    y_list = list(reversed(y_list))
    y_list = list(map(int, y_list))

    sums = []
    differences = []
    for pixel in tck_pix:
        
        # Compute difference between top and here
        differences.append(pixel)

        # Find the coordinates of the arc here
        new_y_coords = [math.floor(y + pixel) for y in y_coords[0]]


        # Find the sum of the coordinates of the arc here
        sum = 0
        i = 0
        
        for val in new_y_coords:
            if x_coords[0][i] < pr_img.shape[1]:
                if val < pr_img.shape[0]:
                    sum = sum + (pr_img[val, x_coords[0][i]][0])/255
                    i = i + 1

        # Save sum to list
        sums.append(sum)

    # Choose the arc that has the most amount of pixels that are 255
    maximum = max(sums)
    maximum_pos = np.where(sums == np.max(sums))[0][0]

    # Define final output bottom layer
    shifted_y = []
    for val in y_coords[0]:
        shifted_y.append(int(val+(differences[maximum_pos])))

    return x_coords, y_coords, [shifted_y], img, thickness

def draw_masks(im, file, hulls_upper_x, hulls_upper_y, hulls_shifted_y, thickness, data_path, result_path):
    """
    Draw the masks on the original image and save the output images.
    """

    # Draw the image with just the outline of the upper mask
    img = np.zeros([im.shape[0], im.shape[1]])
    
    my_dpi = 77
    fig, ax = plt.subplots(1, figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi))
    ax.set_aspect('equal')
    ax.imshow(img)
    ax.axis('off')

    for i in range(len(hulls_upper_x)):
        ax.plot(hulls_upper_x[i], hulls_upper_y[i], color = 'yellow', linestyle='solid')

    # Save mask image to file
    plt.savefig(result_path+'/blank_mask_outlines/' + file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # draw the image with the original image and the outline of the upper mask
    img = cv.imread(data_path+'/normal/' + file)
    img = cv.resize(img, dsize=(2200, 820), interpolation=cv.INTER_CUBIC)
   
    my_dpi = 77
    fig, ax = plt.subplots(1, figsize=(img.shape[1]/my_dpi, img.shape[0]/my_dpi))
    ax.set_aspect('equal')
    ax.imshow(img)
    ax.axis('off')
    
    if thickness != 0:
        for i in range(len(hulls_shifted_y)):
            if len(hulls_shifted_y[i]) > len(hulls_upper_y[i]):
                ax.plot(hulls_upper_x[i], hulls_shifted_y[i][0:len(hulls_upper_y[i])], color = 'yellow', linestyle='solid')
            else:
                ax.plot(hulls_upper_x[i][0:len(hulls_shifted_y[i])], hulls_shifted_y[i], color = 'yellow', linestyle='solid')
                ax.plot(hulls_upper_x[i], hulls_upper_y[i], color = 'yellow', linestyle='solid')

    # Save mask image to file
    plt.savefig(result_path + '/mask_overlays_outlines/' + file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Draw the image with just the mask filled in
    my_dpi = 77
    fig, ax = plt.subplots(1, figsize=(im.shape[1]/my_dpi, im.shape[0]/my_dpi))
    ax.set_aspect('equal')
    ax.imshow(np.zeros([im.shape[0],im.shape[1]]))
    ax.axis('off')

    if thickness != 0:
            # Yellow class: EP layer
        for i in range(len(hulls_shifted_y)):
            if len(hulls_shifted_y[i]) > len(hulls_upper_y[i]):
                ax.fill_between(x= hulls_upper_x[i], y1= hulls_upper_y[i], y2= hulls_shifted_y[i][0:len(hulls_upper_y[i])], color = 'yellow')
            else:
                ax.fill_between(x= hulls_upper_x[i][0:len(hulls_shifted_y[i])], y1= hulls_upper_y[i][0:len(hulls_shifted_y[i])], y2= hulls_shifted_y[i], color = 'yellow')

    # Save mask image to file
    plt.savefig(result_path + 'mask_slices/' + file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return

def middle_points(hulls_upper_x, hulls_upper_y, swap):
    # Find the midway points between the upper and shifted lines, 
    # spaced 50 pixels apart

    # Find the midway points: half of the shifted lines addition
    patch_centers_x = []
    patch_centers_y = []
    for i in range(len(hulls_upper_y)):

        mid_x = hulls_upper_x[i]
        mid_y = hulls_upper_y[i]

        # Find the spaced out points on the middle arc
        center = 50
        while center < len(mid_y):
            patch_centers_x.append(mid_x[center])
            patch_centers_y.append(mid_y[center])
            if swap == 0:
                center = center + 50
            else:
                center = center + 50
    
    return patch_centers_x, patch_centers_y

def find_slopes(file, patch_centers_x, patch_centers_y, result_path):
    """
    Find the slopes of the lines at the patch centers.
    """
    
    img = cv.imread(result_path + '/blank_mask_outlines/' + file)

    slopes = []
    corners_x = []
    corners_y = []
    for i in range(len(patch_centers_x)):
        # Crop the image
        cropped, corner_x, corner_y = crop_box(img, patch_centers_x[i], patch_centers_y[i])
        cropped = np.array(cropped)
        corners_x.append(corner_x)
        corners_y.append(corner_y)

        # Binarize cropped image
        ret,binary = cv.threshold(cropped,127,255,cv.THRESH_BINARY)

        # Find the start and end points: max and min of the remaining line
        locs = np.nonzero(binary)
        
        # Start points
        max_x = np.max(np.unique(locs[1]))
        max_y = np.max(np.unique(locs[0]))

        # End points
        min_x = np.min(np.unique(locs[1]))
        min_y = np.min(np.unique(locs[0]))

        # Find the slope
        slope = ((max_y - min_y)/(max_x - min_x))

        # Check to see if the slope should be negative
        if locs[1][0] < locs[1][-1]:
            slope = -1*slope

        slopes.append(slope)

    return slopes

def rotate_crop(file, patch_centers_x, patch_centers_y, slopes, data_path, result_path, thickness):
    """
    Rotate the original image and the mask image based on the slopes of the lines at the patch centers, 
    crop the patches, and save them to files.
    """
    
    img1 = cv.imread(result_path + '/mask_slices/' + file)
    img2 = cv.imread(data_path+'/normal/' + file)
    img3 = cv.imread(result_path+'/mask_overlays_outlines/' + file)
    img3 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)

    img2 = cv.resize(img2, dsize=(2200, 820), interpolation=cv.INTER_CUBIC)
    
        
    my_dpi = 77
    fig, ax = plt.subplots(1, figsize=(img2.shape[1]/my_dpi, img2.shape[0]/my_dpi))
    ax.set_aspect('equal')
    ax.imshow(img3)
    ax.axis('off')

    reconstructs = []
    for i in range(len(patch_centers_x)):

        # Based on the slope of this patch, find the angle of rotation
        angle = findAngle(slopes[i], 0)

        if slopes[i] < 0:
            angle = -1*angle

        # Apply rotation to original image
        rotated2, rot_mat = rotate_image(img2, -1*angle, (int(patch_centers_x[i]), int(patch_centers_y[i])))

        # Binarize mask image
        ret, binary1 = cv.threshold(img1,127,255,cv.THRESH_BINARY)

        # Apply rotation to mask image
        rotated1, rot_mat = rotate_image(binary1, -1*angle, (int(patch_centers_x[i]), int(patch_centers_y[i])))
        
        # Apply rotation to center pixel coordinates
        center = [rot_mat[0,0]*patch_centers_x[i] + rot_mat[0,1]*patch_centers_y[i] + rot_mat[0,2], rot_mat[1,0]*patch_centers_x[i] + rot_mat[1,1]*patch_centers_y[i] + rot_mat[1,2]] 
        center = [round(center[0]), round(center[1])]
        
        top = center[1]-50
        left = center[0]-50
        bottom = center[1]+50
        right = center[0]+50
        
        # Account for cases less than 0: just crop from 0
        if top < 0:
            top = top + abs(top)
            bottom = bottom + abs(top)
        if left < 0:
            left = left + abs(left)
            right = right + abs(left)

        # Rotate, crop, and save mask image
        ret, rotated1 = cv.threshold(rotated1,0,255,cv.THRESH_BINARY)
        rotated1 = cv.cvtColor(rotated1, cv.COLOR_BGR2GRAY)
        
        # Crop this patch (mask image)
        patch = rotated1[top:bottom, left:right]
        reconstructs.append((top, left, bottom, right, angle))

        # Save binary patch to file (mask image)
        
        filename = result_path + '/patches/masks/' + file[0:-4] + '_patch_{num}'.format(num=i) + '.png'
        np.save(filename, patch)

        # Crop this patch (original image)
        patch = rotated2[top:bottom, left:right]
        # Save patch to file (original image)
        filename = result_path + '/patches/images/' + file[0:-4] + '_patch_{num}'.format(num=i) + '.png'
        np.save(filename, patch)

        if thickness == 0:
            patch = np.fliplr(patch)
            filename = result_path + '/patches/images/' + file[0:-4] + '_patch_{num}_flip'.format(num=i) + '.png'
            np.save(filename, patch)

        # Find the 4 rotated points
        # Inverse matrix of simple rotation is reversed rotation.
        M_inv = cv.getRotationMatrix2D(center,angle,1)

        # Points
        points = np.array([[left, bottom], [right,bottom], [right,top], [left,top], [left,bottom]])
    
        # Add ones
        ones = np.ones(shape=(len(points), 1))

        points_ones = np.hstack([points, ones])

        # Transform points
        transformed_points = M_inv.dot(points_ones.T).T

        ax.plot([transformed_points[0][0], transformed_points[1][0], transformed_points[2][0], transformed_points[3][0], transformed_points[0][0]], [transformed_points[0][1], transformed_points[1][1], transformed_points[2][1], transformed_points[3][1], transformed_points[0][1]], color='red', linestyle='solid')

    # Saving numpy file for this image
    np.save(result_path + '/reconstructed_npy/' + file + '.npy', reconstructs)

    # Save visualizing patches image to file
    plt.savefig(result_path + '/all_masks_overlays/' + file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return

def process_file(file, data_path, result_path, severities, file_names, epi_thicknesses):
    """
    Process a single file: generate masks, find slopes, rotate and crop patches.
    """
    try:

        # 1. Create masks of the top and bottom lines using Laplacian of Gaussian images
        upper_x, upper_y, shifted_y, im, thickness = LoG_masks(
            file, data_path, severities, file_names, epi_thicknesses)

        # 2. Find the middle points between the upper and shifted lines, for determining
        # where to crop the patches
        patch_centers_x, patch_centers_y = middle_points(upper_x, upper_y, 1)

        # 3. Draw the full-size masks and save them to the result path
        draw_masks(im, file, upper_x, upper_y, shifted_y, thickness, data_path, result_path)

        # 4. Find slopes for rotating the image to create patches
        slopes = find_slopes(file, patch_centers_x, patch_centers_y, thickness, result_path)

        # 5. Rotate and crop patches based on the slopes and save them to the result path
        rotate_crop(file, patch_centers_x, patch_centers_y, slopes, data_path, result_path, thickness)

        return "Completed"

    except Exception as e:
       
       return f"Error processing {file}: {e}"

def run_parallel(files_list, data_path, result_path, severities, file_names, epi_thicknesses, max_workers=12):
    """
    Run the process_file function in parallel using ProcessPoolExecutor.
    """
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for file in files_list:

            futures.append(
                executor.submit(process_file, file, data_path, result_path, severities, 
                                file_names, epi_thicknesses))

        results = []
        for f in tqdm(as_completed(futures), total=len(futures), desc="Creating masks..."):
            results.append(f.result())  
    return results

def create_masks(data_path, result_path, datasheet_path):
    """""
    Create masks: pipeline for loading resized images, diffedge images, normalizing iamges, and generating masks.
    """""

    df = pd.read_excel(datasheet_path)
    file_names = df['Current File Name'].tolist()

    severities = df['Severity'].tolist()
    epi_thicknesses = df['Epithelial Thickness'].tolist()
    random.seed(42)

    files_list = sorted(os.listdir(data_path+'/normal/'), reverse=True)
    results = run_parallel(files_list, data_path, result_path, severities, 
                           file_names, epi_thicknesses, max_workers=96)

    print("Finished mask creation.")
