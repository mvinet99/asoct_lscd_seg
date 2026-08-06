# Postprocess files from output predictions from the model, and perform downstream analysis
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math
import os
import concurrent.futures
from tqdm import tqdm
from scipy import ndimage
from math import floor

def rotate_image(image, angle, image_center):
    """
    Rotate an image by a given angle around a specified center point.
    """
  
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)

    return result

def get_line(x1, y1, x2, y2):
    """
    Get all points on a line between two coordinates.
    """

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

    for x in range(int(x1), int(x2 + 1)):
        if issteep:
            points.append((int(floor(y)), int(floor(x))))
            points.append((int(round(y)), int(round(x))))
        else:
            points.append((int(floor(x)), int(floor(y))))
            points.append((int(round(x)), int(round(y))))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points

def process_file(file, data_path, result_path):
    """
    Process a single file: reconstruct the predicted mask, 
    overlay with original image, and calculate thickness.
    """
    
    load_path = result_path + '/predicted_masks/'

    # Reconstruct the predicted mask
    arr = np.load(result_path+'/reconstructed_npy/'+file)

    # Reconstruct the predicted mask from patches
    transforms = []
    for i in tqdm(range(arr.shape[0])):
        
        img = np.zeros([820, 2200])
        nums = arr[i]
        filename = os.path.join(load_path, f"{file.split('.')[0]}_patch_{i}.png_pred.npy")

        pch = np.load(filename)

        if pch is None:
            pch = np.zeros([100,100])
        #print(np.unique(pch))
        pch = cv.resize(pch, (100, 100), interpolation=cv.INTER_CUBIC)
        pch[pch != 0] = 255
        img = rotate_image(img, -nums[4], (int(nums[1]+50), int(nums[0]+50)))
        try:
            if int(nums[2])-int(nums[0]) == 100 and int(nums[3])-int(nums[1]) == 100:
                img[int(nums[0]):int(nums[2]), int(nums[1]):int(nums[3])] = pch
        except:
            pass
        img = rotate_image(img, nums[4], (int(nums[1]+50), int(nums[0]+50)))
        transforms.append(img)

    final = sum(transforms)
    final[np.isnan(final)] = 0 

    # Remove small objects
    Zlabeled, Nlabels = ndimage.measurements.label(final)
    for label in range(Nlabels + 1):
        if (Zlabeled == label).sum() < 1000:
            final[Zlabeled == label] = 0
    final = final.astype(np.uint8)

    cv.imwrite(os.path.join(result_path, 'reconstructed_masks', file + '.png'), final)

    # Overlay the predicted masks with the original images

    # Get edges of the reconstructed mask
    edges = cv.Canny(final, 25, 150, L2gradient=True)

    # Load original image
    filepath = os.path.join(data_path, "resized", file[0:-8] + '.png')
    img = cv.imread(filepath)
    img = cv.resize(img, dsize=(2200, 820), interpolation=cv.INTER_CUBIC)

    ys = np.where(edges == 255)[0]
    xs = np.where(edges == 255)[1]

    top_compare = []
    bot_compare = []

    for uni in np.unique(xs):
        pos = np.where(uni == xs)
        nums = ys[pos]
        val = int(np.min(nums))
        val2 = int(np.max(nums))
        top_compare.append((uni, val))
        bot_compare.append((uni, val2 - 4))

    # Load patch transform
    arr = np.load(result_path+'/reconstructed_npy/'+file)

    # Draw edges with per-patch colors
    for uni, y in top_compare:
        color = [0, 255, 0]  # Default color (green) for top edges
        thickness = 2 
        half_t = thickness // 2

        # Draw a thicker vertical line
        img[max(y - half_t, 0): y + half_t + 1, uni, :] = np.array(color, dtype=np.uint8)

    for uni, y in bot_compare:
        color = [0, 255, 0]  # Default color (green) for bottom edges
        thickness = 2 
        half_t = thickness // 2

        # Draw a thicker vertical line
        img[max(y - half_t, 0): y + half_t + 1, uni, :] = np.array(color, dtype=np.uint8)

    # Save result
    save_path = result_path + "/reconstructed_predictions/" + file[0:-4]

    cv.imwrite(save_path, img)

    # Calculate the epithelial thickness from the predicted masks

    points_x = []
    points_y = []
    points2_x = []
    points2_y = []
    points2 = []
    top_compare = []
    bot_compare = []
    for uni in np.unique(np.where(edges==255)[1]):
        pos = np.where(uni==np.where(edges==255)[1])
        nums = []
        for po in pos:
            nums.append(ys[po])
        val = min(nums[0])
        val2 = max(nums[0])
        points_x.append(uni)
        points_y.append(val+1)
        points2_x.append(uni)
        points2_y.append(val2)
        points2.append((uni,val2))
        top_compare.append((uni, val))
        bot_compare.append((uni, val2))
    
    x_temp = np.array(points_x)
    positions = []
    start = 0
    for i in range(len(points_x)):
            try:
                p = np.where(x_temp==start)[0][0]
                positions.append(p)
            except:
                p = None
                positions.append(None)
        
            start = start + 20 # Pixel distance (every 20)

    if points_x != []:
        # Fit the polynomial on the curve
        fit = np.polyfit(points_x, points_y, 6)

        # Find the equation for the derivative curve
        deriv = np.polyder(fit)

        x_temp = np.array(points_x)
        bot_int = []
        minXs = []
        maxXs = []
        ylows  = []
        yhighs = []
        c_s = []
        d_s = []
        for p in positions:
            if p is not None:
                # Goal: Find the tangent line at this point
                pointVal = points_x[p]

                # Find the y-value of the derivative at point pointVal
                y_val_point = np.polyval(fit, pointVal)

                # Find the slope of the derivative at point pointVal
                slope_at_point = np.polyval(deriv, pointVal)

                # For plotting - define the min and max x_values
                minX = pointVal-50
                maxX = pointVal+50
                minXs.append(minX)
                maxXs.append(maxX)
                
                # For plotting - define the min and max y_values
                ylow = (minX - pointVal) * slope_at_point + y_val_point
                yhigh = (maxX - pointVal) * slope_at_point + y_val_point
                ylows.append(ylow)
                yhighs.append(yhigh)

                # Find the perpendicular line points
                dy = math.sqrt(3**2/(slope_at_point**2+1))
                dx = -slope_at_point*dy
                # Top point x, top point y
                c = [(minX + (.5*(maxX-minX))) + (16*dx), (ylow + (.5*(yhigh-ylow))) + (16*dy)]
                c_s.append(c)
                # Bottom point x, bottom point y
                d = [(minX + (.5*(maxX-minX))) - (16*dx), (ylow + (.5*(yhigh-ylow))) - (16*dy)]
                d_s.append(d)

                # Find all points for the CD line
                out_points = get_line(c[0], c[1], d[0], d[1])
                
                # Find the intersection point of the CD line and the bottom curve - where the x- and y- coordinates match
                intsec = list(set(points2).intersection(out_points))

                if intsec == []:
                    bot_int.append(None)
                else:
                    bot_int.append(intsec[0])
                                    
        # Append the top line points properly
        up_int = []
        for position in positions:
            if position is not None:
                up_int.append((int(points_x[position]), int(points_y[position])))

        up_int_x = []
        up_int_y = []
        for i in range(len(up_int)):
            if bot_int[i] is not None:
                x, y = up_int[i]
                up_int_x.append(x)
                up_int_y.append(y)

        bot_int_x = []
        bot_int_y = []
        for i in range(len(bot_int)):
            if bot_int[i] is not None:
                x, y = bot_int[i]
                bot_int_x.append(x)
                bot_int_y.append(y-6)

        # Calculate Euclidean distance
        eucs = []
        for i in range(len(bot_int_y)):
            if bot_int_y[i] is not None:

                euc = math.dist((up_int_x[i], up_int_y[i]), (bot_int_x[i], bot_int_y[i]))
                euc = (250/72)*(0.8)*(euc)        
                eucs.append(euc)
        
        # If no predictions at all, treat epithelial thickness as 0
        if len(eucs) == 0:
            eucs = [0]
        
        filename = result_path + '/thickness_npy/' + file[0:-4] + '.npy'
        np.save(filename, eucs)

    else:
        # If no predictions, just save the predicted value as 0
        filename = result_path + '/thickness_npy/' + file[0:-4] + '.npy'
        eucs = [0]
        np.save(filename, eucs)

        plt.close('all')

def postprocess(data_path, result_path):
    """
    Postprocess the predicted masks and perform downstream analysis.
    """

    print("Postprocessing images...")

    files = os.listdir(result_path+'/reconstructed_npy/')

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_file, file, data_path, result_path): file for file in files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()
