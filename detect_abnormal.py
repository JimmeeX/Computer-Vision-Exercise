import sys
import os

import cv2
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from skimage.color import rgb2lab, deltaE_cie76

# HYPERPARAMETERS - These Values should be tuned for optimisation
image_thresholds = { # Colour Difference Threshold for image type [Left Field, Right Field]
    "ndre": [15, 15],
    "ndvi": [15, 15],
    "ccci": [37.5, 40],
    "msavi": [40, 40]
}

K = 3 # No. Neighbours for KNN Dominant Colour Finder

# Parameters for Morphology 
closing_size = (3,3)
opening_size = (3,3)
gradient_size = (5,5)
closing_iter = 7  # More iterations to group more
opening_iter = 5  # More iterations to remove more noise
gradient_iter = 1

def show_colour(col):
    """
    Plot 100x100x3 image of colour
    
    Inputs
    col: 1x3 rgb values as list
    """
    x = np.ones((100, 100, 3), dtype="uint8")
    x[:, :, :] = col
    fig = plt.figure(figsize = (1,1))
    plt.axis("off")
    plt.imshow(x)
    plt.show()

def convertToRGB(im_array):
    """
    Convert MxN to MxNx3 using normalisation + colour mapping
    
    Inputs
    im_array: MxN image as np.array
    
    Return
    MxNx3 rgb image as np.array 
    """
    minima = np.min(im_array)
    maxima = np.max(im_array)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm)
    rgb = mapper.to_rgba(im_array)[:,:,:3]
    
    return np.multiply(rgb, 255).astype("uint8")

def get_mean_colour(img):
    """
    Retrives the mean colour of the image
    
    Inputs
    img: MxNx3 image as np.array
    
    Return
    1x3 rgb values as list
    """
    # Ignore bg
    bg = [0, 0, 0]
    
    # Flatten
    img_flat = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    count = 0
    redB = 0
    greenB = 0
    blueB = 0
    for pixel in img_flat:
        if not np.array_equal(pixel, bg):
            redB += pixel[0]
            greenB += pixel[1]
            blueB += pixel[2]
            count += 1
    return [redB//count, greenB//count, blueB//count]

def get_dominant_colours(img, K):
    """
    Get K dominant colours in img using KMeans clustering (NOT USED)
    
    Input
    img: MxNx3 image as np.array
    K: Number of clusters
    
    Return
    res2: MxNx3 image which consists only of the K dominant colours
    center: List of top K dominant colours (Kx3) 
    """
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2, center

def get_similar(img, colour, threshold):
    """
    Find colour similarity of every pixel in img to specified "colour". Non similar colours which are darker are shown as RED. Non similar colours which are lighter are shown as blue
    
    Input
    img: MxNx3 image as np.array
    colour: Specified colour for comparison
    threshold: Threshold between similar and non similar colours
    
    Return
    img: MxNx3 image with non-similar colours as RED (darker) and BLUE (lighter) as np.array
    red_img: MxNx3 image with ONLY RED marks
    blue_img: MxNx3 image with ONLY BLUE marks
    """

    lab = rgb2lab(img)
    
    colour_3d = np.uint8(np.asarray([[colour]]))
    red_3d = np.uint8(np.asarray([[255, 0, 0]]))
    blue_3d = np.uint8(np.asarray([[0, 0, 255]]))
    
    dE = deltaE_cie76(rgb2lab(colour_3d), lab)
    
    img_flat = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    notBlack = np.array([not np.array_equal(item, [0,0,0]) for item in img_flat]).reshape(img.shape[:2])
    img_light = (np.sum(img_flat, axis=1) >= sum(colour)).reshape(img.shape[:2])
    img_dark = (np.sum(img_flat, axis=1) < sum(colour)).reshape(img.shape[:2])
    
    red_segment = np.logical_and(np.logical_and(dE >= threshold, notBlack), img_dark)
    blue_segment = np.logical_and(np.logical_and(dE >= threshold, notBlack), img_light)
    img[red_segment] = red_3d
    img[blue_segment] = blue_3d
    
    red_img = np.zeros(img.shape, dtype="uint8")
    red_img[red_segment] = red_3d
    blue_img = np.zeros(img.shape, dtype="uint8")
    blue_img[blue_segment] = blue_3d
    
    return img, red_img, blue_img

def get_similar_bg(img, bg_colour, threshold):
    """
    Find colour similarity of every pixel in img to specified "colour". Similar colours are shown as BLACK (ie, black background)
    
    Input
    img: MxNx3 image as np.array
    colour: Specified colour for comparison
    threshold: Threshold between similar and non similar colours
    
    Return
    img: MxNx3 image with similar colours as BLACK as np.array
    """
    lab = rgb2lab(img)
    
    colour_3d = np.uint8(np.asarray([[bg_colour]]))
    black_3d = np.uint8(np.asarray([[0, 0, 0]]))
    
    dE = deltaE_cie76(rgb2lab(colour_3d), lab)
    
    img_flat = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    notBlack = np.array([not np.array_equal(item, [0,0,0]) for item in img_flat]).reshape(img.shape[:2])
    
    img[dE < threshold] = black_3d
    
    return img

def group_segments(img, close_kernel_size, open_kernel_size, grad_kernel_size, close_iter, open_iter, grad_iter):
    """
    Morphological Analysis. Groups concentrated points, and removes the non-concentrated points (noise)
    
    Input
    img: MxNx3 image as np.array
    close/open/grad_kernel_size: The window size for the morphological analysis for closing/opening/gradient transformations
    close/open/grad_iter: The number of iterations for the morphological analysis for closing/opening/gradient transformations
    
    Return
    img: MxNx3 image after the transformations
    """
    # Create kernels
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
    gradient_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, grad_kernel_size)
    
    # Apply closing
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, closing_kernel, iterations=close_iter)
    
    # Apply opening
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, opening_kernel, iterations=open_iter)
    
    # Apply gradient
    gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, gradient_kernel, iterations=grad_iter)
    
    return gradient

def overlay(img_1, img_2):
    """
    Places non-black parts of img_1 over img_2
    
    Input
    img_1: MxNx3 image (on top) as np.array
    img_2: MxNx3 image (in the background) as np.array
    
    Return
    MxNx3 combined image
    """
    
    img_1_flat = img_1.reshape(img_1.shape[0]*img_1.shape[1], img_1.shape[2])
    img_2_flat = img_2.reshape(img_2.shape[0]*img_2.shape[1], img_2.shape[2])
    
    for i, pixel in enumerate(img_1_flat):
        if not np.array_equal(pixel, [0, 0, 0]):
            img_2_flat[i, :] = pixel[:]
    return img_2_flat.reshape(img_1.shape)

def rgb_to_rgba(img):
    """
    Convert rgb image to rgba PIL image
    
    Input
    img: MxNx3 image as np.array
    
    Return
    im: MxNx4 image as PIL object
    """
    im = Image.fromarray(img).convert("RGBA")
    datas = im.getdata()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    im.putdata(newData)
    return im

def export_result(outfile, name, result):
    """Save results as png to outfile"""
    im = rgb_to_rgba(result)
    im.save(outfile + name[:-3] + "png", "PNG")

def detect_irregular(inPath, outPath):
    """The main code sequence. See the jupyter notebook file if confused."""
    im_list_tif = [item for item in os.listdir(inPath) if item[-3:] == "tif"]

    for item in im_list_tif:
        print("Processing " + item)
        # Load Image as array
        im = Image.open(inPath + item)
        im_array = np.array(im)
        
        # Convert to RGB array
        img = convertToRGB(im_array)
        
        # Convert Purple Background to Black
        img = get_similar_bg(img, [68, 1, 84], 5)
        
        # Segment both fields into separate pictures
        img_1 = np.zeros(img.shape, dtype="uint8")
        img_2 = np.zeros(img.shape, dtype="uint8")
        img_1[:, :273, :] = img[:, :273, :]
        img_2[:, 273:, :] = img[:, 273:, :]
        
        # Get mean colour for each field
        mean_colour_1 = get_mean_colour(img_1)
        mean_colour_2 = get_mean_colour(img_2)
        
        # Perform similarity test
        threshold_keys = list(image_thresholds.keys())
        label = threshold_keys[[key in item for key in threshold_keys].index(True)]
        new_img_1, red_1, blue_1 = get_similar(np.copy(img_1), mean_colour_1, image_thresholds[label][0])
        new_img_2, red_2, blue_2 = get_similar(np.copy(img_2), mean_colour_2, image_thresholds[label][1])
        
        # Apply erosion and dilution
        grad_red_1 = group_segments(red_1, closing_size, opening_size, gradient_size, closing_iter, opening_iter, gradient_iter)
        grad_red_2 = group_segments(red_2, closing_size, opening_size, gradient_size, closing_iter, opening_iter, gradient_iter)
        grad_blue_1 = group_segments(blue_1, closing_size, opening_size, gradient_size, closing_iter, opening_iter, gradient_iter)
        grad_blue_2 = group_segments(blue_2, closing_size, opening_size, gradient_size, closing_iter, opening_iter, gradient_iter)
        
        # Add irregular images together
        irregularities_red = cv2.add(grad_red_1, grad_red_2)
        irregularities_blue = cv2.add(grad_blue_1, grad_blue_2)
        irregularities = cv2.add(irregularities_red, irregularities_blue)    
        
        # Overlay irregular image on top of original image
        result = overlay(irregularities, np.copy(img))
        
        # Save result to specified outPath
        export_result(outPath, item, result)

if __name__ == "__main__":
    if len(sys.argv[1:]) != 2:
        print("Error - Please enter inPath and outPath (make sure these directories exist)")
        print("python detect_abnormal.py <inPath> <outPath>")
        print("python detect_abnormal.py Data/ Results/")
    else:
        inPath = sys.argv[1]
        outPath = sys.argv[2]
        detect_irregular(inPath, outPath)