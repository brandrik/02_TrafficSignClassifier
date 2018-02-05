# PRECPROCESSING IMAGE DATA


# IMPORT
import numpy as np
import cv2

# 1 Grayscale
# 2 Normalize


def preprocess(data):
    #return (data - 128) / 128
    ret_arr = np.ndarray(shape = (data.shape[0:3] + tuple([1])))
    a = -0.5
    b = 0.5
    minimum = 0
    i = 0
    maximum = 255
    for img in data:
        # grayscale
        # change color space: Y Cr and Cb images, where Y essentially is a grayscale picture, so take Y image
        YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)  
        img_g = np.resize(YCrCb[:,:,0], (32,32,1)) # only take Y image
        
        # normalize
        img_gn = a + ((img_g - minimum) * (b - a)) / (maximum - minimum)
        ret_arr[i, :, :, :] = img_gn
        i+=1
    return ret_arr




def random_scale(img):   
    rows,cols,_ = img.shape

    # transform limits
    px = np.random.randint(-4,4)

    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])

    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    output = cv2.warpPerspective(img,M,(rows,cols))
    
    output = output[:,:,np.newaxis]
    
    return output


def random_warp(img):
    
    rows,cols,_ = img.shape

    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    warp_coeff = 0.1# this coefficient determines the degree of warping
    rndx *= cols * warp_coeff   
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * warp_coeff

    # 3 starting points for transform, 1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4

    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
                       [y2+rndy[1],x1+rndx[1]],
                       [y1+rndy[2],x2+rndx[2]]])

    M = cv2.getAffineTransform(pts1,pts2)

    output = cv2.warpAffine(img,M,(cols,rows))
    
    output = output[:,:,np.newaxis]
    
    return output

def random_translate(img):
    rows,cols,_ = img.shape
    
    # allow translation up to px pixels in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)

    M = np.float32([[1,0,dx],[0,1,dy]])
    output = cv2.warpAffine(img,M,(cols,rows))
    
    output = output[:,:,np.newaxis]
    
    return output

def random_brightness(img):
    shifted = img + 1.0   # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = min(0.1, max_coef - 1)
    coef = np.random.uniform(min_coef, max_coef)
    output = (shifted * coef) - 1.0 # shift back to (-1,1) range
    return output




