     #  Rotate, scale, and translate and crop to 128 sizes...
     #   Rotate is angle of rotation degrees
     #   Scale is the scale of the image in pixels between [w*cos(45 degrees),w]
     #   Translation is the X and Y translation of the image in pixels of the original image
import cv2
import numpy as np
from random import randrange, randint

def augmentImage(im, rot_angle = -1, scale = -1, tran = -1):
    h, w, c = im.shape
    min_size = 128; # smallest size in the original image for the output image
     # Rotate, Scale and Translate in the same affine rotation.
     #     So translate and crop between buffer of dim and the random scaling, rotate, and random scaling (all at once!)

    if rot_angle < 0:
        rot_angle = randrange(360)


     #print(rot_angle)
    mH, mW = calculateLargestProportionalRect(rot_angle, h,w)
    dim = np.floor(min(mH,mW))
    if scale < 0:
        scale = randrange(min_size,dim)
    elif scale == 0:
        scale = 181 # Max allowed if allowing for any rotation

     # Get rotation and scale matrix
    scale_ratio = 128.0/scale
    M = cv2.getRotationMatrix2D((h/2,w/2),rot_angle,scale_ratio)#randrange(360),1)

      # Find max difference and scale it down and make random

    if tran < 0:
        xshift = scale_ratio*(randint(-(dim-scale),dim-scale)/2.0)
        yshift = scale_ratio*(randint(-(dim-scale),dim-scale)/2.0)
    else:
        xshift = scale_ratio*tran
        yshift = scale_ratio*tran
     #(random.randrange(1,3)*2-3)#
     #xshift = (128.0/rand_scale)*((dim-rand_scale)/2.0)
     #yshift = (128.0/rand_scale)*((dim-rand_scale)/2.0)
    M[0,2] = M[0,2] - (h/2 - 64) - xshift # shift center and add random translation in x
    M[1,2] = M[1,2] - (w/2 - 64) - yshift # shift center and add random translation in y

     #print('Dim: {}\tScale: {}\tRot: {:03d}\txshift: {}\tyshift: {}'.format(dim,scale,rot_angle,xshift,yshift))

     # Apply Transformation in only valid area and use inter_area since downsampling (CV suggestion)
    return cv2.warpAffine(im,M,(128,128),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

def calculateLargestProportionalRect(angle, origHeight,origWidth):
    angle = np.deg2rad(angle)
    if (origWidth <= origHeight):
        w0 = origWidth
        h0 = origHeight

    else:
        w0 = origHeight
        h0 = origWidth

     # Angle normalization in range [-PI..PI)
    ang = angle - np.floor((angle + np.pi) / (2*np.pi)) * 2*np.pi;
    ang = np.abs(ang);
    if (ang > np.pi / 2):
        ang = np.pi - ang;
    c = w0 / (h0 * np.sin(ang) + w0 * np.cos(ang));
    if (origWidth <= origHeight) :
        w = w0 * c
        h = h0 * c

    else:
        w = h0 * c
        h = w0 * c

    return h,w