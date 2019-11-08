import numpy as np
import imageio
import cv2
import sys

# ORB works poorly with lots of dark pixels,
# probably on account of uint8 quantization to zero.
# This is a very simple brightener - you don't want to 
# use this for the final result
def brighten(img,threshold=0.1): 
    if img.mean() > threshold:
        return img
    else:
        return brighten(2*img - img**2,threshold) # Closes over [0,1]
        
# compute a homography (3x3 transform matrix for 2D homogenous coordinates)
# that projects image 1 onto image 2. We use this to line up multiple images
# of the sky that were taken at different times, with the camera at a slightly
# different angle, etc.
def match(im1,im2):
    # OpenCV ORB requires uint8 for some reason
    im1 = (brighten(im1)*255).astype('uint8') 
    im2 = (brighten(im2)*255).astype('uint8')
    # Identify interesting points in the image (i.e. stars)
    det = cv2.ORB_create(nfeatures=50000)
    kp1,desc1 = det.detectAndCompute(im1,None)
    kp2,desc2 = det.detectAndCompute(im2,None)
    # Matches up interesting points in both images, based on their descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
    matches = bf.match(desc1,desc2)
    # Pick the top 10% of matches (by hamming distance of their descriptor vectors)
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:len(matches)//10]
    if len(matches) < 10:
        raise Exception("<10 matching descriptors, poor match")
    # Get the coordinates of the matching stars in each image
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    # Calculate a homography matrix from our set of probably-matching stars.
    # The RANSAC algorithm will try to discard inconsistent outliers.
    # Mask tells us which matches seem internally consistent.
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    if mask.sum() < 10:
       raise Exception("<10 RANSAC inlier descriptors, poor match")
    return M

# Using `im` as the canvas, project every image in `loaders` onto `im` (after lining them up).
# Returns the summed up image, as well as an array telling us what fraction of images contributed
# to each pixel. We probably only want to use pixels that came from every image
# We use `loaders` instead of passing in images directly because we want to 
# load huge TIFF images into memory one at a time rather than all at once.
def add(im,loaders):
    h,w,*_ = im.shape
    out = im.copy()
    count = np.full((h,w),1.0)
    for load_im2 in loaders:
        im2 = load_im2()
        M = match(im2,im)
        out += cv2.warpPerspective(im2,M,(w,h))
        counter = np.full(im2.shape[0:2],1.0)
        count += cv2.warpPerspective(counter,M,(w,h))
    return (out/out.max(),count/count.max())
 
# Normalize various formats to [0,1)       
maxima = { np.uint8 : 255.0
         , np.dtype('u1') : 255.0
         , np.uint16 : 65535.0
         , np.dtype('u2') : 65535.0
         , np.float : 1.0
         , np.double : 1.0
         }

# We process images mostly as 64-bit double-precision floating 
# point arrays, since they have plenty of precision.
def load_img(path):
    print("Loading %s" % path)
    image = imageio.imread(path)
    maximum = maxima[image.dtype]
    image = image.astype('double')/maximum
    return image

if __name__ == '__main__':
    images = sys.argv[1:]
    main_img= images[0]
    main_img = load_img(main_img)
    other_imgs = images[1:]

    combined,count = add(main_img,((lambda: load_img(path)) for path in other_imgs))
    # Any pixel that didn't come from every single image gets marked as white so we can crop it out
    valid_region = count > 0.99
    (_,_,*chs) = main_img.shape
    if chs == []:
        combined *= valid_region
        combined += -(valid_region - 1)
    else:
        for ch in range(chs[0]):
            combined[:,:,ch] *= valid_region
            combined[:,:,ch] += -(valid_region - 1)
    print("saving to combined.tiff")
    imageio.imsave('combined.tiff',(combined*65535).astype('uint16'))
    imageio.imsave('mask.png',(np.stack([count,count,count],axis=-1)*255).astype('uint8'))
