import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
import sys

images = sys.argv[1:]

main_img= images[0]
other_imgs = images[1:]

def match(im1,im2):
    im1 = (im1*256).astype('uint8')
    im2 = (im2*256).astype('uint8')
    det = cv2.ORB_create(nfeatures=10000)
    kp1,desc1 = det.detectAndCompute(im1,None)
    kp2,desc2 = det.detectAndCompute(im2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1,desc2)
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:len(matches)//10]
    assert(len(matches)>10)
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    # Mask tells us which matches seem internally consistent
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    return M
#     matches = cv2.drawMatches(im1,kp1,im2,kp2,[matches[i] for i in range(len(matches)) if mask[i]],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     imageio.imsave('matches.png',matches)

def add(im,ims):
    (h,w,_) = im.shape
    out = im.copy()
    count = np.full((h,w),1.0)
    for load_im2 in ims:
        im2 = load_im2(None)
        M = match(im2,im)
        out += cv2.warpPerspective(im2,M,(w,h))
        counter = np.full(im2.shape[0:2],1.0)
        count += cv2.warpPerspective(counter,M,(w,h))
    return (out/out.max(),count/count.max())
        
maxima = { np.uint8 : 255.0
         , np.dtype('u1') : 255.0
         , np.uint16 : 65535.0
         , np.dtype('u2') : 65535.0
         , np.float : 1.0
         , np.double : 1.0
         }

def load_img(path):
    image = imageio.imread(path)
    maximum = maxima[image.dtype]
    image = image.astype('double')/maximum
    return image

combined,count = add(load_img(main_img),[(lambda _ : load_img(path)) for path in other_imgs])
valid_region = count > 0.99
for ch in [0,1,2]:
    combined[:,:,ch] *= valid_region
    combined[:,:,ch] += -(valid_region - 1)
plt.imsave('combined.tiff',combined)
