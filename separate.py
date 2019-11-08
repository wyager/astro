import numpy as np
import imageio
import cv2
import sys

import combine


def overlay(reference,secondary):
    h,w,*_ = reference.shape
    homography = combine.match(secondary,reference)
    return cv2.warpPerspective(secondary, homography, (w,h))


if __name__ == '__main__':
    images = sys.argv[1:]
    main_img= images[0]
    main_img = combine.load_img(main_img)
    other_imgs = images[1:]

    for img_path in other_imgs:
        secondary_img = combine.load_img(img_path)
        overlayed = overlay(main_img,secondary_img)
        out_path = img_path+".overlay.tiff"
        print("saving to " + out_path)
        imageio.imsave(out_path,(overlayed*65535).astype('uint16'))