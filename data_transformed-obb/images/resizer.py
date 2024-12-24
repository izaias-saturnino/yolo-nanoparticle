# script that creates alternate versions of images in a folder
# each image will have 2 alternate versions: one with redimensioned width and one with redimensioned height

import os
import cv2

folders = ['train', 'test', 'val']
output_folder = 'resized'

factor = 2

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for folder in folders:
    images = os.listdir(folder)

    if not os.path.exists(os.path.join(output_folder, folder)):
        os.makedirs(os.path.join(output_folder, folder))

    for image in images:
        img = cv2.imread(os.path.join(folder, image))
        
        filename, file_extension = os.path.splitext(image)

        # resize the image
        img_resized_width = cv2.resize(img, (int(img.shape[1]/factor), img.shape[0]))
        img_resized_height = cv2.resize(img, (img.shape[1], int(img.shape[0]/factor)))

        # save the resized images
        cv2.imwrite(os.path.join(output_folder, folder, filename+"_w"+file_extension), img_resized_width)
        cv2.imwrite(os.path.join(output_folder, folder, filename+"_h"+file_extension), img_resized_height)