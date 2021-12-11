import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def go_deeper_folder(dir_list):
    new_dir = []
    for top_folder in dir_list:
        for folder in os.listdir(top_folder):
            if folder[0] != '.' and 'HOLD' not in folder and 'zip' not in folder:
                new_dir.append(top_folder + '/' + folder)
    return new_dir

def fix_masks(file_name):
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[img > 150] = 255
    img[img <= 150] = 0
    img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_AREA)
    cv2.imwrite(f"training_data/IPN_hand/mask/{file_name.rsplit('/')[-1][:-4] + '.png'}", img)

def main():
    folder = ['/Users/ethanruoff/Downloads/segment']
    folder = go_deeper_folder(folder)
    folder = go_deeper_folder(folder)
    folder.sort(key = lambda x: (x.split('_')[-4], x.split('_')[-2], x[-10:]))
    i = 0
    while i < 40000:
        fix_masks(folder[i])
        print(i)
        i += 1
    

main()