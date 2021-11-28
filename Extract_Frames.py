import cv2
import os
import numpy as np
import pandas as pd


def go_deeper_folder(dir_list):
    new_dir = []
    for top_folder in dir_list:
        for folder in os.listdir(top_folder):
            if folder[0] != '.' and 'useless' not in folder:
                new_dir.append(top_folder + '/' + folder)
    return new_dir


def extract_frames_and_mask(file_address, video_address, save_location, total_count):
    upper_left_bb = []
    lower_right_bb = []
    finger_coor = []
    with open(file_address, 'r') as f:
        for line in f:
            coor = line.split(' ')
            upper_left_bb.append((int(coor[1]), int(coor[2])))
            lower_right_bb.append((int(coor[3]), int(coor[4])))
            finger_coor.append((int(coor[5]), int(coor[6][0:-1])))

    vidcap = cv2.VideoCapture(video_address)
    success, image = vidcap.read()
    count = 0
    while success:
        # Create mask of the same size and draw bounding box and dot
        mask = np.zeros((480, 640))
        mask = cv2.merge([mask, mask, mask])
        mask = cv2.rectangle(
            mask, upper_left_bb[count], lower_right_bb[count], (255, 255, 255), -1)
        mask = cv2.circle(mask, finger_coor[count], 0, (0, 0, 255), -1)

        cv2.imwrite(
            f"{save_location}/color/color_img{str(total_count).zfill(7)}.jpg", image)
        cv2.imwrite(
            f"{save_location}/mask/mask_img{str(total_count).zfill(7)}.jpg", mask)
        success, image = vidcap.read()

        count += 1
        total_count += 1

    return total_count


def extract_frames(video_address, save_location, total_count):
    vidcap = cv2.VideoCapture(video_address)
    success, image = vidcap.read()
    while success:
        cv2.imwrite(
            f"{save_location}_img{str(total_count).zfill(7)}.jpg", image)
        success, image = vidcap.read()

        total_count += 1
    return total_count


def main():
    folders = ['./SCUT/DATA/Fingertip_Calibration']
    for i in range(4):
        folders = go_deeper_folder(folders)

    total_count_color = 0
    total_count_depth = 0
    total_count_sample = 0
    for folder in folders:
        total_count_color = extract_frames_and_mask(
            folder + '/data.txt', folder + '/sample_color.avi', './SCUT/training_data', total_count_color)
        total_count_depth = extract_frames(
            folder + '/sample_depth8.avi', './SCUT/training_data/depth8/depth8', total_count_depth)
        total_count_sample = extract_frames(
            folder + '/sample_map.avi', './SCUT/training_data/sample_map/sample_map', total_count_sample)
        if total_count_color % 500 == 0:
            print(f"Created {total_count_color} samples")
    print(f"Created {total_count_color} samples")