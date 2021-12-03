import cv2
import os
import numpy as np
from PIL import Image


def go_deeper_folder(dir_list):
    new_dir = []
    for top_folder in dir_list:
        for folder in os.listdir(top_folder):
            if folder[0] != '.' and 'HOLD' not in folder and 'zip' not in folder:
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
        mask = np.zeros(image.shape[:2], dtype='uint8')
        mask = cv2.rectangle(
            mask, upper_left_bb[count], lower_right_bb[count], 255, -1)
        #mask = cv2.rectangle(
        #    mask, finger_coor[count], finger_coor[count], 127, 1)
        mask = cv2.circle(mask, finger_coor[count], 0, 127, -1)
        mask[mask < 127] = 0.0
        mask[mask > 127] = 255.0
        #np.savetxt('test1.txt', mask)
        #mask[finger_coor[count][1]][finger_coor[count][0]] = 127
        #mask = cv2.line(mask, finger_coor[count], finger_coor[count], 127, 1)
        #print(mask)

        cv2.imwrite(
            f"{save_location}/color/color_img{str(total_count).zfill(7)}.jpg", image)
        Image.fromarray(mask).save(f"{save_location}/mask/mask_img{str(total_count).zfill(7)}.png")
        #cv2.imwrite(
        #    f"{save_location}/mask/mask_img{str(total_count).zfill(7)}.jpg", mask)
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
    # Change this line to be where you stored the SCUT folder
    scut_location = '/Users/ethanruoff/Downloads'
    folders = [scut_location + '/SCUT/DATA/Fingertip_Calibration']
    for i in range(4):
        folders = go_deeper_folder(folders)
    folders.sort()

    # Change start count to resume from where you left off if you had to stop it early
    start_count = 0

    total_count_color = start_count
    total_count_depth = start_count
    total_count_sample = start_count
    count_printer = start_count
    
    for i in range(len(folders)):
        total_count_color = extract_frames_and_mask(
            folders[i] + '/data.txt', folders[i] + '/sample_color.avi', './training_data', total_count_color)
        total_count_depth = extract_frames(
            folders[i] + '/sample_depth8.avi', './training_data/depth8/depth8', total_count_depth)
        total_count_sample = extract_frames(
            folders[i] + '/sample_map.avi', './training_data/sample_map/sample_map', total_count_sample)
        if total_count_color > count_printer:
            count_printer += 1000
            print(f"Created {total_count_color} samples up to {i} folders")

    print(f"Created {total_count_color} samples")
main()