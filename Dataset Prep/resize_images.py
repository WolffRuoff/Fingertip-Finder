import cv2
import os

def go_deeper_folder(dir_list):
    new_dir = []
    for top_folder in dir_list:
        for folder in os.listdir(top_folder):
            if folder[0] != '.' and 'xml' not in folder and 'zip' not in folder:
                new_dir.append(top_folder + '/' + folder)
    return new_dir

def main():
    folder = ['/Users/ethanruoff/Downloads/all_frames']
    folder = go_deeper_folder(folder)
    folder = go_deeper_folder(folder)
    folder = go_deeper_folder(folder)
    folder.sort(key = lambda x: (x.split('_')[-4], x.split('_')[-2], x[-10:]))

    i = 0
    while i < 40000:
        img = cv2.imread(folder[i])
        img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_AREA)
        cv2.imwrite(f"training_data/IPN_hand/color/{folder[i].rsplit('/')[-1]}", img)
        print(i)
        i += 1

main()