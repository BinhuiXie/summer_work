from PIL import Image
import numpy as np
import os
import os.path as osp


def get_label_map(img_root, img_folder, label_folder):
    img_path = osp.join(img_root, img_folder)
    img_list = os.listdir(img_path)
    for i in img_list:
        img = Image.open(osp.join(img_path, i))
        try:
            r, g, b, a = img.split()
            arr = np.array(a)
            arr[arr!=0] = 1
            a = Image.fromarray(arr, 'P')
            a.save(img_root + '/' + label_folder + '/' + i)
        except:
            print(img_path + i)


if __name__ == '__main__':
    img_root = '/data1/TL/data/shoe_dataset'
    img_folder = ['source']
    label_folder = ['source_label_map']

    for i in range(len(img_folder)):
        if not osp.exists(osp.join(img_root, label_folder[i])):
            os.makedirs(osp.join(img_root, label_folder[i]))
        get_label_map(img_root, img_folder[i], label_folder[i])
