import os


def get_list(img_root, img_folder):
    img_path = os.path.join(img_root, img_folder)
    img_list = os.listdir(img_path)

    with open(img_folder+'.txt', 'wt') as f:
        for i in img_list:
            print(os.path.join(img_path, i), file=f)


if __name__ == '__main__':
    img_root = '/data1/TL/data/shoe_dataset'
    img_folder = ['source', 'target', 'source_label']

    for i in range(len(img_folder)):
        get_list(img_root, img_folder[i])
