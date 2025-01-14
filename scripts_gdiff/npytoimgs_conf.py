import numpy as np
from PIL import Image
import argparse
import os


def read_file_to_numpy(file):
    npzfile = np.load(file)
    n = len(npzfile)
    if n == 1:
        return list(npzfile['arr_0'])
    else:
        all_images = list(npzfile['arr_0'])
        all_labels = list(npzfile['arr_1'])
        all_confs = list(npzfile['arr_2'])
        return all_images, all_labels, all_confs
    pass


def load_numpy_labels_to_image(array1, array2, array3, folder_path):
    for i in range(len(array1)):
        im = Image.fromarray(array1[i], "RGB")
        img_name = "image"
        img_name += f"{array2[i]}"
        im_path = os.path.join(folder_path, f"class_{array2[i]}")
        os.makedirs(im_path, exist_ok=True)
        im.save(os.path.join(im_path, f"{img_name}_{i}_{array3[i]}.png"))
    pass


def load_numpy_labels_to_grayimage(array1, array2, array3, folder_path):
    for i in range(len(array1)):
        img = np.squeeze(array1[i], axis=-1)
        im = Image.fromarray(img, "L") # for RGB use different
        im.save(os.path.join(folder_path, f"%d_image%d_conf%.2f.png"%(int(array2[i]), i, array3[i])))


parser = argparse.ArgumentParser("numpy to images")
parser.add_argument("--numpy", help="numpy files", default="../outputhfai/runs/")
parser.add_argument("--rgb", action="store_true", help="process rbg")



if __name__ == '__main__':
    args = parser.parse_args()
    file_numpy = args.numpy
    folder_images = os.path.join(os.path.dirname(file_numpy), "images2")
    os.makedirs(folder_images, exist_ok=True)
    images_array, labels, confidences = read_file_to_numpy(file_numpy)
    if not args.rgb:
        load_numpy_labels_to_grayimage(images_array, labels, confidences, folder_images)
    else:
        load_numpy_labels_to_image(images_array, labels, confidences, folder_images)