import os.path

import numpy as np

from guided_diffusion.image_datasets import load_dataset_MNIST_nosampler, load_dataset_CelebA_nosampler
from PIL import Image
import torch

temp_data = load_dataset_CelebA_nosampler(train=True, batch_size=1024, class_cond=True, image_size=64)
nb_images = 50000
save_folder = os.path.join("../output/data", "CelebA64")

raw_images_folder = os.path.join(save_folder, "images")
os.makedirs(raw_images_folder, exist_ok=True)
count = 0
all_images = []
while True:
    batch, cond = next(temp_data)
    sample = ((batch + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    batch = sample.numpy()
    all_images.append(batch)
    cond = cond['y']
    n = batch.shape[0]
    for i in range(n):
        if count >= nb_images:
            print("Complete generating images")
            all_images = np.concatenate(all_images, axis=0)
            all_images = all_images[:nb_images]
            shape_str = "x".join([str(x) for x in all_images.shape])
            np.savez(os.path.join(save_folder, f"real_{shape_str}.npz"), all_images)
            exit(0)
        # img = np.squeeze(batch[i])
        img = Image.fromarray(batch[i], "RGB")
        img_path = os.path.join(raw_images_folder, f"nocond_image{count}.png")
        img.save(img_path)
        count += 1
