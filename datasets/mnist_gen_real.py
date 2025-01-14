import os.path

import numpy as np

from guided_diffusion.image_datasets import load_dataset_MNIST_nosampler
from PIL import Image
import torch

temp_data = load_dataset_MNIST_nosampler(train=True, batch_size=1024, class_cond=True)
nb_images = 10000
save_folder = os.path.join("data", "MNIST")
os.makedirs(save_folder, exist_ok=True)
count = 0
while True:
    batch, cond = next(temp_data)
    sample = ((batch + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    batch = sample.numpy()

    cond = cond['y']
    n = batch.shape[0]
    for i in range(n):
        if count >= nb_images:
            print("Complete generating images")
            exit(0)
        img = np.squeeze(batch[i])
        img = Image.fromarray(img, "L")
        img_path = os.path.join(save_folder, f"{cond[i]}_image{count}.png")
        img.save(img_path)
        count += 1
