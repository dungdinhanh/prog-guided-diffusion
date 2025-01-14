from guided_diffusion.image_datasets import load_dataset_CelebA_nosampler


a = load_dataset_CelebA_nosampler(train=True, batch_size=64, class_cond=True, random_crop=False, random_flip=True, image_size=32)

for i in range(1000):
    batch, _ = next(a)
    print(batch.shape)

