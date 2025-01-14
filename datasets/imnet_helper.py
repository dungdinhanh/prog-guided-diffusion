import hfai
import hfai.datasets

hfai.datasets.set_data_dir('data/imagenet/')
hfai.datasets.download("ImageNet", miniset=False)