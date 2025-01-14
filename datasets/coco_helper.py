import random

from hfai.datasets.coco import CocoCaption, CocoReader
from torch.utils.data.distributed import DistributedSampler
from ffrecord.torch import DataLoader
import hfai.datasets
from hfai.datasets.base import BaseDataset, get_data_dir, register_dataset
from ffrecord import FileReader
from typing import Callable, List, Optional
from pathlib import Path
import  numpy as np
import os
import torchvision.transforms as transforms

class CocoReaderLocal(CocoReader):
    def __init__(self, split: str, check_data: bool = True, miniset: bool = False, local_data=None):
        if local_data is None:
            data_dir = get_data_dir()
        else:
            data_dir = Path(local_data)
        if miniset:
            data_dir = data_dir / "mini"
        self.data_dir = data_dir / "COCO"

        assert split in ["train", "val"]
        self.split = split
        self.fname = self.data_dir / f"{split}2017.ffr"
        self.reader = FileReader(self.fname, check_data)

        self.panoptic_fname = self.data_dir / f"panoptic_{split}2017.ffr"
        self.panoptic_reader = FileReader(self.panoptic_fname, check_data)

        self.ids = None
        self.coco = None


class CocoCaptionOnly(CocoCaption):
    def __init__(
            self,
            split: str,
            transform: Optional[Callable] = None,
            check_data: bool = True,
            miniset: bool = False,
            data_folder="data"
    ):
        self.split = split
        self.reader = CocoReaderLocal(split, check_data, miniset, data_folder)
        self._load_annotations()  # load annotations into memory
        self.transform = transform
        self.coco = self.reader.coco


    def __getitem__(self, indices):
        n = len(indices)
        selected = np.random.randint(0, 5, (n,))
        annos = []
        for i in range(n):
            annos.append(self.reader.read_anno(indices[i])[selected[i]]['caption'])
            # annos.append(self.reader.read_anno(indices[i])[selected[i]]['caption'])
        # annos = [self.reader.read_anno(idx) for idx in indices] # test this one
        # img_ids = [self.reader.ids[idx] for idx in indices]
        # samples = list(zip(img_ids, annos))
        return annos

class CocoCaptionWOthers(CocoCaption):
    def __init__(
            self,
            split: str,
            transform: Optional[Callable] = None,
            check_data: bool = True,
            miniset: bool = False,
            data_folder="data"
    ):
        self.split = split
        self.reader = CocoReaderLocal(split, check_data, miniset, data_folder)
        self._load_annotations()  # load annotations into memory
        self.transform = transform
        self.coco = self.reader.coco


    def __getitem__(self, indices):
        n = len(indices)
        selected = np.random.randint(0, 5, (n,))
        annos = []
        annos_others = []
        for i in range(n):
            annos_other = []
            anno_list = self.reader.read_anno(indices[i])
            len_list_anno = len(anno_list)
            for j in range(len_list_anno):
                if j == selected[i]:
                    annos.append(anno_list[j]['caption'])
                else:
                    if len(annos_other) >= 4:
                        continue
                    annos_other.append(anno_list[j]['caption'])
            annos_others.append(annos_other)
        return list(zip(annos, annos_others))


class CocoCaptionWithImages(CocoCaption):
    def __init__(
            self,
            split: str,
            transform: Optional[Callable] = None,
            check_data: bool = True,
            miniset: bool = False,
    ):
        super(CocoCaptionWithImages, self).__init__(split, transform, check_data, miniset)
        self.split = split
        self.reader = CocoReader(split, check_data, miniset)
        self._load_annotations()  # load annotations into memory
        self.transform = transform
        self.coco = self.reader.coco


    def __getitem__(self, indices):
        n = len(indices)
        imgs = self.reader.read_imgs(indices)
        selected = np.random.randint(0, 5, (n,))
        annos = []
        set_annos = []
        transformed_imgs = []
        for i in range(n):
            set_anno = self.reader.read_anno(indices[i])
            set_annos.append(set_anno)
            annos.append(set_anno[selected[i]]['caption'])
            transformed_imgs.append(self.transform(imgs[i]))
            # annos.append(self.reader.read_anno(indices[i])[selected[i]]['caption'])
        # annos = [self.reader.read_anno(idx) for idx in indices] # test this one
        img_ids = [self.reader.ids[idx] for idx in indices]
        samples = list(zip(transformed_imgs, img_ids, annos, set_annos))
        return samples

class CocoCaptionLocal(CocoCaption):
    def __init__(
            self,
            split: str,
            transform: Optional[Callable] = None,
            check_data: bool = True,
            miniset: bool = False,
            data_folder="data"
    ):
        self.split = split
        self.reader = CocoReaderLocal(split, check_data, miniset, data_folder)
        self._load_annotations()  # load annotations into memory
        self.transform = transform
        self.coco = self.reader.coco

    def __getitem__(self, indices):
        imgs = self.reader.read_imgs(indices)
        annos = [self.reader.read_anno(idx) for idx in indices]
        img_ids = [self.reader.ids[idx] for idx in indices]
        if self.transform is not None:
            samples = [[self.transform(img), img_id, anno] for img, img_id, anno in zip(imgs, img_ids, annos)]
        else:
            samples = list(zip(imgs, img_ids, annos))
        return samples



def load_data_caption(
    *,
    split: str,
    batch_size: int,
):

    dataset = CocoCaptionOnly(split=split, miniset=False, data_folder="data")

    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True, drop_last=True)
    while True:
        yield from loader # put all items of loader into list and concat all list infinitely
    # return loader

def load_data_caption_hfai(
        *,
        split: str,
        batch_size: int,
):
    dataset = CocoCaptionOnly(split=split, miniset=False, data_folder=None)

    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True, drop_last=True)

    while True:
        yield from loader  # put all items of loader into list and concat all list infinitely

def load_data_caption_hfai_robust(
        *,
        split: str,
        batch_size: int,
):
    dataset = CocoCaptionWOthers(split=split, miniset=False, data_folder=None)

    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True, drop_last=True)

    while True:
        yield from loader  # put all items of loader into list and concat all list infinitely

import torch
import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        # if not all(len(elem) == elem_size for elem in it):
        #     raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
def load_image_and_caption_data_hfai(
        *,
        split:str,
        batch_size:int,
        miniset=False,
        transforms = None
):
    dataset = CocoCaptionWithImages(split=split, miniset=miniset, transform=transforms)
    data_sampler = DistributedSampler(dataset, shuffle=True)
    loader = dataset.loader(batch_size, num_workers=8, sampler=data_sampler, pin_memory=True, drop_last=True,
                            collate_fn=custom_collate)
    while True:
        yield from loader



def download_coco_caption():
    hfai.datasets.set_data_dir('data/')
    hfai.datasets.download("CocoCaption", miniset=False)

def coco_ref_64(out_dir):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(64)])
    dataset = CocoCaptionLocal(split="val", data_folder="data", transform=transform)
    n = len(dataset)
    list_images = []
    raw_images = os.path.join("data", "coco")
    os.makedirs(raw_images, exist_ok=True)
    for i in range(n):
        sample = dataset[[i]]
        arr_image = np.asarray(sample[0][0]).astype(np.uint8)
        print(arr_image.shape)
        list_images.append(arr_image)
        sample[0][0].save(os.path.join(raw_images, "image_%d.png"%i))
    list_images = np.stack(list_images)
    print(list_images.shape)
    save_file = os.path.join(out_dir, f"VIRTUAL_MSCOCO_val_64x64_squ128.npz")
    np.savez(save_file, list_images)

def coco_ref_256(out_dir):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256)])
    dataset = CocoCaptionLocal(split="val", data_folder="data", transform=transform)
    n = len(dataset)
    list_images = []
    raw_images = os.path.join("data", "coco")
    os.makedirs(raw_images, exist_ok=True)
    for i in range(n):
        sample = dataset[[i]]
        arr_image = np.asarray(sample[0][0]).astype(np.uint8)
        print(arr_image.shape)
        list_images.append(arr_image)
        sample[0][0].save(os.path.join(raw_images, "image_%d.png"%i))
    list_images = np.stack(list_images)
    print(list_images.shape)
    save_file = os.path.join(out_dir, f"VIRTUAL_MSCOCO_val_256x256_squ256.npz")
    np.savez(save_file, list_images)



if __name__ == '__main__':
    # coco_ref_64("reference")
    coco_ref_256("reference")
    # caption_loader = load_data_caption(split="val", batch_size=100)
    #
    # caption_iter = iter(caption_loader)
    #
    # for i in range(10000000):
    #     print(len(next(caption_iter)))