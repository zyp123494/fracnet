#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from itertools import product
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import torch
from skimage.measure import regionprops
from torch.utils.data import DataLoader, Dataset
import random
        
        
class FracNetTrainDataset(Dataset):

    def __init__(self, image_dir, label_dir=None, crop_size=64,
            transforms=None, num_samples=4, train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.public_id_list = sorted([x.split("-")[0]
            for x in os.listdir(image_dir)])
        self.crop_size = crop_size
        self.transforms = transforms
        self.num_samples = num_samples
        self.train = train


    def __len__(self):
        return len(self.public_id_list)

    @staticmethod
    def _get_pos_centroids(label_arr):
        centroids = [tuple([round(x) for x in prop.centroid])
            for prop in regionprops(label_arr)]

        return centroids

    @staticmethod
    def _get_symmetric_neg_centroids(pos_centroids, x_size):
        sym_neg_centroids = [(x_size - x, y, z) for x, y, z in pos_centroids]

        return sym_neg_centroids

    @staticmethod
    def _get_spine_neg_centroids(shape, crop_size, num_samples):
        x_min, x_max = shape[0] // 2 - 40, shape[0] // 2 + 40
        y_min, y_max = 300, 400
        z_min, z_max = crop_size // 2, shape[2] - crop_size // 2
        spine_neg_centroids = [(
            np.random.randint(x_min, x_max),
            np.random.randint(y_min, y_max),
            np.random.randint(z_min, z_max)
        ) for _ in range(num_samples)]

        return spine_neg_centroids

    def _get_neg_centroids(self, pos_centroids, image_shape):
        num_pos = len(pos_centroids)
        sym_neg_centroids = self._get_symmetric_neg_centroids(
            pos_centroids, image_shape[0])

        if num_pos < self.num_samples // 2:
            spine_neg_centroids = self._get_spine_neg_centroids(image_shape,
                self.crop_size, self.num_samples - 2 * num_pos)
        else:
            spine_neg_centroids = self._get_spine_neg_centroids(image_shape,
                self.crop_size, num_pos)

        return sym_neg_centroids + spine_neg_centroids

    def _get_roi_centroids(self, label_arr):
        #if self.train:
            # generate positive samples' centroids
        pos_centroids = self._get_pos_centroids(label_arr)

            # generate negative samples' centroids
        neg_centroids = self._get_neg_centroids(pos_centroids,
                label_arr.shape)

            # sample positives and negatives when necessary
        num_pos = len(pos_centroids)
        num_neg = len(neg_centroids)
        if num_pos >= self.num_samples:
            num_pos = self.num_samples // 2
            num_neg = self.num_samples // 2
        elif num_pos >= self.num_samples // 2:
            num_neg = self.num_samples - num_pos

        if num_pos < len(pos_centroids):
            pos_centroids = [pos_centroids[i] for i in np.random.choice(
                    range(0, len(pos_centroids)), size=num_pos, replace=False)]
        if num_neg < len(neg_centroids):
            neg_centroids = [neg_centroids[i] for i in np.random.choice(
                    range(0, len(neg_centroids)), size=num_neg, replace=False)]

        roi_centroids = pos_centroids + neg_centroids
        '''else:
            roi_centroids = [list(range(0, x, y // 2))[1:-1] + [x - y // 2]
                for x, y in zip(label_arr.shape, self.crop_size)]
            roi_centroids = list(product(*roi_centroids))

        roi_centroids = [tuple([int(x) for x in centroid])
            for centroid in roi_centroids]'''

        return roi_centroids

    def _crop_roi(self, arr, centroid):
        roi = np.ones(tuple([self.crop_size] * 3)) * (-1024)

        src_beg = [max(0, centroid[i] - self.crop_size // 2)
            for i in range(len(centroid))]
        src_end = [min(arr.shape[i], centroid[i] + self.crop_size // 2)
            for i in range(len(centroid))]
        dst_beg = [max(0, self.crop_size // 2 - centroid[i])
            for i in range(len(centroid))]
        dst_end = [min(arr.shape[i] - (centroid[i] - self.crop_size // 2),
            self.crop_size) for i in range(len(centroid))]
        roi[
            dst_beg[0]:dst_end[0],
            dst_beg[1]:dst_end[1],
            dst_beg[2]:dst_end[2],
        ] = arr[
            src_beg[0]:src_end[0],
            src_beg[1]:src_end[1],
            src_beg[2]:src_end[2],
        ]

        return roi

    def _apply_transforms(self, image,label):
        for t in self.transforms:
            image,label = t(image,label)

        return image,label
         
    def random_scale(self,image,label):
        if random.uniform(0,1)<0.3:
            scale = random.uniform(0.85,1.25)
            scale = np.round(scale,2)
            image,label = torch.from_numpy(image),torch.from_numpy(label)
            image = image.unsqueeze(0).unsqueeze(0)
            label = label.unsqueeze(0).unsqueeze(0).float()
            image = F.interpolate(image,scale_factor = scale,mode = 'trilinear',align_corners = False,recompute_scale_factor = True)
            label = F.interpolate(label,scale_factor = scale,mode = 'nearest',recompute_scale_factor = True)
            image = image.numpy().squeeze(0).squeeze(0)
            label = label.numpy().squeeze(0).squeeze(0).astype(np.uint8)
        return image,label
    def __getitem__(self, idx):
        # read image and label
        public_id = self.public_id_list[idx]
        image_path = os.path.join(self.image_dir, f"{public_id}-image.nii.gz")
        label_path = os.path.join(self.label_dir, f"{public_id}-label.nii.gz")
        image = nib.load(image_path)
        label = nib.load(label_path)
        image_arr = image.get_fdata().astype(np.float)
        label_arr = label.get_fdata().astype(np.uint8)
        #if self.train:
            #image_arr,label_arr = self.random_scale(image_arr,label_arr)
        #if self.transforms is not None:
            #image_arr,label_arr = self._apply_transforms(image_arr,label_arr)

        # calculate rois' centroids
        roi_centroids = self._get_roi_centroids(label_arr)

        # crop rois
        image_rois = [self._crop_roi(image_arr, centroid)
            for centroid in roi_centroids]
        label_rois = [self._crop_roi(label_arr, centroid)
            for centroid in roi_centroids]

        if self.transforms is not None:
            tmp1 = image_rois
            tmp2 = label_rois
            image_rois,label_rois =[],[]
            for i in range(len(tmp1)):
                #image_rois = [self._apply_transforms(image_roi)
                #for image_roi in image_rois]
                image_roi,label_roi = self._apply_transforms(tmp1[i],tmp2[i])
                image_rois.append(image_roi)
                label_rois.append(label_roi)
               
            del tmp1
            del tmp2

        image_rois = torch.tensor(np.stack(image_rois)[:, np.newaxis],
            dtype=torch.float)
        label_rois = (np.stack(label_rois) > 0).astype(np.float)
        label_rois = torch.tensor(label_rois[:, np.newaxis],
            dtype=torch.float)
        

        return image_rois, label_rois


   

    @staticmethod
    def collate_fn(samples):
        image_rois = torch.cat([x[0] for x in samples])
        label_rois = torch.cat([x[1] for x in samples])

        return image_rois, label_rois
        
    
            

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle,
            num_workers=num_workers, collate_fn=FracNetTrainDataset.collate_fn)


class FracNetInferenceDataset(Dataset):

    def __init__(self, image_path, crop_size=64, transforms=None):
        image = nib.load(image_path)
        self.image_affine = image.affine
        self.image = image.get_fdata().astype(np.int16)
        self.crop_size = crop_size
        self.transforms = transforms
        self.centers = self._get_centers()

    def _get_centers(self):
        dim_coords = [list(range(0, dim, self.crop_size // 2))[1:-1]            + [dim - self.crop_size // 2] for dim in self.image.shape]
        centers = list(product(*dim_coords))

        return centers

    def __len__(self):
        return len(self.centers)

    def _crop_patch(self, idx):
        center_x, center_y, center_z = self.centers[idx]
        patch = self.image[
            center_x - self.crop_size // 2:center_x + self.crop_size // 2,
            center_y - self.crop_size // 2:center_y + self.crop_size // 2,
            center_z - self.crop_size // 2:center_z + self.crop_size // 2
        ]

        return patch

    def _apply_transforms(self, image):
        for t in self.transforms:
            image = t(image,training = False)

        return image

    def __getitem__(self, idx):
        image = self._crop_patch(idx)
        center = self.centers[idx]

        if self.transforms is not None:
            image = self._apply_transforms(image)

        image = torch.tensor(image[np.newaxis], dtype=torch.float)

        return image, center

    @staticmethod
    def _collate_fn(samples):
        images = torch.stack([x[0] for x in samples])
        centers = [x[1] for x in samples]

        return images, centers

    @staticmethod
    def get_dataloader(dataset, batch_size, num_workers=0):
        return DataLoader(dataset, batch_size, num_workers=num_workers,
            collate_fn=FracNetInferenceDataset._collate_fn)

