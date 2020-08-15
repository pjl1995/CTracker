from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import cv2
import csv
from six import raise_from
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

import skimage.io
import skimage.transform
import skimage.color
import skimage
from PIL import Image, ImageEnhance

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]

class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, root_path, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform
        self.root_path = root_path
        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, obj_id, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())
        self.name2video_frames = dict()
        self.image_name_prefix = list()
        for image_name in self.image_names:
            self.image_name_prefix.append(image_name[0:-len(image_name.split('/')[-1].split('_')[-1])])
        self.image_name_prefix = set(self.image_name_prefix)
        print('total vedio count: {}'.format(len(self.image_name_prefix)))
        for image_name in self.image_names:
            cur_prefix = image_name[0:-len(image_name.split('/')[-1].split('_')[-1])]
            if cur_prefix not in self.name2video_frames:
                self.name2video_frames[cur_prefix] = 1
            else:
                self.name2video_frames[cur_prefix] = self.name2video_frames[cur_prefix] + 1

    def _extract_frame_index(self, image_name):
        suffix_name = image_name.split('/')[-1].split('_')[-1]
        return int(float(suffix_name.split('.')[0]))
    
    def _get_random_surroud_name(self, image_name, max_diff=3, ignore_equal=True, pos_only=True):
        suffix_name = image_name.split('/')[-1].split('_')[-1]
        prefix = image_name[0:-len(suffix_name)]
        cur_index = int(float(suffix_name.split('.')[0]))
        total_number = self.name2video_frames[prefix]
        if total_number < 2: return image_name
        next_index = cur_index
        while True:
            range_low = max(1, cur_index - max_diff)
            range_high = min(cur_index + max_diff, total_number)
            if pos_only: 
                range_low = cur_index
                if ignore_equal:
                    range_low = range_low + 1
                    if cur_index == total_number:
                        return image_name
        
            next_index = random.randint(range_low, range_high)
            if ignore_equal:
                if next_index == cur_index:
                    continue
            break
        
        return prefix + '{0:06}.'.format(next_index) + suffix_name.split('.')[-1]

    def _extract_name_prefix(self, image_name):
        return image_name[0:-len(image_name.split('/')[-1].split('_')[-1])]

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')


    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        while True:
            try:
                img = self.load_image(idx)
                next_name = self._get_random_surroud_name(self.image_names[idx])
                img_next = self.load_image_by_name(next_name)
                annot = self.load_annotations(idx)
                annot_next = self.load_annotationse_by_name(next_name)

                if (annot.shape[0] < 1) or (annot_next.shape[0] < 1):
                    idx = random.randrange(0, len(self.image_names))
                    continue
            except FileNotFoundError:
                print ('FileNotFoundError in process image.')
                idx = random.randrange(0, len(self.image_names))
                continue
            break

        if np.random.rand() < 0.5:
            sample = {'img': img, 'annot': annot, 'img_next': img_next, 'annot_next': annot_next}
        else:
            sample = {'img': img_next, 'annot': annot_next, 'img_next': img, 'annot_next': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image_by_name(self, image_name):
        img = skimage.io.imread(image_name)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img

    def load_annotationse_by_name(self, image_name):
        # get ground truth annotations
        annotation_list = self.image_data[image_name]
        annotations     = np.zeros((0, 6))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']
            obj_id = a['obj_id']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 6))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotation[0, 5]  = obj_id
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 6))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']
            obj_id = a['obj_id']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 6))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotation[0, 5]  = obj_id

            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, obj_id, x1, y1, x2, y2, class_name = row[:7]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,obj_id,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            img_file = os.path.join(self.root_path, img_file.strip())
            if img_file not in result:
                result[img_file] = []
            class_name = class_name.strip()

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(float(x1), int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(float(y1), int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(float(x2), int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(float(y2), int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name, 'obj_id': obj_id})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    imgs_next = [s['img_next'] for s in data]
    annots_next = [s['annot_next'] for s in data]
        
    widths = [int(s.shape[1]) for s in imgs]
    heights = [int(s.shape[0]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_height, max_width, 3)
    padded_imgs_next = torch.zeros(batch_size, max_height, max_width, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

        img_next = imgs_next[i]
        padded_imgs_next[i, :int(img_next.shape[0]), :int(img_next.shape[1]), :] = img_next

    max_num_annots = max(annot.shape[0] for annot in annots)
    max_num_annots_next = max(annot.shape[0] for annot in annots_next)
    max_num_annots = max(max_num_annots, max_num_annots_next)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 6)) * -1
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 6)) * -1
    
    if max_num_annots > 0:
        annot_padded_next = torch.ones((len(annots_next), max_num_annots, 6)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots_next):
                if annot.shape[0] > 0:
                    annot_padded_next[idx, :annot.shape[0], :] = annot
    else:
        annot_padded_next = torch.ones((len(annots_next), 1, 6)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    padded_imgs_next = padded_imgs_next.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'img_next': padded_imgs_next, 'annot_next': annot_padded_next}

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy + 1), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0] + 1) *
              (box_a[:, 3]-box_a[:, 1] + 1))  # [A,B]
    area_b = ((box_b[2]-box_b[0] + 1) *
              (box_b[3]-box_b[1] + 1))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0] + 1) *
              (box_a[:, 3]-box_a[:, 1] + 1))  # [A,B]
    return inter / area_a  # [A,B]


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, min_side=608, max_side=1024):
        return sample
        image, annots, image_next, annots_next = sample['img'], sample['annot'], sample['img_next'], sample['annot_next']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        image_next = skimage.transform.resize(image_next, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        new_image_next = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image_next[:rows, :cols, :] = image_next.astype(np.float32)

        annots[:, :4] *= scale
        annots_next[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'img_next': torch.from_numpy(new_image_next), 'annot_next': torch.from_numpy(annots_next), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image_next, annots_next = sample['img_next'], sample['annot_next']
            image = image[:, ::-1, :]
            image_next = image_next[:, ::-1, :]

            rows, cols, _ = image.shape
            rows_next, cols_next, _ = image_next.shape
            assert (rows == rows_next) and (cols == cols_next), 'size must be equal between adjacent images pair.'

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            # for next
            x1 = annots_next[:, 0].copy()
            x2 = annots_next[:, 2].copy()
            
            x_tmp = x1.copy()

            annots_next[:, 0] = cols - x2
            annots_next[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots, 'img_next': image_next, 'annot_next': annots_next}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[RGB_MEAN]])
        self.std = np.array([[RGB_STD]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']
        image_next, annots_next = sample['img_next'], sample['annot_next']

        return {'img':torch.from_numpy((image.astype(np.float32) / 255.0 - self.mean) / self.std), 'annot': torch.from_numpy(annots), 'img_next':torch.from_numpy((image_next.astype(np.float32) / 255.0-self.mean)/self.std), 'annot_next': torch.from_numpy(annots_next)}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = RGB_MEAN
        else:
            self.mean = mean
        if std == None:
            self.std = RGB_STD
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def random_brightness(img, img_next):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        delta = np.random.uniform(-0.125, 0.125) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
        img_next = ImageEnhance.Brightness(img_next).enhance(delta)
    return img, img_next


def random_contrast(img, img_next):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        delta = np.random.uniform(-0.5, 0.5) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
        img_next = ImageEnhance.Contrast(img_next).enhance(delta)
    return img, img_next


def random_saturation(img, img_next):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        delta = np.random.uniform(-0.5, 0.5) + 1
        img = ImageEnhance.Color(img).enhance(delta)
        img_next = ImageEnhance.Color(img_next).enhance(delta)
    return img, img_next


def random_hue(img, img_next):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        delta = np.random.uniform(-18, 18)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')

        img_next_hsv = np.array(img_next.convert('HSV'))
        img_next_hsv[:, :, 0] = img_next_hsv[:, :, 0] + delta
        img_next = Image.fromarray(img_next_hsv, mode='HSV').convert('RGB')
    return img, img_next



class PhotometricDistort(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, annots, image_next, annots_next = sample['img'], sample['annot'], sample['img_next'], sample['annot_next']
        prob = np.random.uniform(0, 1)
        # Apply different distort order
        img = Image.fromarray(image)
        img_next = Image.fromarray(image_next)
        if prob > 0.5:
            img, img_next = random_brightness(img, img_next)
            img, img_next = random_contrast(img, img_next)
            img, img_next = random_saturation(img, img_next)
            img, img_next = random_hue(img, img_next)
        else:
            img, img_next = random_brightness(img, img_next)
            img, img_next = random_saturation(img, img_next)
            img, img_next = random_hue(img, img_next)
            img, img_next = random_contrast(img, img_next)

        image = np.array(img)
        image_next = np.array(img_next)
        return {'img': image, 'annot': annots, 'img_next': image_next, 'annot_next': annots_next}

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, image_next):
        for t in self.transforms:
            img, image_next = t(img, image_next)
        return img, image_next

class RandomSampleCrop(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, annots, image_next, annots_next = sample['img'], sample['annot'], sample['img_next'], sample['annot_next']

        #print('crop1',image.dtype)
        height, width, _ = image.shape
        shorter_side = min(height, width)
        crop_size = np.random.uniform(0.3 * shorter_side, 0.8 * shorter_side)
        target_size = 512
        if shorter_side < 384: 
            target_size = 256
        min_iou = 0.2
        crop_success = False
        # max trails (10)
        for _ in range(20):
            left = np.random.uniform(0, width - crop_size)
            top = np.random.uniform(0, height - crop_size)

            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(left), int(top), int(left + crop_size), int(top + crop_size)])

            # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
            overlap = overlap_numpy(annots[:, :4], rect)
            overlap_next = overlap_numpy(annots_next[:, :4], rect)

            if overlap.max() < min_iou or overlap_next.max() < min_iou:
                continue
            crop_success = True
            image = image[rect[1]:rect[3], rect[0]:rect[2], :]
            image_next = image_next[rect[1]:rect[3], rect[0]:rect[2], :]
            annots = annots[overlap > min_iou, :].copy()
            annots_next = annots_next[overlap_next > min_iou, :].copy()

            annots[:, :2] -= rect[:2]
            annots[:, 2:4] -= rect[:2]

            annots_next[:, :2] -= rect[:2]
            annots_next[:, 2:4] -= rect[:2]
            #print('crop1',image.max())


            expand_ratio = 1.0
            if np.random.uniform(0, 1) > 0.75:
                height, width, depth = image.shape
                expand_ratio = random.uniform(1, 3)
                left = random.uniform(0, width * expand_ratio - width)
                top = random.uniform(0, height * expand_ratio - height)

                expand_image = np.zeros((int(height*expand_ratio), int(width*expand_ratio), depth), dtype=image.dtype)
        
                expand_image[:, :, :] = np.array([[RGB_MEAN]]) * 255.0
                expand_image[int(top):int(top + height),
                            int(left):int(left + width)] = image
                image = expand_image

                annots[:, :2] += (int(left), int(top))
                annots[:, 2:4] += (int(left), int(top))


                expand_next_image = np.zeros(
                    (int(height*expand_ratio), int(width*expand_ratio), depth),
                    dtype=image_next.dtype)
                expand_next_image[:, :, :] = np.array([[RGB_MEAN]]) * 255.0
                expand_next_image[int(top):int(top + height),
                            int(left):int(left + width)] = image_next
                image_next = expand_next_image

                annots_next[:, :2] += (int(left), int(top))
                annots_next[:, 2:4] += (int(left), int(top))

            # resize the image with the computed scale

            
            # resize the image with the computed scale
            image = (255.0 * skimage.transform.resize(image, (target_size, target_size))).astype(np.uint8)
            image_next = (255.0 * skimage.transform.resize(image_next, (target_size, target_size))).astype(np.uint8)
            annots[:, :4] *= (target_size / (crop_size * expand_ratio))
            annots_next[:, :4] *= (target_size / (crop_size * expand_ratio))
            #print('crop2',image.max())
            return {'img': image, 'annot': annots, 'img_next': image_next, 'annot_next': annots_next}
        if not crop_success:
            image = (255.0 * skimage.transform.resize(image, (height // 2, width // 2))).astype(np.uint8)
            image_next = (255.0 * skimage.transform.resize(image_next, (height // 2, width // 2))).astype(np.uint8)
            annots[:, :4] *= 0.5
            annots_next[:, :4] *= 0.5
        return {'img': image, 'annot': annots, 'img_next': image_next, 'annot_next': annots_next}


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
