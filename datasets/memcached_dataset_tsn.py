import mc
import numpy as np
import os
from numpy.random import randint
from .memcached_dataset import McDataset
from utils.img import pil_loader


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[2])


class McDataset_tsn(McDataset):
    def __init__(self, root_dir, meta_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='{:06d}.jpg', transform=None,
                 random_shift=True, test_mode=False, temporal_aug=None):
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_segments = num_segments
        self.temporal_aug = temporal_aug
        super(McDataset_tsn, self).__init__(
            root_dir, meta_file,
            transform=transform)

        self._parse_list()

    def __len__(self):
        return len(self.video_list)

    def _parse_list(self):
        self.video_list = []
        for x in open(self.meta_file):
            x_strip = x.strip()
            x_list = x_strip.split(' ')
            if len(x_list) > 3:
                self.video_list.append(VideoRecord(
                    [' '.join(x_list[0:len(x_list)-2]), x_list[-2], x_list[-1]]))
                continue
            self.video_list.append(VideoRecord(x_list))
        # self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.meta_file)]

    def _sample_indices(self, record):
        ave_dur = (record.num_frames - self.new_length +
                   1) // self.num_segments
        if ave_dur > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  ave_dur) + randint(ave_dur, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames -
                                      self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments, ))
            print('the video frames are null')
        return offsets

    def _sample_val_indices(self, record):
        tick = (record.num_frames - self.num_segments + 1) / \
            float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick*x)
                            for x in range(self.num_segments)])
        return offsets

    def __getitem__(self, idx):
        record = self.video_list[idx]

        segment_indices = self._sample_indices(
            record) if self.random_shift else self._sample_val_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        cls = record.label
        frames_path = record.path
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            filename = os.path.join(
                self.root_dir, record.path,  self.image_tmpl.format(p))

            # value_str = filename

            img = []
            try:
                # memcached
                self._init_memcached()
                value = mc.pyvector()
                self.mclient.Get(filename, value)
                value_str = mc.ConvertBuffer(value)
                img = pil_loader(value_str)
            except:
                p -= 1
                filename = os.path.join(
                    self.root_dir, record.path, self.image_tmpl.format(p))
                # memcached
                self._init_memcached()
                value = mc.pyvector()
                self.mclient.Get(filename, value)
                value_str = mc.ConvertBuffer(value)
                img = pil_loader(value_str)

            images.extend(img)

        transformed_group_imgs = self.transform(images)
        # return process_data, record.label
        return transformed_group_imgs, cls, frames_path
