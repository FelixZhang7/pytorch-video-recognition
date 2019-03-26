import mc
import os
from .memcached_dataset import McDataset
from utils.img import pil_loader
import utils.temporal_transforms as tt


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


class McDataset_slowfast(McDataset):
    def __init__(self, root_dir, meta_file,
                 image_tmpl='{:06d}.jpg', transform=None,
                 temporal_transform=None, sample_duration=64,
                 random_shift=True, test_mode=False):
        self.temporal_transform = temporal_transform
        self.image_tmpl = image_tmpl
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.sample_duration = sample_duration

        super(McDataset_slowfast, self).__init__(root_dir, meta_file,
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

    def __getitem__(self, idx):
        record = self.video_list[idx]
        return self.get(record)

    def get(self, record):
        cls = record.label
        clip_path = record.path
        images = list()
        for img_idx in range(record.num_frames):
            p = int(img_idx) + 1
            filename = os.path.join(
                self.root_dir, record.path, self.image_tmpl.format(p))

            img = []
            try:
                # # memcached
                self._init_memcached()
                value = mc.pyvector()
                self.mclient.Get(filename, value)
                value_str = mc.ConvertBuffer(value)
                img = pil_loader(value_str)
            except:
                p -= 1
                filename = os.path.join(
                    self.root_dir, record.path, self.image_tmpl.format(p))
                # # memcached
                self._init_memcached()
                value = mc.pyvector()
                self.mclient.Get(filename, value)
                value_str = mc.ConvertBuffer(value)
                img = pil_loader(value_str)

            images.extend(img)

        if not self.temporal_transform:
            # print("Add temporal_transform!")
            self.temporal_transform = tt.TemporalRandomCrop(
                size=self.sample_duration)

        images = self.temporal_transform(images)

        transformed_clips = self.transform(images)
        return transformed_clips, cls, clip_path
