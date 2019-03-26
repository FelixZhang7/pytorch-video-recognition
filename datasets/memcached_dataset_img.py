import mc
import numpy as np
from PIL import Image
from utils.img import pil_loader
from .memcached_dataset import McDataset


class McDataset_img(McDataset):
    def __init__(self, root_dir, meta_file, transform=None, fake=False):
        self.fake = fake

        super(McDataset_img, self).__init__(
            root_dir, meta_file, transform=transform)
        # prepare fake img
        if self.fake:
            img = np.random.randint(0, 256, (350, 350, 3), dtype=np.uint8)
            self.img = Image.fromarray(img)
            if self.transform is not None:
                self.img = self.transform(self.img)
        self._parse_list()

    def __len__(self):
        return self.num

    def _parse_list(self):
        with open(self.meta_file) as f:
            lines = f.readlines()
        self.metas = []
        for line in lines:
            path, cls = line.rstrip().split()
            self.metas.append((path, int(cls)))
        self.num = len(lines)

    def __getitem__(self, idx):
        if self.fake:
            img = self.img
            cls = 0
        else:
            filename = self.root_dir + '/' + self.metas[idx][0]
            cls = self.metas[idx][1]
            # # memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(filename, value)
            value_str = mc.ConvertBuffer(value)
            img = pil_loader(value_str)

        # # transform
        if self.transform is not None and not self.fake:
            img = self.transform(img)
        return img, cls
