import random


def loop_padding(img_list, size=64):
    out = img_list

    for img in img_list:
        if len(img_list) >= size:
            break
        out.append(img)

    return out


class TemporalBeginCrop(object):
    """
    Temporally crop the given frame indices at a beginning.

        If the number of frames is less than the size,
        loop the indices as many times as necessary to satisfy the size.

        Args:
            size (int): Desired output size of the crop.

    """

    def __init__(self, size=64):
        self.size = size

    def __call__(self, img_list):
        img_list = loop_padding(img_list, self.size)

        out = img_list[:self.size]

        return out


class TemporalRandomCrop(object):
    """
    Temporally crop the given frame indices at a random location.

        If the number of frames is less than the size,
        loop the indices as many times as necessary to satisfy the size.

        Args:
            size (int): Desired output size of the crop.

    """

    def __init__(self, size=64):
        self.size = size

    def __call__(self, img_list):
        """
        Args:
            img_list (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.

        """
        img_list = loop_padding(img_list, self.size)

        rand_end = max(0, len(img_list) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(img_list))

        out = img_list[begin_index:end_index]
        print("output length is {}".format(len(out)))
        return out


class TemporalCenterCrop(object):
    """
    Temporally crop the given frame indices at a center.

        If the number of frames is less than the size,
        loop the indices as many times as necessary to satisfy the size.

        Args:
            size (int): Desired output size of the crop.

    """

    def __init__(self, size=64):
        self.size = size

    def __call__(self, img_list):
        """
        Args:
            img_list (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        img_list = loop_padding(img_list, self.size)

        center_index = len(img_list) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(img_list))

        out = img_list[begin_index:end_index]

        return out
