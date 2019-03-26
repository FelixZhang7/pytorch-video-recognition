#!/usr/bin/env python
"""Dataset related utility functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2016-12
"""
import os
import os.path as osp
import skimage.io
import easydict
from . import pr
from file import load_lines


def load_img_list(list_path, fmt='path_id', ntop=1):
    """Load image list.

    Input
      list_path   -  list path
      fmt         -  format of each line in the list, {'path_id'} | 'path_ids' | 'path_prob' | 'label_path' | 'path_label' | 'path'
                       'path_id': img path + space + label id
                       'path_ids': img path + space + label id 1 + label id 2 + ...
                       'path_prob': img path + space + probability
                       'path_id_prob': img path + space + label id + label probability
                       'label_path': label name + space + img path
                       'path_label': img path + space + label name
                       'path': img path
      ntop        -  ntop, {1} | 2 | ...
                       used when fmt = 'path_ids'

    Output
      img_paths   -  list of image path, n x
      img_labels  -  list of label, n x
    """
    pr.in_func('load_img_list', 'list_path {}, fmt {}'.format(list_path, fmt))

    # load list file
    lines = load_lines(list_path)
    num_line = len(lines)

    # parse each line
    img_paths, img_labels = [], []
    for line_id in xrange(num_line):
        terms = lines[line_id].split()

        if fmt == 'label_path':
            img_label = terms[0]
            img_path = terms[1]

        elif fmt == 'path_label':
            img_label = terms[1]
            img_path = terms[0]

        elif fmt == 'path_id':
            img_label = int(terms[1])
            img_path = terms[0]

        elif fmt == 'path_prob':
            img_label = float(terms[1])
            img_path = terms[0]

        elif fmt == 'path_ids':
            img_path = terms[0]
            img_label = []
            for itop in xrange(ntop):
                img_label.append(int(terms[itop + 1]))

        elif fmt == 'path_id_prob':
            img_path = terms[0]
            img_label = []
            img_label.append(int(terms[1]))
            img_label.append(float(terms[2]))

        elif fmt == 'path':
            img_label = 0
            img_path = terms[0]

        else:
            raise Exception('unknown fmt: {}'.format(fmt))

        img_paths.append(img_path)
        img_labels.append(img_label)

    pr.out_func()
    return img_paths, img_labels


def group_img_list(img_paths, class_ids, num_class):
    """Group img list into class groups.

    Input
      img_paths   -  image paths, num_img x
      clasa_ids   -  class id, num_img x
      num_class   -  #class

    Output
      img_pathss  -  img path groups, num_class x, num_img_i x
    """
    img_pathss = [[] for class_id in xrange(num_class)]
    num_img = len(img_paths)

    for img_id in xrange(num_img):
        img_path = img_paths[img_id]
        class_id = class_ids[img_id]
        if class_id>=0:
            img_pathss[class_id].append(img_path)
        else:
            pass

    return img_pathss
