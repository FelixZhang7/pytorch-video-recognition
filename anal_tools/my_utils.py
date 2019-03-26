def parse_video_folder(clean_video_folder):
    """
    Parse the video folder.
    :param clean_video_folder:(str) Path to the clean videos.
        And the directory tree is like:

            clean_video_folder:
                |- 仓鼠_1548957900724225
                |-  |- v0200f9a0000bcritg8m4cilqdqrmr90.mp4
                |-  |- ...
                |- ...

    :return:
        dict_lst: A dict of list. The keys are tags.
            The list contains all videos under the key.

            dict{
                "排球": ["vid1", "vid2", "vid3", ... ],
                "篮球": [...]
                ...
                }

    """
    dict_lst = {}
    import os
    video_folders = os.listdir(clean_video_folder)
    for folder in video_folders:
        tag = folder.split('_')[0]
        video_lst = os.listdir(os.path.join(clean_video_folder, folder))
        if tag not in dict_lst.keys():
            dict_lst[tag] = [os.path.splitext(video)[0] for video in video_lst]
        else:
            dict_lst[tag].extend([os.path.splitext(video)[0] for video in video_lst])
    return dict_lst


def get_classInd(classInd_file):
    """
    Get classInd from classInd file.

    Args:
        classInd_file: (str) Path to classInd.txt. For example:

            source_classInd_txt = "douyin_v2_153/lists/classInd.txt"

    Returns:
        A list of class_name.

            ["tag0", "tag1", ... ]

    """
    class_names = []
    with open(classInd_file) as f:
        lines = f.readlines()
    for line in lines:
        class_names.append(line.strip().split(' ')[1])
    return class_names


def dump_classInd(class_names, out_file):
    """
    Dump a class_names list to classInd.txt.

    Args:
        class_names: A list of tags.
        out_file(str): Path to output file.

    Return:

    """
    content = []
    for ii, item in enumerate(class_names):
        line = str(ii)+' '+item+'\n'
        content.append(line)
    with open(out_file, "w") as f:
        f.writelines(content)


def parse_train_val_list(list_file):
    """
    Get list of train/val list from list file.

    Args:
        list_file: Path to train/val list file. The format of
            the list file format is:

            "frames/v_跆拳道_v0200f9a0000bcritg8m4cilqdqrmr90 182 37"

    Return:
        list_dic: A list of dic. For example:

            [
            {frame: "Path_to_frame",
             count: number,
             class_index: index},
            {...},
            ...]

    """
    list_dic = []
    with open(list_file) as f:
        lines = f.readlines()
    for line in lines:
        frame_path, count, cls_index = line.strip().split(' ')
        dic = {'frame': frame_path,
               'count': int(count),
               'class_index': int(cls_index)}
        list_dic.append(dic)
    return list_dic


def dump_train_val_list(list_dic, out_file):
    """
    Dump the list of dict into tran/val list file.

    Args:
        list_dic: A list of dict. The format of the dic is:

            [
            {frame: "Path_to_frame",
             count: number,
             class_index: index},
            {...},
            ...]

        out_file: Path to output file.

    Return:

        The format of the output list is:

            "frames/v_跆拳道_v0200f9a0000bcritg8m4cilqdqrmr90 182 37"

    """
    content = []
    for dic in list_dic:
        line = dic.get('frame') + ' ' + str(dic.get('count')) + \
            ' ' + str(dic.get('class_index')) + '\n'
        content.append(line)

    with open(out_file, "w") as f:
        f.writelines(content)


def parse_frame_path(frame_path):
    """
    Parse the frame path
    :param frame_path: (str)Path of the frames.

        The format of the output list is:

            "frames/v_跆拳道_v0200f9a0000bcritg8m4cilqdqrmr90"

    :return:
        tag:(str) The tag of the video.
        vid:(str) The ID of the video.
    """
    return frame_path.split('/')[1].split('_')[1:]


def parse_reference_result(reference_result_file):
    """
    Parse the reference result.

    Args:
        reference_result_file: Path to reference result txt.

    Return:
        y_pred, y_gt: Two list of class index.

    """
    y_pred = []
    y_gt = []
    with open(reference_result_file) as f:
        results = f.readlines()

    for line in results:
        pred = line.split(' ')[2]
        gt = line.split(' ')[3].strip()
        y_pred.append(int(pred))
        y_gt.append(int(gt))

    assert len(y_pred) == len(y_gt), "Length of y_pred and y_gt are not equal!"
    return y_pred, y_gt


def calculate_acc(cnf_matrix, good_pairs=None):
    """
    Calculate ACC form a confusion matrix.
    :param
        cnf_matrix: A confusion matrix.
        good_pairs: A list of extra pairs that count for acc.
    :return:
    """
    right_sum = 0
    for i in range(cnf_matrix.shape[0]):
        right_sum += cnf_matrix[i, i]

    if good_pairs:
        for pair in good_pairs:
            right_sum += cnf_matrix[pair[0], pair[1]]

    acc = right_sum / cnf_matrix.sum()
    return acc
