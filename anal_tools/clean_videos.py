"""
Usage:
    python Analysis/tools/clean_videos.py --source_root="douyin_data/v1" \
                                          --target_root="douyin_data/v2"

source:v2
target:v3

clean_video_folder = target...

dict_lst = get_videos(c_v_f)
            :return dict{
                        "排球": ["vid1", "vid2", "vid3", ... ],
                        "篮球": [...]
                        ...
                        }

list_dic = get_train_val_list(source_list)
print len(l_dic)

new_list_dic = []

for dic in list_dic:
    tag, vid = my_ut.parse_frame()
    clean_videos = dict_lst.get('tag')
    if vid in clean_videos:
        new_list_dic.append(dic)

print len(new_l_dic)

my_utils.dump_list(new_l_dic, out_file)

"""

import my_utils
import argparse

parser = argparse.ArgumentParser(description='Modify lists after cleaning videos')
parser.add_argument('--source_root', '-s')
parser.add_argument('--target_root', '-t')

# source_root = "douyin_v2_153"
# target_root = "douyin_v3_152"


def main():
    args = parser.parse_args()
    source_root = args.source_root
    target_root = args.target_root

    source_train_list = source_root+'/lists/rgb_train.txt'
    source_val_list = source_root+'/lists/rgb_val.txt'

    target_train_list = target_root+'/lists/rgb_train.txt'
    target_val_list = target_root+'/lists/rgb_val.txt'

    clean_video_folder = target_root + '/clean_videos'

    dict_of_vid_lst = my_utils.parse_video_folder(clean_video_folder)

    ssum = 0
    for tag in dict_of_vid_lst.keys():
        ssum += len(dict_of_vid_lst.get(tag))

    print("total clean videos: ", ssum)

    for pair in [(source_train_list, target_train_list),
                 (source_val_list, target_val_list)]:
        list_of_frame_dic = my_utils.parse_train_val_list(pair[0])
        print("length of list before cleaning: ", len(list_of_frame_dic))

        new_list_of_frame_dic = []

        for dic in list_of_frame_dic:
            tag, vid = my_utils.parse_frame_path(dic.get('frame'))
            if tag in dict_of_vid_lst.keys():
                clean_videos = dict_of_vid_lst.get(tag)
                if vid in clean_videos:
                    new_list_of_frame_dic.append(dic)
            else:
                new_list_of_frame_dic.append(dic)

        print("length of list after cleaning: ", len(new_list_of_frame_dic))

        my_utils.dump_train_val_list(new_list_of_frame_dic, pair[1])


if __name__ == '__main__':
    main()
