import my_utils

source_train_txt = "douyin_v3_152/lists/rgb_train_v1.txt"
source_val_txt = "douyin_v3_152/lists/rgb_val_v1.txt"
source_classInd_txt = "douyin_v3_152/lists/classInd.txt"

out_train_txt = "douyin_v3_152/lists/rgb_train_v1.txt"
out_val_txt = "douyin_v3_152/lists/rgb_val_v1.txt"
out_classInd_txt = "douyin_v3_152/lists/classInd.txt"

delete_tags = ["表情"]


def delete_tag_in_calssInd(classInd_file, tags, out_file):
    class_names = my_utils.get_classInd(classInd_file)
    for tag in tags:
        try:
            class_names.remove(tag)
        except ValueError:
            print("no tags in the classInd file!")
    my_utils.dump_classInd(class_names, out_file)
    return class_names


def delete_tag_in_list(source_file, tags, class_names, out_file):
    new_list_dic = []
    list_dic = my_utils.get_train_val_list(source_file)
    print("length of file before deleting is: ", len(list_dic))

    # 删除tag对应的行
    for dic in list_dic:
        tag, _ = my_utils.parse_frame_path(dic.get('frame'))
        if tag not in tags:
            # 更改dic中classIndex的值
            dic['class_index'] = class_names.index(tag)
            new_list_dic.append(dic)
    print("length of file after deleting is:", len(new_list_dic))

    my_utils.dump_train_val_list(new_list_dic, out_file)

# delete_tag(source_train_txt, delete_tags, out_train_txt)
# delete_tag(source_val_txt, delete_tags, out_val_txt)


class_names = delete_tag_in_calssInd(
    source_classInd_txt, delete_tags, out_classInd_txt)
delete_tag_in_list(source_val_txt, delete_tags, class_names, out_val_txt)
delete_tag_in_list(source_train_txt, delete_tags, class_names, out_train_txt)
