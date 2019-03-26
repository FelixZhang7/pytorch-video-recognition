# -*- coding: utf-8 -*-

"""
This script is used for analyzing the reference result of a model.
After evaluation and getting the reference result file.

"""
import my_utils
import argparse
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='Analysis the reference result.')
parser.add_argument('--exp_path', '-e', help="Experiment PATH")


def get_max():
    global cm_bak
    for i in range(cm_bak.shape[0]):
        for j in range(cm_bak.shape[1]):
            if cm_bak[i][j] == cm_bak.max():
                value = cm_bak.max()
                cm_bak[i][j] = 0
                return i, j, value


def save_errors(cnf_matrix, normalized_cnf_matrix, k, class_names, good_pairs):
    # Calculate top k errors and save
    global args
    content = []
    for i in range(k):
        print("number {} !!!!".format(i))
        pair_count = get_max()
        if [pair_count[0], pair_count[1]] in good_pairs:
            continue
        line = "gt:{:>4}({:>5})\tpred:{:>4}({:>5})\tcount:{:>4}\tprec:{}\terr{}\tgt_samples:{}/{}\n".format(
            pair_count[0], class_names[pair_count[0]],
            pair_count[1], class_names[pair_count[1]],
            pair_count[2], round(normalized_cnf_matrix[pair_count[0], pair_count[0]], 2),
            round(normalized_cnf_matrix[pair_count[0], pair_count[1]], 2),
            cnf_matrix[pair_count[0]].sum(), cnf_matrix.sum())
        content.append(line)

    result_file = args.exp_path + '/top{}_errors.txt'.format(k)
    with open(result_file, "w") as f:
        f.writelines(content)


def main():
    global args, config, cm_bak
    args = parser.parse_args()

    # Parse the reference result
    reference_result_file = args.exp_path + '/reference_result.txt'
    y_pred, y_gt = my_utils.parse_reference_result(reference_result_file)

    config_file = args.exp_path + '/config.yaml'
    with open(config_file) as f:
        config = yaml.load(f)

    train_set_root = config.get('common').get('train_root')
    class_ind_file = train_set_root+'/lists/classInd.txt'

    # Get the class name list from classInd file
    class_names = my_utils.get_classInd(class_ind_file)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_gt, y_pred)
    np.set_printoptions(precision=2)

    # Normalize the confusion matrix
    normalized_cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")

    # Zero the element of the diagonal line to analyse errors.
    cm_bak = np.copy(cnf_matrix)
    for i in range(cm_bak.shape[0]):
        cm_bak[i][i] = 0

    # #### add rules ######
    son2father_rules = {
        "火锅": "饮食",
        "多肉": "盆栽",
        "烧烤": "饮食",
        "潜水": "游泳",
        "饺子": "饮食",
        "日出日落": "天空",
    }

    # Mapping the rules to class index.
    good_pairs = []
    for key, value in son2father_rules.items():
        good_pairs.append([class_names.index(key), class_names.index(value)])

    # Calculate the Accuracy
    acc_origin = my_utils.calculate_acc(cnf_matrix)
    acc_with_pairs = my_utils.calculate_acc(cnf_matrix, good_pairs)

    print("ACC_origin is {} !!!!".format(acc_origin))
    print("ACC_new is {} !!!!".format(acc_with_pairs))

    # Write top k errors to file
    save_errors(cnf_matrix, normalized_cnf_matrix, 100, class_names, good_pairs)


if __name__ == '__main__':
    main()

