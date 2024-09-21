import os
import re
import glob
import numpy as np
import pandas as pd


def get_filename_list(dir_path, pattern='*', ext='*'):
    """
    find all extention files under directory
    :param dir_path: directory path
    :param ext: extention name, like wav, png...
    :param pattern: filename pattern for searching
    :return: files path list
    """
    filename_list = []
    for root, _, _ in os.walk(dir_path):
        file_path_pattern = os.path.join(root, f'{pattern}.{ext}')
        files = sorted(glob.glob(file_path_pattern))
        filename_list += files
    return filename_list

def get_meta_list(file_list):

    label_list = []
    source_list = []
    for file in file_list:
        label_list.append(0 if 'normal' in file else 1)
        source_list.append('source' if 'source' in file else 'target')
    return label_list, source_list

def get_data_list(file_dirs):
    file_list = []
    for fd in file_dirs:
        files = get_filename_list(fd, ext="wav")
        file_list.extend(files)
        # print(f'len of files: {fd.split("/")[-2]} | {len(files)}')
    label_list, source_list = get_meta_list(file_list)
    return file_list, label_list, source_list

def get_test_data_list(test_dirs):
    root_path = "../"
    test_file_list = []
    test_label_list = []
    test_machine_list = []
    test_source_list = []
    for td in test_dirs:
        machine = td.split("/")[-2]
        test_files = get_filename_list(td, ext="wav")
        gt_list = np.array(pd.read_csv(root_path + 'dcase2023_task2_evaluator/ground_truth_data/ground_truth_' + machine + '_section_00_test.csv', header=None).iloc[:, 1]).tolist()
        s_list = np.array(pd.read_csv(root_path + 'dcase2023_task2_evaluator/ground_truth_domain/ground_truth_' + machine + '_section_00_test.csv', header=None).iloc[:, 1] == 0).tolist()
        s_list = ['source' if s else 'target' for s in s_list]
        test_file_list.extend(test_files)
        test_label_list.extend(gt_list)
        test_source_list.extend(s_list)
        test_machine_list.extend([machine] * len(test_files))
        # print(f'len of files: {td.split("/")[-2]} | {len(test_files)}')
    test_attr_list = [0] * len(test_file_list)
    return test_file_list, test_label_list, test_source_list, np.array(test_machine_list), test_attr_list

def get_attributes(file_list, source_list):
    machine_list = []
    attrs = []
    for idx, ext_id in enumerate(file_list):
        source = str(source_list[idx])
        machine = ext_id.split('/')[-3]
        file = ext_id.split('/')[-1]
        section_id = re.findall('section_[0-9][0-9]', file)[0].split('_')[-1]
        # if idx == 0:
        # print(f"att id: {file.split('.wav')[0].split('_')[6:]}")
        att_id = '_'.join(file.split('.wav')[0].split('_')[6:])
        machine_id = machine + '_' + section_id
        machine_list.append(machine)
        attrs.append("###".join([machine_id, att_id, source]))
        # attrs.append("###".join([machine_id, att_id]))
    return np.array(machine_list), np.array(attrs)

def prepare_data(split_dirs):
    file_list, label_list, source_list = get_data_list(split_dirs)
    machine_names, file_attrs = get_attributes(file_list, source_list)
    return file_list, label_list, source_list, machine_names, file_attrs

def get_dcase2023(train_split="train+add"):
    """ Args:
        dcase: can be dcase2020, 2021, 2022, and 2023
        train_split: train or train+add
    """
    import yaml
    with open("./data_config.yaml", "r") as ymlfile:
        data_config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    if train_split == "train+add":
        train_dirs = sorted(data_config["dcase2023_train_dirs"]) + sorted(data_config["dcase2023_add_dirs"])
        train_file_list, train_label_list, train_source_list, train_machine_names, train_file_attrs = prepare_data(train_dirs)
    elif train_split == "train":
        train_dirs = data_config["dcase2023_train_dirs"]
        train_file_list, train_label_list, train_source_list, train_machine_names, train_file_attrs = prepare_data(train_dirs)
    elif train_split == "add":
        train_dirs = data_config["dcase2023_add_dirs"]
        train_file_list, train_label_list, train_source_list, train_machine_names, train_file_attrs = prepare_data(train_dirs)
    valid_dirs = sorted(data_config["dcase2023_valid_dirs"])
    test_dirs = sorted(data_config["dcase2023_test_dirs"])
    valid_file_list, valid_label_list, valid_source_list, valid_machine_names, valid_file_attrs = prepare_data(valid_dirs)
    test_file_list, test_label_list, test_source_list, test_machine_names, test_file_attrs = get_test_data_list(test_dirs)

    datasets = {
        "train": {
            "file_list": train_file_list,
            "label_list": train_label_list,
            "source_list": train_source_list,
            "machine_names": train_machine_names,
            "file_attrs": train_file_attrs
        },
        "valid": {
            "file_list": valid_file_list,
            "label_list": valid_label_list,
            "source_list": valid_source_list,
            "machine_names": valid_machine_names,
            "file_attrs": valid_file_attrs
        },
        "test": {
            "file_list": test_file_list,
            "label_list": test_label_list,
            "source_list": test_source_list,
            "machine_names": test_machine_names,
            "file_attrs": test_file_attrs
        }
    }
    return datasets