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
    for file in file_list:
        label_list.append(0 if 'normal' in file else 1)
    return label_list

def get_data_list(file_dirs):
    file_list = []
    for fd in file_dirs:
        files = get_filename_list(fd, ext="wav")
        file_list.extend(files)
        # print(f'len of files: {fd.split("/")[-2]} | {len(files)}')
    label_list = get_meta_list(file_list)
    return file_list, label_list

def get_attributes(file_list):
    attrs = []
    machines = []
    for idx, ext_id in enumerate(file_list):
        machine = ext_id.split('/')[-3]
        file = ext_id.split('/')[-1]
        section_id = re.findall('id_[0-9][0-9]', file)[0].split('_')[-1]
        machine_id = machine + '_' + section_id
        attrs.append(machine_id)
        machines.append(machine)
    return np.array(attrs), np.array(machines)

def prepare_data(split_dirs):
    file_list, label_list = get_data_list(split_dirs)
    file_attrs, machine_names = get_attributes(file_list)
    return file_list, label_list, file_attrs, machine_names

def prepare_test_data():
    df_test = pd.read_csv("./src/datasets/preprocessed_eval_data_list.csv")
    root_path = "../database/dcase2020t2/eval_data/{machine}/test"
    file_list = df_test['file_name'].tolist()
    machine_list = df_test['machine'].tolist()
    label_list = df_test['label'].tolist()
    def get_attributes():
        files = []
        attrs = []
        for file_name, machine_name in zip(file_list, machine_list):
            file_name = os.path.join(root_path.format(machine=machine_name), file_name)
            section_id = re.findall('id_[0-9][0-9]', file_name)[0].split('_')[-1]
            machine_id = machine_name + '_' + section_id
            files.append(file_name)
            attrs.append(machine_id)
        return files, np.array(attrs)
    file_list, file_attrs = get_attributes()
    machine_names = np.array(machine_list)
    return file_list, label_list, file_attrs, machine_names

def get_dcase2020(train_split="train+add"):
    """ Args:
        dcase: can be dcase2020, 2021, 2022, and 2023
        train_split: train or train+add
    """
    import yaml
    with open("./data_config.yaml", "r") as ymlfile:
        data_config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    if train_split == "train+add":
        train_dirs = data_config["dcase2020_train_dirs"] + data_config["dcase2020_add_dirs"]
        train_file_list, train_label_list, train_file_attrs, train_machine_names = prepare_data(train_dirs)
    elif train_split == "train":
        train_dirs = data_config["dcase2020_train_dirs"]
        train_file_list, train_label_list, train_file_attrs, train_machine_names = prepare_data(train_dirs)
    elif train_split == "add":
        train_dirs = data_config["dcase2020_add_dirs"]
        train_file_list, train_label_list, train_file_attrs, train_machine_names = prepare_data(train_dirs)
    valid_dirs = data_config["dcase2020_valid_dirs"]
    # test_dirs = data_config["dcase2020_test_dirs"]
    valid_file_list, valid_label_list, valid_file_attrs, valid_machine_names = prepare_data(valid_dirs)

    test_file_list, test_label_list, test_file_attrs, test_machine_names = prepare_test_data()

    datasets = {
        "train": {
            "file_list": train_file_list,
            "label_list": train_label_list,
            "file_attrs": train_file_attrs,
            "machine_names": train_machine_names
        },
        "valid": {
            "file_list": valid_file_list,
            "label_list": valid_label_list,
            "file_attrs": valid_file_attrs,
            "machine_names": valid_machine_names
        },
        "test": {
            "file_list": test_file_list,
            "label_list": test_label_list,
            "file_attrs": test_file_attrs,
            "machine_names": test_machine_names
        }
    }
    return datasets