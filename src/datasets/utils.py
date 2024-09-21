import numpy as np


def trainDataPct(dataset, pct=1, num_samples=None, machine_names:list=None, seed=42):
    # sample from each machine name 
    np.random.seed(seed)

    if machine_names is not None and type(machine_names) == str:
        machine_names = [machine_names]
        print(machine_names)

    if machine_names is None or machine_names[0] == "all":
        machine_names = np.unique(dataset['machine_names']).tolist()

    file_list = []
    label_list = []
    attr_list = []
    machine_name_list = []
    source_list = None
    if "source_list" in dataset.keys():
        source_list = []
    machine_list = dataset['machine_names']
    for mn in machine_names:
        machine_audio_files = np.array(dataset["file_list"])[machine_list == mn]
        machine_label_files = np.array(dataset["label_list"])[machine_list == mn]
        machine_file_attrs = np.array(dataset["file_attrs"])[machine_list == mn]
        if source_list is not None:
            machine_source_files = np.array(dataset["source_list"])[machine_list == mn]

        num_all = len(machine_audio_files)
        num_train = int(pct * num_all) if num_samples is None else num_samples
        id_list = np.random.choice(num_all, num_train, replace=False)
        
        sample_file_list = [machine_audio_files[i] for i in id_list]
        sample_label_list = [machine_label_files[i] for i in id_list]
        sample_attr_list = [machine_file_attrs[i] for i in id_list]
        if source_list is not None:
            sample_source_list = [machine_source_files[i] for i in id_list]
        # print(f'machine: {mn}, num_all: {num_all}, num_samples: {num_samples}, len(file_list): {len(sample_file_list)}')

        file_list.extend(sample_file_list)
        label_list.extend(sample_label_list)
        attr_list.extend(sample_attr_list)
        if source_list is not None:
            source_list.extend(sample_source_list)
        machine_name_list.extend([mn] * num_train)
    print(f"machine_names: {machine_names}, num_samples: {len(file_list)}")


    # print(f"total samples: {len(file_list)}")
    return {
        'file_list': np.array(file_list),
        'label_list': np.array(label_list),
        'file_attrs': np.array(attr_list),
        'source_list': np.array(source_list) if source_list is not None else None,
        'machine_names': np.array(machine_name_list)
    }