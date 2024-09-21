import pickle
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from scipy.stats import hmean
from src.datasets.prepare_dcase2020 import get_dcase2020
from src.datasets.audio_dataset import AudioDataset
from src.datasets.utils import trainDataPct

import torch 
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="ast", help='ast or beats')
parser.add_argument('--pretrained_model_dir', type=str, default="../transformer-ssl-asd/beats", help='path to saved models')
parser.add_argument('--train_pct', type=float, default=None, help='path to save results')
parser.add_argument('--num_samples', type=int, default=None, help='path to save results')
parser.add_argument('--split_seed', type=int, default=42, help='path to save results')
parser.add_argument('--dataset_name', type=str, default='dcase2020')
parser.add_argument('--eval_split', type=str, default="valid", help='Include test set')
parser.add_argument('--train_split', type=str, default='train', help='train, add, or train+add')
parser.add_argument('--top_k', type=int, default=1, help='Top k')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--temporal_pooling', action="store_true", default=False, help='Temporal pooling or not')
parser.add_argument('--spectral_pooling', action="store_true", default=False, help='Spectral pooling or not')
parser.add_argument('--audio_length', type=int, default=160000, help='Audio length')
parser.add_argument('--save_path', type=str, default='train_features', help='path to save results')

# ======== distance matrix ========
# original
# def calc_dist_matrix(x, y):
#     """Calculate Euclidean distance matrix with torch.tensor"""
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     x = x.unsqueeze(1).expand(n, m, d)
#     y = y.unsqueeze(0).expand(n, m, d)
#     dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
#     return dist_matrix

def calc_dist_matrix(x, y, bs=16):
    """Calculate Euclidean distance matrix with torch.tensor"""
    if bs > x.shape[0]:
        bs = x.shape[0]
    dist_matrices = torch.zeros(x.size(0), y.size(0))
    d = x.size(1)
    for i in range(y.shape[0]):
        m = 1
        dist_x_list = torch.zeros(x.size(0))
        x_batch = x.size(0)//bs + 1 if x.size(0) % bs != 0 else x.size(0)//bs
        for j in range(x_batch):
            x_i = x[j*bs:j*bs+bs]
            n = x_i.size(0)
            x_i = x_i.unsqueeze(1).expand(n, m, d) # 16, 1, 768
            y_i = y[i].unsqueeze(0).expand(n, m, d)
            dist_matrix = torch.sqrt(torch.pow(x_i - y_i, 2).sum(2))
            dist_x_list[j*bs:j*bs+bs] = dist_matrix.reshape(-1)
        dist_matrices[:, i] = dist_x_list
    return dist_matrices

# ======== evaluation function ========
def eval_score(gt_list, scores):
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, scores)
    img_roc_auc = roc_auc_score(gt_list, scores)
    pauc = roc_auc_score(gt_list, scores, max_fpr=0.1)
    precision, recall, thresholds = precision_recall_curve(gt_list, scores)
    f1_scores = (2 * precision * recall) / (precision + recall + np.finfo(float).eps)
    idx = np.argmax(f1_scores)
    return img_roc_auc, pauc, f1_scores[idx]

args = parser.parse_args()
model_name = args.model_name
dataset_name = args.dataset_name
pt_model_dir = args.pretrained_model_dir
train_split = args.train_split
split_seed = args.split_seed
top_k = args.top_k
batch_size = args.batch_size
temporal_pooling = args.temporal_pooling
spectral_pooling = args.spectral_pooling
if temporal_pooling:
    pooling_feature = "temporal"
elif spectral_pooling:
    pooling_feature = "spectral"
eval_split = args.eval_split
audio_length = args.audio_length

train_pct = args.train_pct # 0 - 1
num_samples = args.num_samples

if num_samples is not None:
    log_condition = f'GenRep_model_name{model_name}_topk{top_k}_trainsplit{train_split}_evalsplit{eval_split}_pooling{pooling_feature}_num_samples{num_samples}_seed{split_seed}'
elif train_pct is not None:
    log_condition = f'GenRep_model_name{model_name}_topk{top_k}_trainsplit{train_split}_evalsplit{eval_split}_pooling{pooling_feature}_trainpct{train_pct}_seed{split_seed}'
else:
    log_condition = f'GenRep_model_name{model_name}_topk{top_k}_trainsplit{train_split}_evalsplit{eval_split}_pooling{pooling_feature}'

save_dir = f"out/{dataset_name}_test/patch_diff_{model_name}/{log_condition}/log_{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)

if dataset_name == "dcase2020":
    datasets = get_dcase2020(train_split)

    # apply training percentage
    if train_pct is not None:
        datasets['train'] = trainDataPct(datasets['train'], pct=train_pct, machine_names="all", seed=split_seed)
    if num_samples is not None:
        datasets['train'] = trainDataPct(datasets['train'], pct=None, num_samples=num_samples, machine_names="all", seed=split_seed)

    train_data = datasets['train']
    test_data = datasets[eval_split]
    train_file_list=train_data["file_list"]
    train_label_list=train_data["label_list"]
    train_machine_list=train_data["machine_names"]
    train_source_list=None
    test_file_list=test_data["file_list"]
    test_label_list=test_data["label_list"]
    test_machine_list=test_data["machine_names"]
    test_source_list=None

train_class_ids = np.array(train_data['file_attrs'])
test_class_ids = np.array(test_data['file_attrs'])
        
# ======== model ========
print(f'model name used: {model_name}')
if model_name.startswith("beats"):
    import torch
    from beats.BEATs import BEATs, BEATsConfig
    if model_name == "beats":
        ckpt_path = f"{pt_model_dir}/BEATs_iter3_plus_AS2M.pt" 
    elif model_name == "beats_ft1":
        ckpt_path = f"{pt_model_dir}/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt"
    elif model_name == "beats_ft2":
        ckpt_path = f"{pt_model_dir}/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    elif model_name == "beats_iter3":
        ckpt_path = f"{pt_model_dir}/BEATs_iter3.pt"
    elif model_name == "beats_iter3_ft1":
        ckpt_path = f"{pt_model_dir}/BEATs_iter3_finetuned_on_AS2M_cpt1.pt"
    elif model_name == "beats_iter3_ft2":
        ckpt_path = f"{pt_model_dir}/BEATs_iter3_finetuned_on_AS2M_cpt2.pt"
    checkpoint = torch.load(ckpt_path)

    cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint['model'])

elif model_name == "beats_tokenizer":
    import torch
    from beats.Tokenizers import TokenizersConfig, Tokenizers

    # load the pre-trained checkpoints
    checkpoint = torch.load(f'{pt_model_dir}/Tokenizer_iter3_plus_AS2M.pt')

    cfg = TokenizersConfig(checkpoint['cfg'])
    model = Tokenizers(cfg)
    model.load_state_dict(checkpoint['model'])

print(sum([p.numel() for p in model.parameters() if p.requires_grad]))


device = "cuda"
model.to(device)
model.eval()

df_log = pd.DataFrame()
df_log_wo_norm = pd.DataFrame()

# ======= Evaluation =======
machine_names = np.unique(train_data["machine_names"])
print(f'machine names: {machine_names}')

for class_name in machine_names:

    train_dataset = AudioDataset(
        file_list=train_file_list,
        label_list=train_label_list,
        machine_list=train_machine_list,
        source_list=train_source_list,
        machine_name=class_name,
        audio_length=audio_length,
    )
    test_dataset = AudioDataset(
        file_list=test_file_list,
        label_list=test_label_list,
        machine_list=test_machine_list,
        source_list=test_source_list,
        machine_name=class_name,
        audio_length=audio_length,
    )

    # get class ids
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    train_machine_labels = train_class_ids[np.array(train_machine_list) == class_name]
    test_machine_labels = test_class_ids[np.array(test_machine_list) == class_name]
    le.fit(np.concatenate([train_machine_labels, test_machine_labels]))
    train_targets = le.transform(train_machine_labels)
    unique_targets = np.unique(train_targets)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=False)
    print(f'class name: {class_name}')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=False)

    # extract train set features
    print(f'len train dataset: {len(train_dataset)}')
    train_feature_layers = []
    train_all_lasts = []
    
    # save the embeddings if None
    save_path = args.save_path + f'_{model_name}'
    save_dir_path = f"temp_{dataset_name}_{train_split}"
    os.makedirs(os.path.join(save_path, save_dir_path), exist_ok=True)
    if train_pct is not None:
        if split_seed != 42:
            feature_name = f'train_{pooling_feature}_trainpct{train_pct}_seed{split_seed}_%s.pkl' % class_name
        else:
            feature_name = f'train_{pooling_feature}_trainpct{train_pct}_%s.pkl' % class_name
    elif num_samples is not None:
        if split_seed != 42:
            feature_name = f'train_{pooling_feature}_num_samples{num_samples}_seed{split_seed}_%s.pkl' % class_name
        else:
            feature_name = f'train_{pooling_feature}_num_samples{num_samples}_%s.pkl' % class_name
    else:
        feature_name = f'train_{pooling_feature}_%s.pkl' % class_name

    train_feature_filepath = os.path.join(save_path, save_dir_path,  feature_name)
    if not os.path.exists(train_feature_filepath):
        for batch in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            x = batch['input']
            x = x.to(device)
            bs = x.size(0)
            padding_mask = torch.zeros(x.shape).bool()
            padding_mask = padding_mask.to(device)

            # forward
            with torch.no_grad():
                _, _, attns = model.extract_features(x, padding_mask=padding_mask, need_weights=True, layer=11)

            # pick pooling technique
            if temporal_pooling:
                train_out_layers = [f_layer[0].transpose(0, 1).reshape(bs, 62, 8, -1).mean(1).cpu().unsqueeze(0) for f_layer in attns][1:] # skip the first pass: 12, bs, 496, 768 -> 12, bs, 768
            elif spectral_pooling:
                train_out_layers = [f_layer[0].transpose(0, 1).reshape(bs, 62, 8, -1).mean(2).cpu().unsqueeze(0) for f_layer in attns][1:] # skip the first pass: 12, bs, 496, 768 -> 12, bs, 768
            else:
                train_out_layers = [f_layer[0].transpose(0, 1).mean(1).cpu().unsqueeze(0) for f_layer in attns][1:] # skip the first pass: 12, bs, 496, 768 -> 12, bs, 768
            train_feature_layers.append(torch.cat(train_out_layers, 0))
            
        torch.cuda.empty_cache()
        train_feature_layers = torch.cat(train_feature_layers, 1).flatten(2)
        print(f"train_all_lasts.size(): {train_feature_layers.size()}")
        # save extracted feature
        print(f'save train set feature to: {train_feature_filepath}')
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_feature_layers, f)
    else:
        print('load train set feature from: %s' % train_feature_filepath)
        with open(train_feature_filepath, 'rb') as f:
            train_feature_layers = pickle.load(f)
        print(f"train_all_lasts.size(): {train_feature_layers.size()}")

    gt_list = []
    test_outputs = []
    test_feature_layers = []
    test_all_lasts = []

    save_dir_path = f"temp_{dataset_name}_{eval_split}"
    os.makedirs(os.path.join(save_path, save_dir_path), exist_ok=True)
    test_feature_filepath = os.path.join(save_path, save_dir_path, f'{eval_split}_{pooling_feature}_%s.pkl' % class_name)

    if not os.path.exists(test_feature_filepath):
        # extract test set features
        for batch in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            x = batch['input']
            x = x.to(device)
            bs = x.size(0)
            y = batch['label']
            gt_list.extend(y.cpu().detach().numpy())
            padding_mask = torch.zeros(x.shape).bool()
            padding_mask = padding_mask.to(device)

            # forward
            with torch.no_grad():
                _, _, attns = model.extract_features(x, padding_mask=padding_mask, need_weights=True, layer=11)

            # pick pooling technique
            if temporal_pooling:
                test_out_layers = [f_layer[0].transpose(0, 1).reshape(bs, 62, 8, -1).mean(1).cpu().unsqueeze(0) for f_layer in attns][1:] # skip the first pass: 12, bs, 496, 768 -> 12, bs, 768
            elif spectral_pooling:
                test_out_layers = [f_layer[0].transpose(0, 1).reshape(bs, 62, 8, -1).mean(2).cpu().unsqueeze(0) for f_layer in attns][1:] # skip the first pass: 12, bs, 496, 768 -> 12, bs, 768
            else:
                test_out_layers = [f_layer[0].transpose(0, 1).mean(1).cpu().unsqueeze(0) for f_layer in attns][1:] # skip the first pass: 12, bs, 496, 768 -> 12, bs, 768
            test_feature_layers.append(torch.cat(test_out_layers, 0))
            
        torch.cuda.empty_cache()

        test_feature_layers = torch.cat(test_feature_layers, 1).flatten(2)
        print(f"test_all_lasts.size(): {test_feature_layers.size()}")
        # save extracted feature
        print(f'save test set feature to: {test_feature_filepath}')
        with open(test_feature_filepath, 'wb') as f:
            pickle.dump(test_feature_layers, f)
    else:
        print('load test set feature from: %s' % test_feature_filepath)
        with open(test_feature_filepath, 'rb') as f:
            test_feature_layers = pickle.load(f)
        print(f"test_all_lasts.size(): {test_feature_layers.size()}")
        # get ground truth
        for batch in test_dataloader:
            y = batch['label']
            gt_list.extend(y.cpu().detach().numpy())

    root_path = "./"
    machine_list = np.array(test_data['machine_names'])
    gt_list = np.asarray(gt_list).reshape(-1)

    for i, (train_all_lasts, test_all_lasts) in enumerate(zip(train_feature_layers, test_feature_layers)):
        print(f' layer {i+1}')

        # the size of dist matrix is (len(test_all_lasts), len(train_all_lasts)) : 200, 1000 (990, 10)
        dist_matrix = calc_dist_matrix(torch.flatten(test_all_lasts, 1),
                                    torch.flatten(train_all_lasts, 1),
                                    bs=64)
        print(f'dist_matrix.size(): {dist_matrix.size()}')

        # ======= default =======
        # with norm
        score_dists = []       
        for cid in unique_targets:
            topk_values, topk_indices = torch.topk(dist_matrix[:, torch.from_numpy(train_targets) == cid], k=top_k, dim=1, largest=False)
            # print(f'topk_values.size(): {topk_values.size()}')
            scores = torch.mean(topk_values, 1, keepdim=True).cpu().detach().numpy()
            scores = (scores - scores.mean()) / scores.std()
            # print(f'scores.size(): {scores.shape}')
            score_dists.append(scores)
        scores = np.concatenate(score_dists, 1) 
        score_w_norm = np.min(scores, axis=1)
        auc_all, pauc_all, f1_all = eval_score(gt_list, score_w_norm)

        # without norm
        topk_values, topk_indices = torch.topk(dist_matrix, k=top_k, dim=1, largest=False)
        scores_wo_norm = torch.mean(topk_values, 1).cpu().detach().numpy()
        auc_all_wo_norm, pauc_all_wo_norm, f1_all_wo_norm = eval_score(gt_list, scores_wo_norm)

        print('AUC: %.4f, PAUC: %.4f, F1: %.4f' % (auc_all, pauc_all, f1_all))
        print('AUC_wo_norm: %.4f, PAUC_wo_norm: %.4f, F1_wo_norm: %.4f' % (auc_all_wo_norm, pauc_all_wo_norm, f1_all_wo_norm))
        
        df_log = pd.concat([df_log, pd.DataFrame({
            'layer': [i+1], 
            'machine': [class_name], 
            'auc': [auc_all], 
            'pauc': [pauc_all], 
            'f1': [f1_all],
            })]
        )
        df_log.to_csv(f'{save_dir}/result.csv', index=False)

        df_log_wo_norm = pd.concat([df_log_wo_norm, pd.DataFrame({
            'layer': [i+1], 
            'machine': [class_name], 
            'auc': [auc_all_wo_norm], 
            'pauc': [pauc_all_wo_norm], 
            'f1': [f1_all_wo_norm],
            })]
        )
        df_log_wo_norm.to_csv(f'{save_dir}/result_wo_norm.csv', index=False)


# ======== Summary with norm ========
df_test = df_log.reset_index(drop=True)
columns = df_test.columns
df_avg = df_test.groupby('layer')[columns[2:]].agg(hmean)
df_avg['oc'] = df_avg[['auc', 'pauc']].agg(np.mean, axis=1) # just in case
df_test['oc'] = df_test[['auc', 'pauc']].agg(np.mean, axis=1)
print(df_avg)

print(f'BEST LAYER WISE SCORE')
best_layer_wise = df_avg['oc'].argmax()+1 # index + 1 to get the layer
machine_layer_wise = df_test[df_test['layer'] == best_layer_wise][['machine', 'auc', 'pauc', 'oc']]
print(machine_layer_wise)

print(f' All layer wise score:')
final_best_layer_wise = pd.DataFrame(df_avg.iloc[df_avg['oc'].argmax()]).T
print(final_best_layer_wise)

print(f'BEST each machine')
best_each_machine = df_test.loc[df_test.groupby('machine')['oc'].idxmax()]
print(best_each_machine)

print(f'FINAL BEST SCORE')
final_best_score = pd.DataFrame(best_each_machine[['auc', 'pauc', 'f1', 'oc']].agg(np.mean)).T
print(final_best_score)

df_log.to_csv(f'{save_dir}/result.csv', index=False)

df_avg.to_csv(f'{save_dir}/df_avg.csv', index=False)
machine_layer_wise.to_csv(f'{save_dir}/machine_layer_wise.csv', index=False)
final_best_layer_wise.to_csv(f'{save_dir}/final_best_layer_wise.csv', index=False)
best_each_machine.to_csv(f'{save_dir}/best_each_machine.csv', index=False)
final_best_score.to_csv(f'{save_dir}/final_best_score.csv', index=False)

# ======== Summary without norm ========
df_test_wo_norm = df_log_wo_norm.reset_index(drop=True)
columns = df_test_wo_norm.columns
df_avg_wo_norm = df_test_wo_norm.groupby('layer')[columns[2:]].agg(hmean)
df_avg_wo_norm['oc'] = df_avg_wo_norm[['auc', 'pauc']].agg(np.mean, axis=1) # just in case
df_test_wo_norm['oc'] = df_test_wo_norm[['auc', 'pauc']].agg(np.mean, axis=1)
print(df_avg_wo_norm)

print(f'BEST LAYER WISE SCORE')
best_layer_wise_wo_norm = df_avg_wo_norm['oc'].argmax()+1 # index + 1 to get the layer
machine_layer_wise_wo_norm = df_test_wo_norm[df_test_wo_norm['layer'] == best_layer_wise_wo_norm][['machine', 'auc', 'pauc', 'oc']]
print(machine_layer_wise_wo_norm)

print(f' All layer wise score:')
final_best_layer_wise_wo_norm = pd.DataFrame(df_avg_wo_norm.iloc[df_avg_wo_norm['oc'].argmax()]).T
print(final_best_layer_wise_wo_norm)

print(f'BEST each machine')
best_each_machine_wo_norm = df_test_wo_norm.loc[df_test_wo_norm.groupby('machine')['oc'].idxmax()]
print(best_each_machine_wo_norm)

print(f'FINAL BEST SCORE')
final_best_score_wo_norm = pd.DataFrame(best_each_machine_wo_norm[['auc', 'pauc', 'f1', 'oc']].agg(np.mean)).T
print(final_best_score_wo_norm)

df_log_wo_norm.to_csv(f'{save_dir}/result_wo_norm.csv', index=False)

df_avg_wo_norm.to_csv(f'{save_dir}/df_avg_wo_norm.csv', index=False)
machine_layer_wise_wo_norm.to_csv(f'{save_dir}/machine_layer_wise_wo_norm.csv', index=False)
final_best_layer_wise_wo_norm.to_csv(f'{save_dir}/final_best_layer_wise_wo_norm.csv', index=False)
best_each_machine_wo_norm.to_csv(f'{save_dir}/best_each_machine_wo_norm.csv', index=False)
final_best_score_wo_norm.to_csv(f'{save_dir}/final_best_score_wo_norm.csv', index=False)

print(f"Done")