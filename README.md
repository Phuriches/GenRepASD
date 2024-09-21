# GenRepASD
Pytorch implementation of Deep Generic Representations for Domain-Generalized Anomalous Sound Detection: https://arxiv.org/abs/2409.05035

## Setting up
1. Install the requirements `pip install -r requirements.txt`

2. Download the [DCASE2020T2](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds#download) and [DCASE2023T2](https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#download) datasets and place the datasets according to the given directory structure specified in `data_config.yaml`.

3. Download pre-trained weights of BEATs from https://github.com/microsoft/unilm/tree/master/beats and place them in a pre-trained directory.

## Domain-shift experiment on DCASE2023T2 Eval set.
Run GenRep using MemMixup with Ks=990 (best performance).
```
python run_genrep_dcase2023.py \
    --dataset_name dcase2023 \
    --model_name beats_ft1 \
    --pretrained_model_dir <path_to_pretrained_model> \
    --temporal_pooling \
    --n_mix_support 990 \
    --alpha 0.9 \
    --save_official
```

After run the above script, you can evaluate the performance using DCASE2023T2 official evaluator:
```
bash dcase2023_task2_evaluator/03_evaluation_eval_data.sh
```

## Low-shot experiment on DCASE2020T2 Dev set.
with 200-shot
```
python run_genrep_dcase2020.py \
    --dataset_name dcase2020 \
    --model_name beats_ft1 \
    --pretrained_model_dir <path_to_pretrained_model> \
    --temporal_pooling \
    --num_samples 200
```

## Acknowledgement
- We thanks the authors of [BEATs](https://arxiv.org/abs/2212.09058) for providing the pre-trained weights.
- We thanks [SPADE](https://github.com/byungjae89/SPADE-pytorch), [ssl4asd](https://github.com/wilkinghoff/ssl4asd), [STgram-MFN](https://github.com/liuyoude/STgram-MFN/tree/main), [DCASE2023](https://github.com/nttcslab/dcase2023_task2_evaluator) for providing the codebase and evaluation scripts.

## Citation
If you find this work useful, please consider citing:
```
@misc{saengthong2024deep,
    title={Deep Generic Representations for Domain-Generalized Anomalous Sound Detection},
    author={Phurich Saengthong and Takahiro Shinozaki},
    year={2024},
    eprint={2409.05035},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
}
```