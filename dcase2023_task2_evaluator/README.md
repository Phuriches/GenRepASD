# dcase2023\_task2\_evaluator
The **dcase2023\_task2\_evaluator** is a script for calculating the AUC, pAUC, precision, recall, and F1 scores from the anomaly score list for the [evaluation dataset](https://zenodo.org/record/7860847) in DCASE 2023 Challenge Task 2 "First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring."

[https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring](https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)

## Description

The **dcase2023\_task2\_evaluator** consists of two scripts:

- `dcase2023_task2_evaluator.py`
    - This script outputs the AUC and pAUC scores by using:
      - Ground truth of the normal and anomaly labels
      - Anomaly scores for each wave file listed in the csv file for each machine type, section, and domain
      - Detection results for each wave file listed in the csv file for each machine type, section, and domain
- `03_evaluation_eval_data.sh`
    - This script execute `dcase2023_task2_evaluator.py`.

## Usage
### 1. Clone repository
Clone this repository from Github.

### 2. Prepare data
- Anomaly scores
    - Generate csv files `anomaly_score_<machine_type>_section_<section_index>_test.csv` and `decision_result_<machine_type>_section_<section_index>_test.csv` or `anomaly_score_DCASE2023T2<machine_type>_section_<section>_test_seed<seed><tag>_Eval.csv` and `decision_result_DCASE2023T2<machine_type>_section_<section>_test_seed<seed><tag>_Eval.csv` by using a system for the [evaluation dataset](https://zenodo.org/record/7860847). (The format information is described [here](https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#submission).)
- Rename the directory containing the csv files to a team name
- Move the directory into `./teams/`

### 3. Check directory structure
- ./dcase2023\_task2\_evaluator
    - /dcase2023\_task2\_evaluator.py
    - /03\_evaluation\_eval\_data.sh
    - /ground\_truth\_attributes
        - ground\_truth\_bandsaw\_section\_00\_test.csv
        - ground\_truth\_grinder\_section\_00\_test.csv
        - ...
    - /ground\_truth\_data
        - ground\_truth\_bandsaw\_section\_00\_test.csv
        - ground\_truth\_grinder\section\_00\_test.csv
        - ...
    - /ground\_truth\_domain
        - ground\_truth\_bandsaw\_section\_00\_test.csv
        - ground\_truth\_grinder\_section\_00\_test.csv
        - ...
    - /teams
        - /\<team\_name\_1\>
            - /\<system\_name\_1\>
                - anomaly\_score\_bandsaw\_section\_00\_test.csv
                - anomaly\_score\_grinder\_section\_00\_test.csv
                - ...
                - decision\_result\_ToyTank\_section\_00\_test.csv
                - decision\_result\_Vacuum\_section\_00\_test.csv
            - /\<system\_name\_2\>
                - anomaly\_score\_DCASE2023T2bandsaw\_section\_00\_test\_seed\<--seed\>\<--tag\>\_Eval.csv
                - anomaly\_score\_DCASE2023T2grinder\_section\_00\_test\_seed\<--seed\>\<--tag\>\_Eval.csv
                - ...
                - decision\_result\_DCASE2023T2ToyTank\_section\_00\_test\_seed\<--seed\>\<--tag\>\_Eval.csv
                - decision\_result\_DCASE2023T2Vacuum\_section\_00\_test\_seed\<--seed\>\<--tag\>\_Eval.csv
        - /\<team\_name\_2\>
            - /\<system\_name\_3\>
                - anomaly\_score\_bandsaw\_section\_00\_test.csv
                - anomaly\_score\_grinder\_section\_00\test.csv
                - ...
                - decision\_result\_ToyTank\_section\_00\_test.csv
                - decision\_result\_Vacuum\_section\_00\_test.csv
        - ...
    - /teams\_result
        - \<system\_name\_1\>\_result.csv
        - \<system\_name\_2\>\_result.csv
        - \<system\_name\_3\>_result.csv
        - ...
    - /teams\_additional\_result \*`out_all==True`
        - teams\_official\_score.csv
        - teams\_official\_score\_paper.csv
        - teams\_section\_00\_auc.csv
        - teams\_section\_00\_score.csv
        - /\<system\_name\_1\>
            - official\_score.csv
            - \<system\_name\_1\>\_bandsaw\_section\_00\_anm\_score.png
            - ...
            - \<system\_name\_1\>\_Vacuum\_section\_00\_anm\_score.png
        - /\<system\_name\_2\>
            - official\_score.csv
            - \<system\_name\_2\>\_bandsaw\_section\_00\_anm\_score.png
            - ...
            - \<system\_name\_2\>\_Vacuum\_section\_00\_anm\_score.png
        - /\<system\_name\_3\>
            - official\_score.csv
            - \<system\_name\_3\>\_bandsaw\_section\_00\_anm\_score.png
            - ...
            - \<system\_name\_3\>\_Vacuum\_section\_00\_anm\_score.png
        - ...
    - /tools
        - plot\_anm\_score.py
        - test\_plots.py
    - /README.md


### 4. Change parameters
The parameters are defined in the script `dcase2023_task2_evaluator.py` as follows.
- **MAX\_FPR**
    - The FPR threshold for pAUC : default 0.1
- **--result\_dir**
    - The output directory : default `./teams_result/`
- **--teams\_root\_dir**
    - Directory containing team results. : default `./teams/`
- **--dir\_depth**
    - What depth to search `--teams_root_dir` using glob. : default `2`
    - If --dir\_depth=2, then `glob.glob(<teams_root_dir>/*/*)`
- **--tag**
    - File name tag. : default `_id(0_)`
    - If using filename is DCASE2023 baseline style, change parameters as necessary. 
- **--seed**
    - Seed used during train. : default `13711`
    - If using filename is DCASE2023 baseline style, change parameters as necessary.
- **--out\_all**
    - If this parameter is `True`, export supplemental data. : default `False`
- **--additional\_result\_dir**
    - The output additional results directory. : default `./teams_additional_result/`
    - Used when `--out_all==True`.

### 5. Run script
Run the script `dcase2023_task2_evaluator.py`
```
$ python dcase2023_task2_evaluator.py
```
or
```
$ bash 03_evaluation_eval_data.sh
```
The script `dcase2023_task2_evaluator.py` calculates the AUC, pAUC, precision, recall, and F1 scores for each machine type, section, and domain and output the calculated scores into the csv files (`<system_name_1>_result.csv`, `<system_name_2>_result.csv`, ...) in **--result\_dir** (default: `./teams_result/`).
If **--out\_all=True**, each team results are then aggregated into a csv file (`teams_official_score.csv`, `teams_official_score_paper.csv`) in **--additional\_result\_dir** (default: `./teams_additional_result`).

### 6. Check results
You can check the AUC, pAUC, precision, recall, and F1 scores in the `<system_name_N>_result.csv` in **--result\_dir**.
The AUC, pAUC, precision, recall, and F1 scores for each machine type, section, and domain are listed as follows:

`<section_name_N>_result.csv`
```
ToyDrone
section,AUC (all),AUC (source),AUC (target),pAUC,precision (source),precision (target),recall (source),recall (target),F1 score (source),F1 score (target)
00,0.6789,0.7968000000000001,0.561,0.5368421052631579,0.7560975609756098,0.5079365079365079,0.62,0.64,0.6813186813186813,0.5663716814159292
,,AUC,pAUC,precision,recall,F1 score
arithmetic mean,,0.6789000000000001,0.5368421052631579,0.6320170344560588,0.63,0.6238451813673053
harmonic mean,,0.6584250994255415,0.5368421052631579,0.6076569678407351,0.6298412698412698,0.6185502727981294
source harmonic mean,,0.7968000000000001,0.5368421052631579,0.7560975609756098,0.62,0.6813186813186813
target harmonic mean,,0.561,0.5368421052631579,0.5079365079365079,0.64,0.5663716814159292

...

shaker
section,AUC (all),AUC (source),AUC (target),pAUC,precision (source),precision (target),recall (source),recall (target),F1 score (source),F1 score (target)
00,0.6253625362536254,0.69428983714698,0.5604118104118104,0.5491654428600755,0.578125,0.4936708860759494,0.6981132075471698,0.8478260869565217,0.6324786324786325,0.624
,,AUC,pAUC,precision,recall,F1 score
arithmetic mean,,0.6273508237793952,0.5491654428600755,0.5358979430379747,0.7729696472518457,0.6282393162393163
harmonic mean,,0.6202083584461523,0.5491654428600755,0.5325705849787784,0.765720350225524,0.628210709621245
source harmonic mean,,0.69428983714698,0.5491654428600755,0.578125,0.6981132075471698,0.6324786324786325
target harmonic mean,,0.5604118104118104,0.5491654428600755,0.4936708860759494,0.8478260869565217,0.624

...

,,AUC,pAUC,precision,recall,F1 score
"arithmetic mean over all machine types, sections, and domains",,0.6403576632460674,0.5535708782745333,0.5364448553682957,0.7308992232966801,0.6006994950456381
"harmonic mean over all machine types, sections, and domains",,0.6152996272906976,0.5517419647782388,0.5032829900980702,0.7137886024875123,0.590331192259057
"source harmonic mean over all machine types, sections, and domains",,0.7423494890244248,0.5517419647782388,0.5356629533316296,0.660146438268587,0.5914253446046336
"target harmonic mean over all machine types, sections, and domains",,0.5253826834789426,0.5517419647782388,0.4745945156180243,0.7769195103318602,0.5892410808585007

official score,,0.5925469043549957
official score ci95,,1.531898879903843e-05
```

Aggregated results for each baseline are listed as follows:

```_seed13711_official_score_paper.csv
System,metric,h-mean,a-mean,bandsaw,grinder,shaker,ToyDrone,ToyNscale,ToyTank,Vacuum
baseline_MAHALA,AUC (source),0.7871141310430937,0.7922050973063174,0.836434267021059,0.7409411378914575,0.8476602762317048,0.8130000000000001,0.6678000000000002,0.8011999999999999,0.8383999999999999
baseline_MAHALA,AUC (target),0.53090862919051,0.5643372095607136,0.5885844748858448,0.50682261208577,0.6299533799533801,0.4622,0.409,0.4532,0.9006
baseline_MAHALA,"pAUC (source, target)",0.5682103280865886,0.5727080310911363,0.5753594967896497,0.5955291246149972,0.6233307541280444,0.5142105263157895,0.5089473684210526,0.5384210526315789,0.653157894736842
baseline_MAHALA,TOTAL score,0.6105082186925268,0.6430834459860557,,,,,,,
baseline_MSE,AUC (source),0.7423494890244248,0.7482088346609131,0.6667348190554078,0.706837186424004,0.69428983714698,0.7968000000000001,0.77,0.7212000000000001,0.8816
baseline_MSE,AUC (target),0.5253826834789426,0.5325064918312215,0.48287671232876717,0.5516569200779727,0.5604118104118104,0.561,0.4716,0.6468,0.45320000000000005
baseline_MSE,"pAUC (source, target)",0.5517419647782388,0.5535708782745333,0.5091087658743451,0.5846166760294185,0.5491654428600755,0.5368421052631579,0.5178947368421053,0.5826315789473684,0.5947368421052631
baseline_MSE,TOTAL score,0.5925469043549957,0.6114287349222227,,,,,,,

```

If you use this system, please cite all the following four papers:

+ Kota Dohi, Keisuke Imoto, Noboru Harada, Daisuke Niizumi, Yuma Koizumi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, and Yohei Kawaguchi, "Description and Discussion on DCASE 2023 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring," in arXiv-eprints: 2305.07828, 2023. [URL](https://arxiv.org/abs/2305.07828)
+ Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, Shoichiro Saito, "ToyADMOS2: Another Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection under Domain Shift Conditions," in Proc. DCASE 2022 Workshop, 2022. [URL](https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Harada_6.pdf)
+ Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi, "MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task," in Proc. DCASE 2022 Workshop, 2022. [URL](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Dohi_62.pdf)
+ Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, "First-Shot Anomaly Detection for Machine Condition Monitoring: A Domain Generalization Baseline," in arXiv e-prints: 2303.00455, 2023. [URL](https://arxiv.org/abs/2303.00455)