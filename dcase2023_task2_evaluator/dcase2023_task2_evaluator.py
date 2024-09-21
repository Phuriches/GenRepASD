import os
import sys
import csv
from io import StringIO
import glob
import re
import numpy
import itertools
import scipy.stats
from sklearn import metrics
import pandas as pd
import argparse
from pathlib import Path
sys.path.append('./')
from tools.plot_anm_score import AnmScoreFigData
from distutils.util import strtobool

##############################################################################
# static values
##############################################################################
# Expected directory structure
# ./dcase2023_task2_evaluator/
#       ./teams "Directory containing team results"
#               ./<team name> "Directory containing anomaly score and decision result"
#       ./ground_truth_data "Directory where the true value is stored"
#       ./ground_truth_domain "Directory where the domain assignment is stored"
#       ./teams_result "Directory created after execution."

# directory path
GROUND_TRUTH_DATA_DIR = "ground_truth_data"
GROUND_TRUTH_DOMAIN_DIR = "ground_truth_domain"
GROUND_TRUTH_ATTRIBUTES_DIR = "ground_truth_attributes"

# Table columns
COLUMNS = ["AUC (all)", "AUC (source)", "AUC (target)", "pAUC", "precision (source)", "precision (target)",
    "recall (source)", "recall (target)", "F1 score (source)", "F1 score (target)"]
OFFICIAL_SCORE_COLUMNS = [
    "official score", "arithmetic mean", "harmonic mean (source)", "harmonic mean (target)",
    "ToyDrone AUC (source)", "ToyNscale AUC (source)", "ToyTank AUC (source)", "Vacuum AUC (source)", "bandsaw AUC (source)", "grinder AUC (source)", "shaker AUC (source)",
    "ToyDrone AUC (target)", "ToyNscale AUC (target)", "ToyTank AUC (target)", "Vacuum AUC (target)", "bandsaw AUC (target)", "grinder AUC (target)", "shaker AUC (target)",
]
MACHINE_TYPE_SCORE_COLMNS = [
    "AUC (source)", "AUC (target)", "pAUC",
    "precision (source)", "precision (target)", 
    "recall (source)", "recall (target)", 
    "F1 score (source)", "F1 score (target)"
]
SCORE_COLUMNS = [
    "official score", "arithmetic mean", "harmonic mean (source)", "harmonic mean (target)",
    "ToyDrone Score (source)", "ToyNscale Score (source)", "ToyTank Score (source)", "Vacuum Score (source)", "bandsaw Score (source)", "grinder Score (source)", "shaker Score (source)",
    "ToyDrone Score (target)", "ToyNscale Score (target)", "ToyTank Score (target)", "Vacuum Score (target)", "bandsaw Score (target)", "grinder Score (target)", "shaker Score (target)",
]
PAPER_OFFICIAL_SCORE_COLUMNS = ["h-mean", "a-mean", "ToyDrone", "ToyNscale", "ToyTank", "Vacuum", "bandsaw", "grinder", "shaker"]
SYSTEM_OFFICIAL_SCORE_INDEXES = ["AUC (source)", "AUC (target)", "pAUC (source, target)", "TOTAL score"]

# variables that do not change
MAX_FPR = 0.1
FILE_NAME_COL = 0
SCORE_COL = 1

##############################################################################
# common def
##############################################################################
# save csv
def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


# CSV format text to a list of rows decomposed as lists
def csv_text_to_list(csv_text):
    f = StringIO(csv_text)
    reader = csv.reader(f, delimiter=',')
    return [row for row in reader]


# extract machine types from ground truth
def get_machines(load_dir, ext=".csv"):
    query = os.path.abspath("{base}/ground_truth_*{ext}".format(base=load_dir,
                                                                ext=ext))
    machines = sorted(glob.glob(query))
    machines = [os.path.basename(f).split("_")[2] for f in machines]
    machines = sorted(list(set(machines)))
    return machines


# extract section id from anomaly score csv
def get_section_ids(target_dir, ext=".csv"):
    query = os.path.abspath("{target_dir}/ground_truth_*{ext}".format(target_dir=target_dir,
                                                                      ext=ext))
    paths = sorted(glob.glob(query))
    ids = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('section_[0-9][0-9]', ext_id) for ext_id in paths]
    ))))
    return ids


# read score from csv
def read_score(file_path, decision=False, attribute=None):
    with open(file_path) as score_file:
        score_list = list(csv.reader(score_file))
    if attribute:
        score_list = [[attribute[score[FILE_NAME_COL]], score[SCORE_COL]] if score[FILE_NAME_COL] in attribute else score for score in score_list]
    score_data = [float(score[SCORE_COL]) for score in sorted(score_list)]
    if decision:
        score_data = [int(s) for s in score_data]

    return numpy.array(score_data)


# Jackknife resampling - https://en.wikipedia.org/wiki/Jackknife_resampling
def jackknife_estimate(fn, var_list):
    # See section IV on page 6 on https://hal.inria.fr/hal-02067935/file/mesaros_TASLP19.pdf
    # Reference: A. Mesaros et al., "Sound Event Detection in the DCASE 2017 Challenge," in IEEE/ACM Transactions on Audio,
    #            Speech, and Language Processing, vol. 27, no. 6, pp. 992-1006, June 2019, doi: 10.1109/TASLP.2019.2907016.

    def removed_i(var_list, remove_i):
        return [v[[i for i in range(len(v)) if i != remove_i]] for v in var_list]
    var_list = [numpy.array(v) for v in var_list]
    N = len(var_list[0])
    # (1)
    theta_hat = fn(*var_list)
    # (2)
    thetai_hats = [fn(*removed_i(var_list, i)) for i in range(N)]
    # (3)
    theta_hat_mean = numpy.mean(thetai_hats)
    # (4)
    thetai_tildes = [N * theta_hat - (N - 1) * thetai_hat for thetai_hat in thetai_hats]
    # (5)
    theta_hat_jack = numpy.mean(thetai_tildes)
    # (6)
    sigma_hat_jack = numpy.sqrt(numpy.sum([(thi - theta_hat_mean)**2 for thi in thetai_hats]) / (N * (N-1)))
    # (7) - CI only
    confidence = 0.95
    dof = N - 1
    t_crit = numpy.abs(scipy.stats.t.ppf((1 - confidence) / 2, dof))
    ci95_jack = t_crit * sigma_hat_jack

    return theta_hat_jack, ci95_jack


# [main] output the result from the specified directory and machine type
def output_result(target_dir, machines, section_ids, result_dir, additional_result_dir, seed, tag, out_all=False):
    print(target_dir)
    csv_lines = []
    all_y_preds, all_y_trues = [], []
    all_df = pd.DataFrame(columns=["section"] + COLUMNS)
    official_score_df = pd.DataFrame(
        index=[os.path.basename(target_dir)],
        columns=OFFICIAL_SCORE_COLUMNS,
    )
    index_df = pd.DataFrame({
        "System":[os.path.basename(target_dir)] * len(SYSTEM_OFFICIAL_SCORE_INDEXES),
        "metric":SYSTEM_OFFICIAL_SCORE_INDEXES
    })
    multi_index = pd.MultiIndex.from_frame(index_df)
    paper_official_score_df = pd.DataFrame(
        index=multi_index,
        columns=PAPER_OFFICIAL_SCORE_COLUMNS
    )
    auc_df = {}
    index_list = []
    score_df = {}
    for section_id in section_ids:
        auc_df[section_id] = pd.DataFrame(
            index=[os.path.basename(target_dir)],
            columns=OFFICIAL_SCORE_COLUMNS,
        )
        index_list.append(section_id.split("_", 1)[1])
    
        score_df[section_id] = pd.DataFrame(
            index=[os.path.basename(target_dir)],
            columns=SCORE_COLUMNS,
        )

    y_pred_domain_list = []
    y_true_domain_list = []
    y_true_list = []

    for machine_idx, target_machine in enumerate(machines):
        anm_score_figdata = AnmScoreFigData()
        machine_type_score_df = pd.DataFrame(
            index=index_list + ["arithmetic mean", "harmonic mean"],
            columns=MACHINE_TYPE_SCORE_COLMNS
        )
        print("[{idx}/{total}] machine type : {target_machine}".format(target_machine=target_machine,
                                                                       idx=machine_idx+1,
                                                                       total=len(machines)))
        csv_lines.append([target_machine])
        df = pd.DataFrame(columns=["section"] + COLUMNS).set_index('section')

        y_pred_domain_id_list = []
        y_true_domain_id_list = []
        y_true_id_list = []
        for section_id in section_ids:
            sidx = section_id.split("_", 1)[1]
            print(section_id)

            # Load results and ground truth
            anomaly_score_path = "{dir}/anomaly_score_{machine}_{section}_test.csv".format(dir=target_dir,
                                                                                                machine=target_machine,
                                                                                                section=section_id)
            decision_result_path = "{dir}/decision_result_{machine}_{section}_test.csv".format(dir=target_dir,
                                                                                                    machine=target_machine,
                                                                                                    section=section_id)
            ground_truth_path = "{dir}/ground_truth_{machine}_{section}_test.csv".format(dir=GROUND_TRUTH_DATA_DIR,
                                                                                                machine=target_machine,
                                                                                                section=section_id)
            gt_domain_path = "{dir}/ground_truth_{machine}_{section}_test.csv".format(dir=GROUND_TRUTH_DOMAIN_DIR,
                                                                                                machine=target_machine,
                                                                                                section=section_id)
            gt_attribute_path = "{dir}/ground_truth_{machine}_{section}_test.csv".format(dir=GROUND_TRUTH_ATTRIBUTES_DIR,
                                                                                                machine=target_machine,
                                                                                                section=section_id)
            if not os.path.exists(anomaly_score_path) or \
                not os.path.exists(decision_result_path):
                # Load DCASE2023 baseline results
                anomaly_score_path = "{dir}/anomaly_score_DCASE2023T2{machine}_{section}_test_seed{seed}{tag}_Eval.csv".format(
                    dir=target_dir,
                    machine=target_machine,
                    section=section_id,
                    seed=seed,
                    tag=tag)
                decision_result_path = "{dir}/decision_result_DCASE2023T2{machine}_{section}_test_seed{seed}{tag}_Eval.csv".format(
                    dir=target_dir,
                    machine=target_machine,
                    section=section_id,
                    seed=seed,
                    tag=tag)
            
            if not os.path.exists(anomaly_score_path) or \
                not os.path.exists(decision_result_path) or \
                not os.path.exists(ground_truth_path) or \
                not os.path.exists(gt_domain_path) or \
                not os.path.exists(gt_attribute_path):
                print(f"not have the all score : {target_dir}")
                return 0, None, None, None, None
            with open(gt_attribute_path) as attribute_file:
                attribute_list = list(csv.reader(attribute_file))
                attribute_dict = {f"{attribute[1]}.wav":attribute[0] for attribute in attribute_list}
            y_pred_all = read_score(os.path.abspath(anomaly_score_path), attribute=attribute_dict)
            y_true_all = read_score(os.path.abspath(ground_truth_path))
            y_domain = read_score(os.path.abspath(gt_domain_path))
            decision_result_data_all = read_score(os.path.abspath(decision_result_path), decision=True, attribute=attribute_dict)
            all_y_preds.extend(y_pred_all)
            all_y_trues.extend(y_true_all)

            # Evaluate for whole section
            df.loc[sidx, 'AUC (all)'] = metrics.roc_auc_score(y_true_all, y_pred_all)
            df.loc[sidx, 'pAUC'] = metrics.roc_auc_score(y_true_all, y_pred_all, max_fpr=MAX_FPR)
            
            # set score for each machine ID
            machine_type_score_df.loc[sidx, 'pAUC'] = df.loc[sidx, 'pAUC']

            for label_idx, label in enumerate(['normal', 'anomaly']):
                print(f"{label} : {len(y_true_all[y_true_all == label_idx])} files")
                
            # Evaluate for each domain
            for domain in ['source', 'target']:
                domain_idx = {'source': 0, 'target': 1}[domain]
                print(f"{domain} : {len(y_domain[y_domain == domain_idx])} files")

                # Filter by domain
                y_pred_auc = y_pred_all[(y_domain == domain_idx) | (y_true_all != 0)]
                y_true_auc = y_true_all[(y_domain == domain_idx) | (y_true_all != 0)]
                y_pred = y_pred_all[y_domain == domain_idx]
                y_true = y_true_all[y_domain == domain_idx]
                y_pred_nml = y_pred_all[(y_domain == domain_idx) & (y_true_all == 0)]
                decision_result_data = decision_result_data_all[y_domain == domain_idx]

                if len(y_true) != len(y_pred) or len(y_true) != len(decision_result_data):
                    print("number of reference elements:", len(y_true))
                    print("anomaly score element count:", len(y_pred), " path:", anomaly_score_path)
                    print("decision data element count:", len(decision_result_data), " path:", decision_result_path)
                    print("some elements are missing")
                    return -1, None, None, None, None

                # calc result
                df.loc[sidx, f'AUC ({domain})'] = metrics.roc_auc_score(y_true_auc, y_pred_auc)
                tn, fp, fn, tp = metrics.confusion_matrix(y_true, decision_result_data).ravel()
                prec = tp / numpy.maximum(tp + fp, sys.float_info.epsilon)
                recall = tp / numpy.maximum(tp + fn, sys.float_info.epsilon)
                df.loc[sidx, f'precision ({domain})'] = prec
                df.loc[sidx, f'recall ({domain})'] = recall
                df.loc[sidx, f'F1 score ({domain})'] = 2.0 * prec * recall / numpy.maximum(prec + recall, sys.float_info.epsilon)

                # set score for each machine ID
                machine_type_score_df.loc[sidx, f'AUC ({domain})'] = df.loc[sidx, f'AUC ({domain})']
                # machine_type_score_df.loc[sidx, f'pAUC ({domain})'] = metrics.roc_auc_score(y_true_auc, y_pred_auc, max_fpr=MAX_FPR)
                machine_type_score_df.loc[sidx, f'precision ({domain})'] = df.loc[sidx, f'precision ({domain})']
                machine_type_score_df.loc[sidx, f'recall ({domain})'] = df.loc[sidx, f'recall ({domain})']
                machine_type_score_df.loc[sidx, f'F1 score ({domain})'] = df.loc[sidx, f'F1 score ({domain})']

                anm_score_figdata.append_figdata(anm_score_figdata.anm_score_to_figdata(
                    scores=[[t, p] for t, p in zip(y_true, y_pred)],
                    title=f"{section_id}_{domain}_AUC{df.loc[sidx, f'AUC ({domain})']}"
                ))

                score_df[section_id].at[os.path.basename(target_dir), f"{target_machine} Score ({domain})"] = y_pred_nml.mean()



        y_pred_domain_list.append(sum(y_pred_domain_id_list, []))
        y_true_domain_list.append(sum(y_true_domain_id_list, []))
        y_true_list.append(sum(y_true_id_list, []))
        csv_lines.extend(csv_text_to_list(df.to_csv()))
        all_df = pd.concat([all_df, df])

        csv_lines.append(["", "", "AUC", "pAUC", "precision", "recall", "F1 score"])
        performance = numpy.array([
            df[["AUC (source)"]].values[:, 0].tolist() + df[["AUC (target)"]].values[:, 0].tolist(),
            df[["pAUC"]].values[:, 0].tolist() + df[["pAUC"]].values[:, 0].tolist(),
            df[["precision (source)"]].values[:, 0].tolist() + df[["precision (target)"]].values[:, 0].tolist(),
            df[["recall (source)"]].values[:, 0].tolist() + df[["recall (target)"]].values[:, 0].tolist(),
            df[["F1 score (source)"]].values[:, 0].tolist() + df[["F1 score (target)"]].values[:, 0].tolist(),
        ], dtype=float)
        amean_performance = numpy.mean(performance, axis=1)
        csv_lines.append(["arithmetic mean", ""] + list(amean_performance))
        hmean_performance = scipy.stats.hmean(numpy.maximum(performance, sys.float_info.epsilon), axis=1)
        csv_lines.append(["harmonic mean", ""] + list(hmean_performance))
        hmean_performance = scipy.stats.hmean(numpy.maximum(performance[:, :performance.shape[1]//2], sys.float_info.epsilon), axis=1)
        csv_lines.append(["source harmonic mean", ""] + list(hmean_performance))
        hmean_performance = scipy.stats.hmean(numpy.maximum(performance[:, performance.shape[1]//2:], sys.float_info.epsilon), axis=1)
        csv_lines.append(["target harmonic mean", ""] + list(hmean_performance))
        csv_lines.append([])

        official_score_df.at[os.path.basename(target_dir), f"{target_machine} AUC (source)"] = numpy.mean(df[["AUC (source)"]].values[:, 0].tolist())
        official_score_df.at[os.path.basename(target_dir), f"{target_machine} AUC (target)"] = numpy.mean(df[["AUC (target)"]].values[:, 0].tolist())

        paper_official_score_df.at[(os.path.basename(target_dir), "AUC (source)"), target_machine] = numpy.mean(df[["AUC (source)"]].values[:, 0].tolist())
        paper_official_score_df.at[(os.path.basename(target_dir), "AUC (target)"), target_machine] = numpy.mean(df[["AUC (target)"]].values[:, 0].tolist())
        paper_official_score_df.at[(os.path.basename(target_dir), "pAUC (source, target)"), target_machine] = numpy.mean(df[["pAUC"]].values[:, 0].tolist())

        # set score for each machine ID
        for dataset_type in ["source", "target"]:
            # pauc_list = [x for x in machine_type_score_df[f'pAUC ({dataset_type})'].tolist() if numpy.isnan(x) == False]
            machine_type_score_df.loc["arithmetic mean", f"AUC ({dataset_type})"] = numpy.mean(df[[f"AUC ({dataset_type})"]].values[:, 0].tolist())
            machine_type_score_df.loc["arithmetic mean", f"precision ({dataset_type})"] = numpy.mean(df[[f"precision ({dataset_type})"]].values[:, 0].tolist())
            machine_type_score_df.loc["arithmetic mean", f"recall ({dataset_type})"] = numpy.mean(df[[f"recall ({dataset_type})"]].values[:, 0].tolist())
            machine_type_score_df.loc["arithmetic mean", f"F1 score ({dataset_type})"] = numpy.mean(df[[f"F1 score ({dataset_type})"]].values[:, 0].tolist())
            machine_type_score_df.loc["harmonic mean", f"AUC ({dataset_type})"] = scipy.stats.hmean(df[[f"AUC ({dataset_type})"]].values[:, 0].tolist())
            machine_type_score_df.loc["harmonic mean", f"precision ({dataset_type})"] = scipy.stats.hmean(df[[f"precision ({dataset_type})"]].values[:, 0].tolist())
            machine_type_score_df.loc["harmonic mean", f"recall ({dataset_type})"] = scipy.stats.hmean(df[[f"recall ({dataset_type})"]].values[:, 0].tolist())
            machine_type_score_df.loc["harmonic mean", f"F1 score ({dataset_type})"] = scipy.stats.hmean(df[[f"F1 score ({dataset_type})"]].values[:, 0].tolist())
        machine_type_score_df.loc["arithmetic mean", "pAUC"] = numpy.mean(df[["pAUC"]].values[:, 0].tolist())
        machine_type_score_df.loc["harmonic mean", "pAUC"] = scipy.stats.hmean(df[["pAUC"]].values[:, 0].tolist())

        for section_id, auc_source, auc_target in zip(section_ids, df[["AUC (source)"]].values[:, 0].tolist(), df[["AUC (target)"]].values[:, 0].tolist()):
            auc_df[section_id].at[os.path.basename(target_dir), f"{target_machine} AUC (source)"] = auc_source
            auc_df[section_id].at[os.path.basename(target_dir), f"{target_machine} AUC (target)"] = auc_target

        if out_all:
            os.makedirs(f"{additional_result_dir}/{os.path.basename(target_dir)}", exist_ok=True)
            anm_score_figdata.show_fig(
                title=f"{os.path.basename(target_dir)}_{target_machine}_{section_id}_anm_score",
                export_dir=f"{additional_result_dir}/{os.path.basename(target_dir)}/"
            )

    csv_lines.append(["", "", "AUC", "pAUC", "precision", "recall", "F1 score"])
    performance_over_all = numpy.array([
        all_df[["AUC (source)"]].values[:, 0].tolist() + all_df[["AUC (target)"]].values[:, 0].tolist(),
        all_df[["pAUC"]].values[:, 0].tolist() + all_df[["pAUC"]].values[:, 0].tolist(),
        all_df[["precision (source)"]].values[:, 0].tolist() + all_df[["precision (target)"]].values[:, 0].tolist(),
        all_df[["recall (source)"]].values[:, 0].tolist() + all_df[["recall (target)"]].values[:, 0].tolist(),
        all_df[["F1 score (source)"]].values[:, 0].tolist() + all_df[["F1 score (target)"]].values[:, 0].tolist(),
    ], dtype=float)
    # calculate averages for AUCs and pAUCs
    ## a-mean (all)
    amean_performance = numpy.mean(performance_over_all, axis=1)
    csv_lines.append(["arithmetic mean over all machine types, sections, and domains", ""] + list(amean_performance))
    official_score_df.at[os.path.basename(target_dir), "arithmetic mean"] = float(amean_performance[0])
    paper_official_score_df.at[(os.path.basename(target_dir),"pAUC (source, target)"), "a-mean"] = float(amean_performance[1])
    ## a-mean (source)
    n_source = len(all_df[["AUC (source)"]].values[:, 0])
    amean_performance = numpy.mean(performance_over_all[:, :n_source], axis=1)
    paper_official_score_df.at[(os.path.basename(target_dir), "AUC (source)"), "a-mean"] = float(amean_performance[0])
    ## a-mean (target)
    amean_performance = numpy.mean(performance_over_all[:, n_source:], axis=1)
    paper_official_score_df.at[(os.path.basename(target_dir), "AUC (target)"), "a-mean"] = float(amean_performance[0])

    ## h-mean (all)
    hmean_performance = scipy.stats.hmean(numpy.maximum(performance_over_all, sys.float_info.epsilon), axis=1)
    csv_lines.append(["harmonic mean over all machine types, sections, and domains", ""] + list(hmean_performance))
    paper_official_score_df.at[(os.path.basename(target_dir), "pAUC (source, target)"), "h-mean"] = float(hmean_performance[1])
    ## h-mean (source)
    hmean_performance = scipy.stats.hmean(numpy.maximum(performance_over_all[:, :n_source], sys.float_info.epsilon), axis=1)
    official_score_df.at[os.path.basename(target_dir), "harmonic mean (source)"] = float(hmean_performance[0])
    paper_official_score_df.at[(os.path.basename(target_dir), "AUC (source)"), "h-mean"] = float(hmean_performance[0])
    csv_lines.append(["source harmonic mean over all machine types, sections, and domains", ""] + list(hmean_performance))
    ## h-mean (target)
    hmean_performance = scipy.stats.hmean(numpy.maximum(performance_over_all[:, n_source:], sys.float_info.epsilon), axis=1)
    official_score_df.at[os.path.basename(target_dir), "harmonic mean (target)"] = float(hmean_performance[0])
    paper_official_score_df.at[(os.path.basename(target_dir), "AUC (target)"), "h-mean"] = float(hmean_performance[0])
    csv_lines.append(["target harmonic mean over all machine types, sections, and domains", ""] + list(hmean_performance))
    csv_lines.append([])

    all_perf = numpy.array([
        all_df[["AUC (source)"]].values[:, 0].tolist() + all_df[["AUC (target)"]].values[:, 0].tolist()
        + all_df[["pAUC"]].values[:, 0].tolist(),
    ], dtype=float)
    official_score = scipy.stats.hmean(numpy.maximum(all_perf, sys.float_info.epsilon), axis=None)
    csv_lines.append(["official score", "", str(official_score)])
    official_score_df.at[os.path.basename(target_dir), "official score"] = float(official_score)
    paper_official_score_df.at[(os.path.basename(target_dir), "TOTAL score"), "h-mean"] = float(official_score)
    paper_official_score_df.at[(os.path.basename(target_dir), "TOTAL score"), "a-mean"] = float(numpy.mean(all_perf))

    auc_jack, auc_ci95 = jackknife_estimate(fn=metrics.roc_auc_score, var_list=[all_y_trues, all_y_preds])
    pauc_jack, p_auc_ci95 = jackknife_estimate(fn=lambda a,b: metrics.roc_auc_score(a, b, max_fpr=MAX_FPR), var_list=[all_y_trues, all_y_preds])
    print('########## CI95', auc_ci95, p_auc_ci95, 'official score', official_score, 'auc/pauc jack', auc_jack, pauc_jack)
    csv_lines.append(["official score ci95", "", str(numpy.mean([auc_ci95, p_auc_ci95]))])
    csv_lines.append([])

    # output results
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = "{result_dir}/{target_dir}_result.csv".format(result_dir=result_dir,
                                                                     target_dir=os.path.basename(target_dir))
    print("results -> {}".format(result_file_path))
    save_csv(save_file_path=result_file_path, save_data=csv_lines)

    result_file_path = "{result_dir}/{target_dir}/official_score.csv".format(result_dir=additional_result_dir,
                                                                             target_dir=os.path.basename(target_dir))
    if out_all:
        print("official score -> {}".format(result_file_path))
        official_score_df.to_csv(result_file_path)

    return 0, official_score_df, auc_df, score_df, paper_official_score_df

##############################################################################
# main
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams_root_dir", type=str, default="./teams",
                        help="Directory containing team results."+
                        "./<team name> 'Directory containing anomaly score and decision result'")
    parser.add_argument("--result_dir", type=str, default="./teams_result",
                        help="./teams_result 'Directory created after execution.'")
    parser.add_argument("--additional_result_dir", type=str, default="./teams_additional_result",
                        help="./teams_result 'Directory created after execution.'")
    parser.add_argument("--dir_depth", type=int, default=2,
                        help="what depth to search '--teams_root_dir' using glob."+
                        "Example, if --dir_depth=2, then 'glob.glob(<teams_root_dir>/*/*)'")
    parser.add_argument("--out_all", type=strtobool, default=False,
                        help="if 'out_true=True`, export supplemental data.")
    parser.add_argument("--seed", type=int, default=13711)
    parser.add_argument('-tag','--model_name_suffix',type=str, default='_id(0_)', 
                        help='Add a word to file name')

    args = parser.parse_args()
    
    teams_root_dir = args.teams_root_dir
    result_dir = args.result_dir
    additional_result_dir = args.additional_result_dir
    out_all = args.out_all

    Path(result_dir).mkdir(parents=True, exist_ok=True)
    if out_all:
        Path(additional_result_dir).mkdir(parents=True, exist_ok=True)

    machine_types = get_machines(load_dir=GROUND_TRUTH_DATA_DIR)
    section_ids = get_section_ids(target_dir=GROUND_TRUTH_DATA_DIR)

    team_dirs = list(numpy.sort(glob.glob(teams_root_dir + "/*" * args.dir_depth)))
    # if os.path.isdir(result_dir):
    #     print("the result directory exist")
    #     sys.exit(-1)
    teams_official_score_df = pd.DataFrame(
        columns=OFFICIAL_SCORE_COLUMNS,
    )

    multi_index = pd.MultiIndex.from_product([[],[]], names=["System", "metric"])
    teams_paper_official_score_df = pd.DataFrame(
        index=multi_index,
        columns=PAPER_OFFICIAL_SCORE_COLUMNS
    )

    teams_auc_df = {} # AUC for each section id
    teams_score_df = {} # anomary score for each section id
    for section_id in section_ids:
        teams_auc_df[section_id] = pd.DataFrame(
            columns=OFFICIAL_SCORE_COLUMNS,
        )
        teams_score_df[section_id] = pd.DataFrame(
            columns=SCORE_COLUMNS,
        )

    for idx, team_dir in enumerate(team_dirs):
        print("[{idx}/{total}] team name : {team_dir}".format(team_dir=os.path.basename(team_dir),
                                                              idx=idx+1,
                                                              total=len(team_dirs)))
        if os.path.isdir(team_dir):
            normal_end_flag, official_score_df, auc_df, score_df, paper_official_score_df = output_result(
                team_dir,
                machine_types,
                section_ids,
                result_dir=result_dir,
                additional_result_dir=additional_result_dir,
                seed=args.seed,
                tag=args.model_name_suffix,
                out_all=out_all,
            )
            if normal_end_flag == -1:
                print("abnormal termination")
                sys.exit(-1)

            if type(official_score_df) == pd.core.frame.DataFrame:
                teams_official_score_df = pd.concat([teams_official_score_df, official_score_df])
            if type(paper_official_score_df) == pd.core.frame.DataFrame:
                teams_paper_official_score_df = pd.concat([teams_paper_official_score_df, paper_official_score_df])

            # concat all teams auc
            if auc_df is not None:
                for section_id in section_ids:
                    if type(auc_df[section_id]) == pd.core.frame.DataFrame:
                        teams_auc_df[section_id] = pd.concat([teams_auc_df[section_id], auc_df[section_id]])

            # concat all teams score
            if score_df is not None:
                for section_id in section_ids:
                    if type(score_df[section_id]) == pd.core.frame.DataFrame:
                        teams_score_df[section_id] = pd.concat([teams_score_df[section_id], score_df[section_id]])

        else:
            print("{} is not directory.".format(team_dir))
    result_file_path = "{result_dir}/{target_dir}_official_score.csv".format(result_dir=additional_result_dir,
                                                                             target_dir=os.path.basename(teams_root_dir))
    if out_all:
        print(f"teams result -> {result_file_path}")
        teams_official_score_df.to_csv(result_file_path)

    # export paper score
    result_file_path = "{result_dir}/{target_dir}_official_score_paper".format(result_dir=additional_result_dir,
                                                                               target_dir=os.path.basename(teams_root_dir))
    if out_all:
        print(f"teams result -> {result_file_path}.csv")
        teams_paper_official_score_df.to_csv(f"{result_file_path}.csv")
        # print(f"teams result -> {result_file_path}_raund.csv")
        # teams_paper_official_score_df.astype(numpy.float64).to_csv(f"{result_file_path}_raund.csv", float_format='%.4f')
        # print(f"teams result -> {result_file_path}.txt")
        # teams_paper_official_score_df.to_latex(f"{result_file_path}.txt")

    for section_id in section_ids:
        result_file_path = "{result_dir}/{target_dir}_{section_id}_auc.csv".format(result_dir=additional_result_dir,
                                                                                                target_dir=os.path.basename(teams_root_dir),
                                                                                                section_id=section_id)
        if out_all:
            print(f"AUC section {section_id} -> {result_file_path}")
            teams_auc_df[section_id].to_csv(result_file_path)

        result_file_path = "{result_dir}/{target_dir}_{section_id}_score.csv".format(result_dir=additional_result_dir,
                                                                                                  target_dir=os.path.basename(teams_root_dir),
                                                                                                  section_id=section_id)
        if out_all:
            print(f"AUC section {section_id} -> {result_file_path}")
            teams_score_df[section_id].to_csv(result_file_path)

