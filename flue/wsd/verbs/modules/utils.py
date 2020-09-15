# coding: utf8

import pandas as pd

from collections import defaultdict
from transformers import AutoModel, AutoTokenizer

def dump_logs(logs, outfile):


    with open(outfile, 'w') as f:
        for instance_id, log in logs.items():
            key,labels,first_label, pred, candidates, scores = log
            labels = '|'.join(list(labels))
            pred = list(pred)[0]
            candidates = " ".join(["{}/{}".format(x[0],x[1]) for x in zip(candidates,scores)])
            f.write("ID={} KEY={} LABELS={} FIRST_LABEL={} PRED={} CANDIDATES={}\n".format(instance_id, key, labels, first_label, pred, candidates))


def dump_preds(preds, outfile):

    with open(outfile, 'w') as f:
        for instance_id, pred in preds.items():
            pred = list(pred)[0] if pred else None
            f.write("{} {}\n".format(instance_id, pred))


def compute_scores(logs, exp_name=""):

    precision_ = []
    recall_ = []
    f_score_ = []
    sources_ = []
    pos_ = []
    exp_name_ = []

    # compute scores per source
    for source in logs['source'].unique():
        df_source = logs.loc[logs['source'] == source]
        n_total_all = len(df_source)
        n_pred_all = len(df_source.loc[df_source['pred'].notnull()])
        n_correct_all = sum(df_source['correct'])
        precision = n_correct_all / n_pred_all
        recall = n_correct_all / n_total_all
        f_score = 2 * (precision*recall) / (precision + recall)

        sources_.append(source)
        pos_.append('all_pos')
        precision_.append(precision)
        recall_.append(recall)
        f_score_.append(f_score)
        exp_name_.append(exp_name)

        # pos per source
        for pos in logs.loc[logs['source'] == source]['pos'].unique():
            df_source_pos = logs.loc[(logs['source'] == source) & (logs['pos'] == pos)]
            n_total_source_pos = len(df_source_pos)
            n_pred_source_pos = len(df_source_pos.loc[df_source_pos['pred'].notnull()])
            n_correct_source_pos = sum(df_source_pos['correct'])

            precision = n_correct_source_pos / n_pred_source_pos if (n_correct_source_pos and n_pred_source_pos) > 0 else 0.0
            recall = n_correct_source_pos / n_total_source_pos
            f_score = 2 * (precision*recall) / (precision + recall) if precision > 0  and recall > 0 else 0.0

            sources_.append(source)
            pos_.append(pos)
            precision_.append(precision)
            recall_.append(recall)
            f_score_.append(f_score)
            exp_name_.append(exp_name)


    # Compute pos
    if len(logs['source'].unique()) > 1:
        for pos in logs['pos'].unique():
            df_pos = logs.loc[logs['pos'] == pos]
            n_total_pos = len(df_pos)
            n_pred_pos = len(df_pos.loc[df_pos['pred'].notnull()])
            n_correct_pos = sum(df_pos['correct'])

            precision = n_correct_pos / n_pred_pos
            recall = n_correct_pos / n_total_pos
            f_score = 2 * (precision*recall) / (precision + recall)

            sources_.append("all_sources")
            pos_.append(pos)
            precision_.append(precision)
            f_score_.append(f_score)
            recall_.append(recall)
            exp_name_.append(exp_name)


        # Compute all
        n_total_all = len(logs)
        n_pred_all = len(logs.loc[logs['pred'].notnull()])
        n_correct_all = sum(logs['correct'])

        precision = n_correct_all / n_pred_all
        recall = n_correct_all / n_total_all
        f_score = 2 * (precision*recall) / (precision + recall)

        sources_.append("all_sources")
        pos_.append("all_pos")
        precision_.append(precision)
        f_score_.append(f_score)
        recall_.append(recall)
        exp_name_.append(exp_name)


    df = pd.DataFrame(data={'exp_name':exp_name,  'source':sources_, 'pos':pos_,
                        'f-score':f_score_, 'precision':precision_, 'recall':recall_})

    return df


def compute_logs(logs, exp_name):

    new_logs = defaultdict(list)

    new_logs['exp_name'] = [exp_name for x in logs]

    for id in logs:

        key, pos, source, labels, first_label, pred, candidates, scores = logs[id]

        new_logs['instance_id'].append(id)
        new_logs['key'].append(key)
        new_logs['pos'].append(pos)
        new_logs['source'].append(source)
        new_logs['labels'].append('|'.join(list(labels)))
        new_logs['first_label'].append(first_label)
        candidates = " ".join(["{}/{}".format(x[0],x[1]) for x in zip(candidates,scores)]) if candidates else None
        new_logs['candidates'].append(candidates)

        labels = set(list(labels))
        correct = 1 if (pred and pred.intersection(labels)) else 0

        pred = '|'.join([str(x) for x in list(pred)]) if pred else None
        new_logs['pred'].append(pred)

        new_logs['correct'].append(correct)



    df = pd.DataFrame(data=new_logs, index=None)


    return df


def load_model(model_path):

    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer
