# coding: utf8

import argparse
import os

from modules.dataset import WSDDatasetReader
from modules.classifier import WSDKnnClassifier
from modules import utils

from pudb import set_trace


usage = '''
            ====================================================================================
            This script performs WSD evaluation using a Knn classifier.

            - input:
                * train data: wsd data dir (should contain:  .data.xml + .gold.key.txt)
                * train vectors: train vectors (format: id \\t vec)
                * test data: wsd data dir (should contain: .data.xml + .gold.key.txt)
                * test vectors : test vectors (format: id \\t vec)

            -output:
                * predictions (optional): a file containing predictions of the WSD Knn classifier
                (instance_id \\t prediction per line)
                * logs (optional): a .csv file containing the output logs
                * score (optional): a .csv file containing the output score

            ====================================================================================
        '''


def read_data(infile):
    """ Reads file containing vectors for WSD instances and return data as dict.

        :param infile: path to file containing vectors
        :type infile: str

        :return: dict -> {instance_id: vec}
        :rtype: dict
    """

    data = {}

    with open(infile) as f:
        line = f.readline()

        while line:
            line = line.rstrip('\n').split()
            instance_id = line[0]
            vec = [float(x) for x in line[1:]]
            data[instance_id] = vec
            line = f.readline()
    return data


def add_vecs_to_dataset(dataset, vecs):
    """ Add vectors to wsd dataset's instances

        :param dataset: wsd dataset
        :type dataset: WSDDataset
        :param vecs: context vectors as dict {instance_id: vector}
        :type vecs: dict

        :return: wsd dataset but with vectors added to instances
        :rtype: WSDDataset
    """

    for instance_id, vec in vecs.items():
        if instance_id in dataset.id2instance:
            dataset.id2instance[instance_id].context_vec = vec

    return dataset


def main():

    parser.add_argument('--train_data', help="dirpath to wsd traindata", required=True, metavar=('file'))
    parser.add_argument('--train_vecs', help="fpath to train vectors", required=True, metavar=('file'))
    parser.add_argument('--targets', help="fpaht to targets", default=None, metavar=('file'))
    parser.add_argument('--test_data', help="dirpath to wsd test data", required=True, metavar=('file'))
    parser.add_argument('--test_vecs', help="fpath to test vectors", required=True, metavar=('file'))
    parser.add_argument('--exp_name', help="name of the experiment", required=True, metavar=('name'))
    parser.add_argument('--target_pos', help="to evaluation only on target PoS", default=None)
    parser.add_argument('--mfs_backoff', help="to use mfs backoff", action="store_true")
    parser.add_argument('--mfs_files', help="path to file containing the mfs backoff predictions", metavar=('file'), default=None, nargs='+')
    parser.add_argument('--output_logs', help="fpath to output logs", metavar=('file'), default=None)
    parser.add_argument('--output_pred', help="fpath to ouptut predictions", metavar=('file'), default=None)
    parser.add_argument('--output_score', help="fpath to output score", metavar=('file'), default=None)

    knn_group = parser.add_mutually_exclusive_group()
    knn_group.add_argument('--average', help="to average vectors of the same label", action='store_true', default=False)
    knn_group.add_argument('--k', help="the number of k most similar vectors to compute prediction. if > 1, predictions is based on the majority label", default=1,type=int, metavar=('int'))

    args = parser.parse_args()

    # train and test data paths
    test_dirpath = args.test_data
    train_dirpath = args.train_data

    # Reads WSD dataset
    wsd_reader = WSDDatasetReader()
    test_dataset = wsd_reader.read_from_data_dirs([test_dirpath], target_pos=args.target_pos)
    targets = wsd_reader.read_target_keys(args.targets) if args.targets else None
    if not targets:
        targets = test_dataset.get_target_keys()
    train_dataset = wsd_reader.read_from_data_dirs([train_dirpath], target_keys=targets)

    # reads vectors from train data
    train_vecs_fpath = args.train_vecs
    train_vecs = read_data(train_vecs_fpath)

    # reads vectors from test data
    test_vecs_fpath = args.test_vecs
    test_vecs = read_data(test_vecs_fpath)

    # Adds vec to wsd dataset's instancs
    train_dataset = add_vecs_to_dataset(train_dataset, train_vecs)
    test_dataset = add_vecs_to_dataset(test_dataset, test_vecs)
    test_dataset = add_vecs_to_dataset(test_dataset, test_vecs)

    # Instantiates wsd knn classifier
    wsd_classifier = WSDKnnClassifier(average=args.average, k=args.k,
                                      mfs_backoff=args.mfs_backoff,
                                      mfs_files=args.mfs_files)

    wsd_classifier.fit(train_dataset)  # fit knn on train data
    pred, logs = wsd_classifier.predict(test_dataset)  # predict on test

    # Computes logs
    df_logs = utils.compute_logs(logs, exp_name=args.exp_name)

    # Dumps logs if specificied
    if args.output_logs:
        if os.path.exists(args.output_logs):
            df_logs.to_csv(args.output_logs, mode='a', header=False, index=False)
        else:
            df_logs.to_csv(args.output_logs, index=False)

    # Dumps predictions if specified
    if args.output_pred:
        utils.dump_preds(pred, args.output_pred)

    # Dumpes score if specificied
    df_scores = utils.compute_scores(df_logs, args.exp_name)
    print(df_scores)  # print score on stdout
    if args.output_score:
        if os.path.exists(args.output_score):
            df_scores.to_csv(args.output_score, mode='a', header=False, index=False)
        else:
            df_scores.to_csv(args.output_score, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage)

    main()
