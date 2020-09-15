# coding: utf8

import argparse
import os

from modules.dataset import WSDDatasetReader
from shutil import copyfile

usage = '''
        ===============================================================================================================================================
        This script prepares data to run and evaluate transformer models on the FrenchSemEval (FSE) dataset (Segonne et al. 2019).

            - inputs:
                * FSE dir : Directory containing the FSE dataset ==> http://www.llf.cnrs.fr/dataset/fse/
                * train dir (optional): A train directory which should contain .data.xml and .gold.key.txt files
                    (see Raganato's Evaluation Framework (Raganato et al. 2017) ==> http://lcl.uniroma1.it/wsdeval/ for format details)
                    This will replace the original training data provided by the FSE dataset.

            - outputs:
                * WSD data dir: A directory containing formated WSD data ready to be used by run_models.py and wsd_evaluation.py

        ===============================================================================================================================================
        '''


def get_data_paths(dirpath):
    """ Get data paths from FSE dir"""

    paths = {}

    for f in os.listdir(dirpath):
        if f.startswith("FSE"):
            data = "test"

        else:
            data = "train"

        if f.endswith("xml"):
            data_type = "xml"
        elif f.endswith("gold.key.txt"):
            data_type = "gold"
        else:
            continue

        paths["_".join((data, data_type, "path"))] = os.path.join(dirpath, f)

    return paths


def main(args):

    # get data paths
    paths = get_data_paths(args.data)

    # make output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    test_dirpath = args.output_dir + "test/"
    os.mkdir(test_dirpath)
    train_dirpath = args.output_dir + "train/"
    os.mkdir(train_dirpath)

    # copy FSE file to new test directory
    for p in paths:
        data = p.split('_')[0]
        data_prefix = paths[p].split('/')[0]
        filename = paths[p].split('/')[-1]
        copyfile(paths[p], args.output_dir + data + '/'  + filename)

    # if another train data dir is specified
    if args.train:
        for f in os.listdir(args.train):
            file_path = os.path.join(args.train, f)
            copyfile(file_path, args.output_dir + "train/" + filename)

    # get target keys from FSE and dump to main dir
    fse_dataset = WSDDatasetReader().read_from_data_dirs([test_dirpath])
    target_keys = list(fse_dataset.get_target_keys())
    with open(args.output_dir + 'targets', 'w') as f:
        for key in target_keys:
            f.write(key + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage)

    parser.add_argument('--data', help="path to FSE dir", metavar=('dirpath'))
    parser.add_argument('--train', help="path to another train directory than the one provided in FSE", metavar=('dirpath'))
    parser.add_argument('--output_dir', help="path to output dir", metavar=('dirpath'))

    args = parser.parse_args()

    main(args)
