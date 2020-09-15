# coding: utf8

import argparse
import os
import subprocess

from pudb import set_trace

usage = '''
            ====================================================================================================================================================
            This script runs a Transformer model (from Hugginface API -> https://huggingface.co/) on WSD data and performs an evaluation using a Knn Classifier.

            - inputs:
                * model : a pretrained or checkpoint model. It should be one of the Flaubert/Camembert/Bert
                transformer models from the Hugginface API.
                * wsd data: A WSD data dir with train and test directories. Each subdirectory
                should contain .data.xml and .gold.key.txt files (see
                Raganato's Evaluation Framework (Raganato et al. 2017) ==> http://lcl.uniroma1.it/wsdeval/ for format details).

            - outputs:
                * vectors: vectors output by the model for each instance of the wsd data (instance_id
                \\t vector per line)
                * predictions (optional): a file containing predictions of the WSD Knn classifier
                (instance_id \\t prediction per line)
                * logs (optional): a .csv file containing the output logs
                * score (optional): a .csv file containing the output score

            ====================================================================================================================================================

        '''


RUN_MODEL = os.path.join(os.getcwd(), "run_model.py")
WSD_EVAL = os.path.join(os.getcwd(), "wsd_evaluation.py")

def main():

    parser = argparse.ArgumentParser(usage)

    parser.add_argument('--exp_name', required=True, help="name of the experiments / will be use as name for outputs", metavar=('name'))
    parser.add_argument('--model', required=True, help="either path to model checkpoint or pretrained model from transformers API", metavar=('file'))
    parser.add_argument('--data', required=True, help="path to the directory output by prepare_data.py", metavar=('dir'))
    parser.add_argument('--padding', default=0, help="padding sequences")
    parser.add_argument('--batchsize', default=1, help="batchsize", )
    parser.add_argument('--device', default=-1, help="GPU device on which to run the model. If not specified the model will be run on CPU", metavar=('int'))
    parser.add_argument('--output', default=None, help="dirpath to output vectors. Default is data dir", metavar=('file'), required=True)
    parser.add_argument('--output_pred', default='', help="output fpath to dump predictions", metavar=('file'))
    parser.add_argument('--output_logs', default='', help="output fpath to dump logs", metavar=('file'), dest="output_logs")
    parser.add_argument('--output_score', default='', help="output fpath to dump score as .csv", metavar=('file'), dest="output_score")

    args = parser.parse_args()

    # paths
    train_path = os.path.join(args.data, 'train/')
    test_path = os.path.join(args.data, 'test/')
    targets_path = os.path.join(args.data, 'targets')
    train_vecs_output = os.path.join(args.output, '{}.train.vecs'.format(args.exp_name))
    test_vecs_output = os.path.join(args.output, '{}.test.vecs'.format(args.exp_name))

    print('\nEncode training data')
    subprocess.run(["python", RUN_MODEL,
                    "--model", args.model,
                    "--padding", args.padding,
                    "--device", args.device,
                    "--batchsize", args.batchsize,
                    "--data", train_path,
                    "--targets", targets_path,
                    "--output", train_vecs_output])

    print('Done! output vectors: {}\n'.format(train_vecs_output))


    print('Encode evaluation data')
    subprocess.run(["python", RUN_MODEL,
                    "--model", args.model,
                    "--padding", args.padding,
                    "--device", args.device,
                    "--batchsize", args.batchsize,
                    "--data", test_path,
                    "--targets", targets_path,
                    "--output", test_vecs_output])

    print('Done! output vectors: {}\n'.format(test_vecs_output))


    # WSD evaluation
    print("Perform evaluation")
    subprocess.run(["python", WSD_EVAL,
                    "--exp_name", args.exp_name,
                    "--train_data",train_path,
                    "--test_data",test_path,
                    "--train_vecs",train_vecs_output,
                    "--test_vecs",test_vecs_output,
                    "--target_pos", "V",
                    "--average",
                    "--output_logs",args.output_logs,
                    "--output_pred",args.output_pred,
                    "--output_score",args.output_score])



if __name__ == '__main__':

    main()
