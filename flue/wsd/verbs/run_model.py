# coding: utf8

import argparse

from modules.wsd_encoder import TransformerWSDEncoder
from modules.dataset import WSDDatasetReader
from modules import utils
from WSD_evaluation.modules.dataset import WSDDatasetReader

from pudb import set_trace

from tqdm import tqdm

usage = '''
            ===================================================================================================================
            This script run a transformer model (from hugginface: https://huggingface.co/) on WSD data.

            - input:
                * wsd directory : must contain .data.xml and .gold.key.txt files (see
                Raganato's Evaluation Framework (Raganato et al. 2017) ==> http://lcl.uniroma1.it/wsdeval/ for format details)
                * target (optional) : a file containaining target word types (format: WORD.POS per
                line)
                * model : a pretrained (or checkpoint path) model from hugginface's transformers
                (must be a Flaubert/Camembert/Bert model)

            - output:
                * vectors : contextual vectors output by the model for every instance of the wsd
                dataset (format: instance_id \\t vector)

            ====================================================================================================================
        '''


def make_batches(dataset, encoder, batchsize, padding):
    """ Make batches based from dataset instances
        :param dataset: wsd dataset
        :type dataset: WSDDataset
        :param encoder: encoder to obtain contextual vector
        :type encoder: WSDEncoder
        :param batchsize: size of batches
        :type batchsize: int
        :param padding: padding length
        :type padding: int
    """

    batches = []
    batch = []

    for sent_id, instances in tqdm(dataset.sent_id2instances.items(), "Making batches", leave=False):

        # if sentence with no instances continue
        if len(instances) == 0:
            continue

        sent = dataset.sent_id2sent[sent_id]

        sent_lst = sent
        tok_ids, att_mask, span = encoder.encode(sent_lst, padding=padding)
        length = len(tok_ids)

        # if length of tokenizer encodings > padding
        if length > padding and padding > 0:
            batch_ = [((tok_ids, att_mask,span), instances)]
            inputs_batch, instances_batch = list(zip(*batch_))
            inputs = encoder.collate_fn(inputs_batch)
            wsd_data = list(zip(*[(i.id,i.tok_id,  i.key.replace(".","<COMMA>"), i.first_label, i.labels, j) for j,x in enumerate(instances_batch) for i in x]))
            batches.append((inputs, wsd_data))
            continue

        batch.append(((tok_ids, att_mask,span), instances))

        if len(batch) == batchsize:
            inputs_batch, instances_batch = list(zip(*batch))
            inputs = encoder.collate_fn(inputs_batch)
            wsd_data = list(zip(*[(i.id,i.tok_id,  i.key.replace(".","<COMMA>"), i.first_label, i.labels, j) for j,x in enumerate(instances_batch) for i in x]))
            batches.append((inputs, wsd_data))
            batch = []

    # last batch
    if len(batch) > 0:
        inputs_batch, instances_batch = list(zip(*batch))
        inputs = encoder.collate_fn(inputs_batch)
        wsd_data = list(zip(*[(i.id,i.tok_id,  i.key.replace(".","<COMMA>"), i.first_label, i.labels, j) for j,x in enumerate(instances_batch) for i in x]))
        batches.append((inputs, wsd_data))

    return batches

def read_targets(path):

    return [x.rstrip('\n') for x in open(path).readlines()]

def main():

    parser = argparse.ArgumentParser(usage)

    parser.add_argument('--data', help="dirpath containing .data.xml and .gold.key.txt files")
    parser.add_argument('--targets', help="fpath to target keys", metavar=('file'))
    parser.add_argument('--output', help="fpath to output vectors", metavar=('file'))
    parser.add_argument('--model', help="fpath to model checkpoint or name of the pretrained model", metavar=('file'))
    parser.add_argument('--padding', type=int, default=0, help="input padding", metavar=('int'))
    parser.add_argument('--batchsize', type=int, default=1, help="size of batch", metavar=('int'))
    parser.add_argument('--device', default=-1, type=int, help="to run model on GPU, -1 for CPU", metavar=('int'))

    args = parser.parse_args()

    # Loads model and tokenizer
    model, tokenizer = utils.load_model(args.model)

    # puts model on specified device
    if args.device == -1:
        model.cpu()
        device = "cpu"
    else:
        device = int(args.device)
        model.cuda(int(args.device))

    # puts model on eval mode
    model.eval()

    # Loads wsd encoder with model and tokenizer
    wsd_encoder = TransformerWSDEncoder(model, tokenizer)

    # Loads wsd dataset
    wsd_rdr = WSDDatasetReader()
    targets = read_targets(args.targets) if args.targets else None # target keys from test dataset
    wsd_dataset = wsd_rdr.read_from_data_dirs([args.data], target_keys=targets)

    # making batches
    bs = args.batchsize
    padding = args.padding
    batches = make_batches(wsd_dataset, wsd_encoder, bs, padding)

    # Run model on batches and output context vector representations for targeted occurrences
    with open(args.output, 'w') as f:
        for batch in tqdm(batches, "Running model on data", leave=False):
            inputs, wsd_data = batch
            inputs = [x.to(device) for x in inputs]
            output = wsd_encoder(inputs)

            instance_ids = wsd_data[0]
            tgt = wsd_data[1]  # token indices
            batch_idx = wsd_data[-1]  # batch indices

            target_output = output[batch_idx, tgt].tolist()

            for instance_id, vec in zip(instance_ids, target_output):
                vec = ' '.join([str(x) for x in vec])
                f.write("{} {}\n".format(instance_id, vec))

            del inputs, output, target_output


if __name__ == '__main__':

    main()
