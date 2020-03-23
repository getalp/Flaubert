import os, sys
import torch
import shutil
import argparse
from transformers import XLMModel, XLMTokenizer

from convert_xlm_original_pytorch_checkpoint_to_pytorch import convert_xlm_checkpoint_to_pytorch

def rename_parameters(filename):
    state_dict = torch.load(filename)
    update = {k.replace("module.", "") : v for k, v in state_dict.items()}
    torch.save(update, filename)


def main(args):
    """
    Converts XLM checkpoint to a model loadable from transformers

        python convert_to_transformers.py <path to BERT-{Base,Large}-lower-case> <output dir>

    ex:
        python convert_to_transformers.py BERT-Base-lower-case/ flaubert_base_lower

    The output dir will be created.

    The conversion is based on the following script:
        https://github.com/huggingface/transformers/blob/master/transformers/convert_xlm_original_pytorch_checkpoint_to_pytorch.py
    The ouput of the conversion is then corrected (name of parameters in state_dict).

    Upon loading the model with the transformers package, we need to make sure that no error occurs
    (output_loading_info=True returns the log which should not contain errors)
    and that the do_lowercase_and_remove_accent attribute of the tokenizer is set to False
    (otherwise the BPE tokenization will ignore diacritics and be completely wrong).

    >>> model, log = XLMModel.from_pretrained(modelname, output_loading_info=True)
    >>> tokenizer = XLMTokenizer.from_pretrained(modelname, do_lower_case=False)
    >>> tokenizer.do_lowercase_and_remove_accent = False

    """
    outputdir = os.path.expanduser(args.outputdir)
    inputdir = os.path.expanduser(args.inputdir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # shutil.copyfile(f"{args.inputdir}/BPE/codes", f"{args.outputdir}/merges.txt") 

    # convert_xlm_checkpoint_to_pytorch(f"{args.inputdir}/best-valid_fr_mlm_ppl.pth", args.outputdir)

    shutil.copyfile("{}/codes".format(inputdir), "{}/merges.txt".format(outputdir))
    convert_xlm_checkpoint_to_pytorch("{}/best-valid_fr_mlm_ppl.pth".format(inputdir), outputdir)

    rename_parameters("{}/pytorch_model.bin".format(outputdir))

    test_model(outputdir)


def test_model(modelname):
    model, log = XLMModel.from_pretrained(modelname, output_loading_info=True)
    tokenizer = XLMTokenizer.from_pretrained(modelname, do_lower_case=False)

    # this line is important: by default, XLMTokenizer removes diacritics, even with do_lower_case=False flag
    tokenizer.do_lowercase_and_remove_accent = False
    print("Dictionary values must be empty lists:")
    print(log)

    # print(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=main.__doc__)
    parser.add_argument("--inputdir",
                        type = str,
                        help = "Path to FlauBERT directory (BERT-Base-lower-case or BERT-Large-lower-case")
    parser.add_argument("--outputdir",
                        type = str,
                        help = "Output directory")

    args = parser.parse_args()

    main(args)
