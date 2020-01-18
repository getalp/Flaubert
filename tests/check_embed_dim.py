import os, sys
import torch
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.expanduser('~/Flaubert'))
from xlm.utils import AttrDict
from xlm.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from xlm.model.transformer import TransformerModel

sys.path.append(os.path.expanduser('~/transformers/src'))
from transformers import FlaubertModel, FlaubertTokenizer

# Below is one way to bpe-ize sentences
codes = os.path.expanduser('~/pretrained_models/Flaubert/flaubert_base_cased_xlm/codes')
fastbpe =  os.path.expanduser('~/Flaubert/tools/fastBPE/fast')

def to_bpe(sentences):
    # write sentences to tmp file
    with open('/tmp/sentences.bpe', 'w') as fwrite:
        for sent in sentences:
            fwrite.write(sent + '\n')
    
    # apply bpe to tmp file
    os.system('%s applybpe /tmp/sentences.bpe /tmp/sentences %s' % (fastbpe, codes))
    
    # load bpe-ized sentences
    sentences_bpe = []
    with open('/tmp/sentences.bpe') as f:
        for line in f:
            sentences_bpe.append(line.rstrip())
    
    return sentences_bpe

def main():
    model_path = os.path.expanduser('~/pretrained_models/Flaubert/flaubert_base_cased_xlm/flaubert_base_normal.pth')
    # model_path = os.path.expanduser('~/pretrained_models/Flaubert/flaubert_base_uncased_xlm/best-valid_fr_mlm_ppl.pth')
    
    # reload model
    reloaded = torch.load(model_path)
    state_dict = reloaded['model']
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    params = AttrDict(reloaded['params'])
    print("Supported languages: %s" % ", ".join(params.lang2id.keys()))

    # build dictionary / update parameters
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    params.n_words = len(dico)
    params.bos_index = dico.index(BOS_WORD)
    params.eos_index = dico.index(EOS_WORD)
    params.pad_index = dico.index(PAD_WORD)
    params.unk_index = dico.index(UNK_WORD)
    params.mask_index = dico.index(MASK_WORD)

    # build model / reload weights
    model = TransformerModel(params, dico, True, True)
    model.eval()
    model.load_state_dict(state_dict)

    # Below is one way to bpe-ize sentences
    codes = "" # path to the codes of the model
    fastbpe = os.path.join(os.getcwd(), 'tools/fastBPE/fast')

    sentences = ['Le chat mange une pomme .']
    # sentences = [s.lower() for s in sentences]
    # bpe-ize sentences
    sentences = to_bpe(sentences)
    print('\n\n'.join(sentences))

    # check how many tokens are OOV
    n_w = len([w for w in ' '.join(sentences).split()])
    n_oov = len([w for w in ' '.join(sentences).split() if w not in dico.word2id])
    print('Number of out-of-vocab words: %s/%s' % (n_oov, n_w))

    # add </s> sentence delimiters
    sentences = [(('</s> %s </s>' % sent.strip()).split()) for sent in sentences]

    bs = len(sentences)
    slen = max([len(sent) for sent in sentences])

    word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
    for i in range(len(sentences)):
        sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
        word_ids[:len(sent), i] = sent

    lengths = torch.LongTensor([len(sent) for sent in sentences])
                                
    # NOTE: No more language id (removed it in a later version)
    # langs = torch.LongTensor([params.lang2id[lang] for _, lang in sentences]).unsqueeze(0).expand(slen, bs) if params.n_langs > 1 else None
    langs = None    
        
    tensor_xlm = model('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
    tensor_xlm = tensor_xlm.permute(1, 0, 2)
    print('tensor_xlm norm=', torch.norm(tensor_xlm))
    print('tensor_xlm.size()=', tensor_xlm.size())

    # Hugging Face
    modelname = os.path.expanduser('~/pretrained_models/Flaubert/flaubert_base_cased')
    flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)
    sentence = "Le chat mange une pomme."
    token_ids = torch.tensor([flaubert_tokenizer.encode(sentence)])
    tensor_hf = flaubert(token_ids)[0]

    print('tensor_hf norm=', torch.norm(tensor_hf))
    print('tensor_hf.size()=', tensor_hf.size())

    print('torch.norm(tensor_hf - tensor_xlm)=', torch.norm(tensor_hf - tensor_xlm))

if __name__ == "__main__":
    main()