# coding: utf8

from abc import abstractmethod
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


class WSDEncoder(nn.Module):
    """ Generic Class to run a model on WSD dataset """

    def __init__(self, model, tokenizer):
        super(WSDEncoder, self).__init__()
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def encode(self, seq, padding=None):
        """ This method should use the tokenizer to encode sequence and add padding if specified"""
        pass

    @abstractmethod
    def forward(self, inputs):
        """ This method should run the model on inputs and outputs context vectors """
        pass

    @abstractmethod
    def collate_fn(self, inputs):
        """ This method should collate multiple inputs into a single a batch """
        pass

class TransformerWSDEncoder(WSDEncoder):
    """ Class to run transformer model on WSD dataset """

    def __init__(self, *args, **kwargs):
        super(TransformerWSDEncoder, self).__init__(*args, **kwargs)

    def encode(self, seq, padding=None):
        """ Encode a sequence to token ids and keep track of bpe span per token"""

        # dict to store bpe span: {token_idx: [start,end]}
        tok2span = {}

        tok_ids = []
        start = 0
        end = 0

        # cls token
        tok_ids.append(self.tokenizer.cls_token_id)
        end += 1


        tok2span = [(start, end)]
        start = end

        # iterate over sequence and encode on token ids
        for tok_id, tok in enumerate(seq):
            tok_encoding = self.tokenizer.encode(tok, add_special_tokens=False)
            end += len(tok_encoding)
            tok2span.append([start, end])
            start = end
            tok_ids.extend(tok_encoding)

        # add sep token
        tok_ids.append(self.tokenizer.sep_token_id)
        end += 1
        tok2span.append([start, end])

        # token ids to tensor
        tok_ids = torch.tensor(tok_ids)

        # padding
        if padding and len(tok_ids) < padding:
            tok_ids = F.pad(tok_ids, (0,padding-len(tok_ids)),value=self.tokenizer.pad_token_id)

        # attention mask on pad tokens
        att_mask = torch.ne(tok_ids, self.tokenizer.pad_token_id)
        tok2span.extend([[end,end] for x in range(len(tok_ids)-len(tok2span))])

        # span to tensor
        span = torch.tensor(tok2span)

        return tok_ids, att_mask, span

    def collate_fn(self, inputs):
        """ Collate inputs into a single batch """

        tok_ids, att_mask, span = [torch.stack(x) for x in list(zip(*inputs))]

        return (tok_ids, att_mask, span)

    def forward(self, inputs):
        """Run transformer model on inputs. Average bpes per token and remove cls and sep vectors"""

        tok_ids, att_mask, span = inputs

        output  = self.model(tok_ids, attention_mask=att_mask)[0] # mask is used for pad tokens

        # compute number of bpe per token
        first_bpe = span[:,:,0] # first bpe indice
        last_bpe = span[:,:,1] # last bpe indice
        n_bpe = last_bpe-first_bpe # number of bpe by token = first - last bpe from span

        # mask pad tokens
        mask = n_bpe.ne(0)
        n_bpe = n_bpe[mask] # get only actual token bpe

        # compute mean : sum up corresponding bpe then divide by number of bpe
        indices = torch.arange(n_bpe.size(0), device=output.device).repeat_interleave(n_bpe) # indices for index_add
        average_vectors = torch.zeros(n_bpe.size(0), output.size(2), device=output.device) # starts from zeros vector
        average_vectors.index_add_(0, indices, output[att_mask]) # sum of bpe based in indices
        average_vectors.div_(n_bpe.view(n_bpe.size(0),1)) # divide by number of bpe

        output_ = torch.zeros_like(output) # new output vector to match outputsize
        output_[mask] = average_vectors

        output = output_[:,1:-1,:] # get rid of cls and sep

        return output
