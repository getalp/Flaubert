# coding: utf8

import os

from collections import defaultdict
from lxml import etree


class Instance:
    """ Class to represent an instance to be disambiguated """

    def __init__(self, id, sent_id=None, word_form=None, key=None, lemma=None, pos=None, tok_id=None, labels=None, first_label=None,
                 source=None, is_mwe=False, context=None):
        self.id = id
        self.sent_id = sent_id
        self.word_form = word_form
        self.key = key
        self.lemma = lemma
        self.pos = pos
        self.tok_id = tok_id
        self.labels = labels
        self.first_label = first_label
        self.source = source
        self.is_mwe = is_mwe
        self.context = context

    def __str__(self):

        string = []

        id_str = "ID = {}".format(self.id)
        string.append(id_str)

        key = "Key = {}".format(self.key)
        string.append(key)


        label = "First Label = {} ".format(self.first_label)
        string.append(label)

        labels = "Labels = {}".format(' '.join(self.labels))
        string.append(labels)

        tok_id = "Tok ID = {}".format(self.tok_id)
        string.append(tok_id)

        context = "\nContext\n{}".format(self.context)
        string.append(context)

        return '\n'.join(string)


class WSDDataset:
    """ Class containing Instances"""

    def __init__(self, sent_id2sent=None, sent_id2instances=None,  id2instance=None, key2instances=None):

        self.sent_id2sent, self.sent_id2instances,  self.id2instance, self.key2instances = sent_id2sent, sent_id2instances,  id2instance, key2instances
        self.instances = list(self.id2instance.values())

    def get_target_pos(self):
        """ Return PoS contained in dataset """
        return {x.split('.')[1] for x in self.key2instances}

    def get_target_words(self):
        """ Return vocabulary contined in dataset """
        return {x.split('.')[0] for x in self.key2instances}

    def get_target_keys(self):
        """ Return keys contained in dataset """
        return self.key2instances.keys()

    def __len__(self):
        return sum([len(self.sent_id2instances[id]) for id in self.sent_id2instances])

    def get_instances(self):

        for instance in self.id2instance.values():
            yield instance

    def get_labels(self):

        labels = defaultdict(bool)
        for i, instance in self.id2instance.items():
            labels[instance.first_label] = True

        return list(labels.keys())


class WSDDatasetReader:
    """ Class to read a WSD data directory. The directory should contain .data.xml and .gold.key.txt files"""

    def count_tokens(self, data_dir):
        """ Count number of tokens in WSD data """

        count = 0
        xml_fpath, gold_fpath = self.get_data_paths(data_dir)
        tree = etree.parse(xml_fpath)
        corpus = tree.getroot()

        for text in corpus:
            for sent in text:
                count += len(list(sent))

        return count

    def get_data_paths(self, indir):
        """ Get file paths from WSD dir """

        xml_fpath, gold_fpath = None, None

        for f in os.listdir(indir):
            if f.endswith('.data.xml'):
                xml_fpath = indir + f
            if f.endswith('.gold.key.txt'):
                gold_fpath = indir +f

        return xml_fpath, gold_fpath

    def read_gold(self, infile):
        """ Read .gold.key.txt and return data as dict.
            :param infile: fpath to .gold.key.txt file
            :type infile: str
            :return: return data into dict format : {str(instance_id): set(label)}
            :rtype: dict
        """
        return {line.split()[0]: tuple(line.rstrip('\n').split()[1:]) for line in open(infile).readlines()}


    def read_from_data_dirs(self, data_dirs, target_pos=None, target_words=None, target_keys=None,
                            ignore_source=[], keep_mwe=False, add_context_to_instance=False):
        """ Read WSD data and return as WSDDataset """

        id2sent = {}
        id2instance = {}
        key2instances = defaultdict(list)
        sent_id2instances = defaultdict(list)


        for d in data_dirs:
            xml_fpath, gold_fpath = self.get_data_paths(d)
            target_pos, target_words, target_keys = target_pos, target_words, target_keys

            # read gold file
            id2gold = self.read_gold(gold_fpath)

            sentences = self.read_sentences(d, keep_mwe=keep_mwe)

            # Parse xml
            tree = etree.parse(xml_fpath)
            corpus = tree.getroot()

            # process data
            # iterate over document
            for text in corpus:
                text_id = text.get('id')
                if len(text.get('id').split('.'))> 1:
                    source = text.get('id').split('.')[0]
                else:
                    source = corpus.get('source')

                # iterates over sentences
                for sentence in text:

                    if source in ignore_source:
                        continue

                    sent_id = sentence.get('id') # sentence id
                    sent_id_with_source = source + "." + sent_id if len(sent_id.split('.')) == 2 else source + ' '.join(sent_id.split('.')[1:])

                    flag = False # use to check if annotated instance in sentence

                    sent = next(sentences) # get sentence

                    tok_idx = 0

                    # iterate over tokens
                    for tok in sentence:

                        lemma, pos = tok.get('lemma'), tok.get('pos')
                        key = lemma + '__' + pos
                        wf = tok.text
                        subtokens = wf.split(' ')

                        is_mwe = True if len(subtokens) > 1 else False  # is multiword expression

                        # add sense annotated token
                        if tok.tag == "instance":

                            if target_pos and pos not in target_pos:
                                pass
                            elif target_words and lemma not in target_words:
                                pass
                            elif target_keys and key not in target_keys:
                                pass
                            else:

                                id = tok.get("id")
                                id_with_source = source + "." + id if len(id.split('.')) == 3 else source + '.' + '.'.join(id.split('.')[1:])

                                target_labels = id2gold[id]
                                target_first_label = target_labels[0]

                                if keep_mwe:
                                    tgt_idx = tok_idx
                                else:
                                    # We focus on the head of the target mwe instance
                                    if pos == "VERB":
                                        tgt_idx = tok_idx # head is mostly the first token as most mwe verb targets are phrasal verbs (i.g lift up)
                                    else:
                                        tgt_idx = tok_idx + len(subtokens)-1 # other pos head are generally the last token of the mwe (i.g European Union)

                                # creates Instance
                                instance = Instance(id_with_source, sent_id_with_source, wf, key, lemma=lemma,pos=pos,
                                                    tok_id=tgt_idx, labels=target_labels,
                                                    first_label=target_first_label,
                                                    source=source, is_mwe=is_mwe)

                                if add_context_to_instance:
                                    instance.context = sent

                                # add instance to dataset
                                key2instances[key].append(instance)
                                sent_id2instances[sent_id_with_source].append(instance)
                                id2instance[id_with_source] = instance
                                flag = True

                        # updates token indice
                        off_set = 1 if keep_mwe else len(subtokens)
                        tok_idx += off_set

                        # add sentence to dataset if it contains instances to be disambiguated
                        if flag:
                            id2sent[sent_id_with_source] = sent


        return WSDDataset(id2sent, sent_id2instances,  id2instance, key2instances)


    def read_sentences(self, data_dir, keep_mwe=True):
        """ Read sentences from WSD data"""

        xml_fpath,_ = self.get_data_paths(data_dir)
        return self.read_sentences_from_xml(xml_fpath,  keep_mwe=keep_mwe)

    def read_sentences_from_xml(self, infile, keep_mwe=False):
        """ Read sentences from xml file """

        # Parse xml
        tree = etree.parse(infile)
        corpus = tree.getroot()

        for text in corpus:
            for sentence in text:
                if keep_mwe:
                    sent = [tok.text.replace(' ', '_') for tok in sentence]
                else:
                    sent = [subtok for tok in sentence for subtok in tok.text.split(' ') ]
                yield sent


    def read_target_keys(self, infile):
        """ Read target keys """
        return [x.rstrip('\n') for x in open(infile).readlines()]
