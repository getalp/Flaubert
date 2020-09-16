# French Verb Sense Disambiguation
 ## Task Description
 Verb Sense Disambiguation (VSD) is a subpart of the Word Sense Disambiguation (WSD) task where only verbs are the target of the disambiguation.
 We used the FrenchSemEval (FSE) dataset [[1]](#References) to evaluate our model on this task for French. 
 
 ## Dataset
 The dataset includes:
  - Training data: Examples extracted from the sense entries in Wiktionary (dump of the 04-20-2018 ).
  - Evaluation data: Manual annotations of verbs with senses from Wiktionary. The occurrences of verbs were extracted from the French Wikipedia, the French Treebank [[2]]([#References]) and the Sequoia corpus [[3]](#References) .   
 
  
Both the training and evaluation data are in the format proposed in Raganato's WSD evaluation framework [[4]](#References) (http://lcl.uniroma1.it/wsdeval/ for more details).
This format consists of:
 - a ``.data.xml`` file which contains the sentences in xml format
 - a ``.gold.key.txt`` file which contains the labels of the instances


## Disambiguation process
The disambiguation is performed following [[1]](#References):
 1. We run the Flaubert model on the training / evaluation data to obtain contextual representations for the target occurrences.
 2. We compute sense representations by averaging the vector representations of their instances.
 3. We use a knn classifier based on cosine similarity to predict the labels of the evaluation instances by comparing them to the sense representations.
 
## Run the code

**1. Download the FrenchSemEval (FSE) dataset available [here](http://www.llf.cnrs.fr/dataset/fse/)** (called ```$FSE_DIR``` hereafter)

**2. Prepare the data**
  ```python
  python prepare_data.py --data $FSE_DIR --output $DATA_DIR
  ```
  Options:
  
  ``--train $OTHER_TRAIN_DIR``: To use other training data than Wiktionary (it must be in the same format described [above](#dataset)). 

**3. Run the model and evaluate with ```flue_vsd.py```**
  ```python
  python flue_vsd.py --exp_name myexp --model flaubert-base-cased --data $DATA_DIR --padding 80 --batchsize 32 --device 0 --output $OUTPUT_DIR
  ```
Options:

 ``--exp_name [name]`` (str) : the name of the experiment.
 
 ``--model [name|path]`` (str) : name of the pretrained model (i.g 'flaubert-base-cased') or the path to a model checkpoint. The pretrained model or model checkpoint should be one of the Flaubert/Camembert/Bert class from the Hugginface API.
 
 ``--data [dirpath]`` (str) : path to the directory containing both subdirectories ``train``and ``test``
 
 ``--padding [n]`` (int) : pad sentences to length ``n``
 
 ``--batchsize [n]`` (int): the size of batches
 
 ``--device [n]`` (int) : to run the model on GPU. (default -1 is CPU).
 
 ``--output [dirpath]`` : the dirpath where the vectors will be output
 
 ``--output_logs [path]`` (str) : path to output logs (.csv file)
 
 ``--output_pred [path]`` (str): path to output the predictions (one ``instance_id \\t pred`` per line)
 
 ``--output_score [path]`` (str) : path to output score (.csv file).
 
 
## Use other vectors
It is possible to evaluate vectors from any other model than transformers. To do so:

1. Dump the vectors output by the model in ``$TRAIN_VECS``and ``$TEST_VECS`` files which format should be one ``instance_id \t vector `` per line.

2. Run ``wsd_evaluation.py``
```python
python wsd_evaluation --exp_name myexp --train_data $TRAIN_DIR --train_vecs $TRAIN_VECS --test_data $TEST_DIR --test_vecs $TEST_VECS --average --target_pos V
```
See online help for further details and options.

# References
[1] Segonne, V., Candito, M., and Crabb ́e, B. (2019). Usingwiktionary as a resource for wsd: the case of frenchverbs. *
InProceedings of the 13th International Confer-ence on Computational Semantics-Long Papers, pages259–270*

[2] Abeill ́e, A. and N. Barrier (2004).  Enriching a french treebank.  *InProceedings of LREC 2004, Lisbon,Portugal*

[3] Candito,  M.  and  D.  Seddah  (2012).   Le  corpus  sequoia:  annotation  syntaxique  et  exploitation  pourl’adaptation d’analyseur par pont lexical. *InTALN 2012-19e conf ́erence sur le Traitement Automatiquedes Langues Naturelles.*

[4] Raganato, A., J. Camacho-Collados, and R. Navigli (2017). Word sense disambiguation: A unified eval-uation framework and empirical comparison.  *InProceedings of the 15th Conference of the EuropeanChapter of the Association for Computational Linguistics:  Volume 1,  Long Papers,  Volume 1,  pp.99–110.*
