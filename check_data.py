import torch

def main():
    path = '/home/getalp/lethip/Data/BERT-FR/data_processed/combined_corpus_nc/BPE/50k/train.fr.0.pth'
    data = torch.load(path)

if __name__ == "__main__":
    main()