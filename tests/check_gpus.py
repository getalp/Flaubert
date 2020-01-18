import torch

def main():
    print('NUMBER OF GPUS DETECTED:', torch.cuda.device_count())

if __name__ == "__main__":
    main()