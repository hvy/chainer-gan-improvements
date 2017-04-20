import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--mbd', action='store_true', default=True)
    parser.add_argument('--out-generator-filename', type=str, default='./generator.model')
    return parser.parse_args()
