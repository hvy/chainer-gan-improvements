import argparse
import numpy as np
from chainer import serializers
from models import Generator
import plot


# Resize the MNIST dataset to 32x32 images for convenience
# since the generator will create images with dimensions
# of powers of 2 (doubling upsampling in each deconvolution).
im_shape = (32, 32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-z', type=int, default=10)
    parser.add_argument('--n-samples', type=int, default=128)
    parser.add_argument('--in-generator-filename', type=str, default='generator.model')
    parser.add_argument('--out-filename', type=str, default='sample.png')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    n_z = args.n_z
    n_samples = args.n_samples
    in_generator_filename = args.in_generator_filename
    out_filename = args.out_filename

    generator = Generator(n_z, im_shape)
    serializers.load_hdf5(in_generator_filename, generator)

    zs = np.random.uniform(-1, 1, (n_samples, n_z)).astype(np.float32)
    x = generator(zs)

    plot.save_ims(out_filename, x.data)
