import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import numpy as np
from PIL import Image
from chainer import Variable, datasets, cuda
from chainer import optimizers as O
from chainer import functions as F
from models import Generator, Discriminator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--n-z', type=int, default=10)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    return parser.parse_args()


def resize(im, shape):
    im = Image.fromarray(im)
    im = im.resize(shape, Image.ANTIALIAS)
    return np.asarray(im)


if __name__ == '__main__':
    args = parse_args()
    gpu = args.gpu
    n_z = args.n_z
    max_epochs = args.max_epochs
    batch_size = args.batch_size

    if gpu >= 0:
        xp = cuda.cupy
    else:
        xp = np

    train, _ = datasets.get_mnist(withlabel=False, ndim=2)
    train = [resize(t, (32, 32)) for t in train]
    train = xp.asarray(train)

    train_size = train.shape[0]
    in_shape = train.shape[1:]

    generator = Generator(n_z, in_shape)
    generator_optimizer = O.Adam(alpha=1e-3, beta1=0.5)
    generator_optimizer.setup(generator)

    discriminator = Discriminator(in_shape)
    discriminator_optimizer = O.Adam(alpha=2e-4, beta1=0.5)
    discriminator_optimizer.setup(discriminator)

    if gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()
        generator.to_gpu()
        discriminator.to_gpu()

    for epoch in range(max_epochs):

        generator_epoch_loss = np.float32(0)
        discriminator_epoch_loss = np.float32(0)

        for i in range(0, train_size, batch_size):
            # Forward
            zs = xp.random.uniform(-1, 1, (batch_size, n_z)).astype(xp.float32)
            x_fake = generator(Variable(zs))
            print('x_fake.shape', x_fake.shape)
            y_fake = discriminator(x_fake)
            print('y_fake.shape', y_fake.shape)
            x_real = xp.zeros((batch_size, *in_shape), dtype=xp.float32)
            for xi in range(len(x_real)):
                x_real[xi] = train[xp.random.randint(train_size)]
            x_real = xp.expand_dims(x_real, 1)  # Grayscale channel axis
            y_real = discriminator(Variable(x_real))
            print('y_real.shape', y_real.shape)

            # Losses
            generator_loss = F.softmax_cross_entropy(y_fake, Variable(xp.ones(batch_size, dtype=xp.int32)))
            discriminator_loss = F.softmax_cross_entropy(y_fake, Variable(xp.zeros(batch_size, dtype=xp.int32)))
            discriminator_loss += F.softmax_cross_entropy(y_real, Variable(xp.ones(batch_size, dtype=xp.int32)))
            discriminator_loss /= 2

            # Backprop
            generator_optimizer.zero_grads()
            generator_loss.backward()
            generator_optimizer.update()

            discriminator_optimizer.zero_grads()
            discriminator_loss.backward()
            discriminator_optimizer.update()

            generator_epoch_loss += generator_loss.data
            discriminator_epoch_loss += discriminator_loss.data

        generator_avg_loss = round(generator_epoch_loss / train_size, 10)
        discriminator_avg_loss = round(discriminator_epoch_loss / train_size, 10)

        print('Epoch {} Loss Generator: {} Loss Discriminator: {}'
            .format(epoch + 1, generator_avg_loss, discriminator_avg_loss))

    print('Finished training')
