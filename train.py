import argparse
import numpy as np
from PIL import Image
from chainer import datasets, cuda, serializers
from chainer import optimizers as O
from chainer import functions as F
from models import Generator, Discriminator


# Resize the MNIST dataset to 32x32 images for convenience
# since the generator will create images with dimensions
# of powers of 2 (doubling upsampling in each deconvolution).
im_shape = (32, 32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--n-z', type=int, default=10)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--minibatch-discrimination', action='store_true', default=True)
    parser.add_argument('--out-generator-filename', type=str, default='./generator.model')
    return parser.parse_args()


def resize_im(im, shape):
    im = Image.fromarray(im)
    im = im.resize(shape, Image.ANTIALIAS)
    return np.asarray(im)


if __name__ == '__main__':
    args = parse_args()
    gpu = args.gpu
    n_z = args.n_z
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    out_generator_filename = args.out_generator_filename
    minibatch_discrimination = args.minibatch_discrimination

    # Prepare the training data
    train, _ = datasets.get_mnist(withlabel=False, ndim=2)
    train = [resize_im(t, im_shape) for t in train]
    train = np.asarray(train)
    train_size = train.shape[0]
    assert train.shape[1:] == im_shape

    # Prepare the models
    generator = Generator(n_z, im_shape)
    generator_optimizer = O.Adam(alpha=1e-3, beta1=0.5)
    generator_optimizer.setup(generator)

    discriminator = Discriminator(im_shape)
    discriminator_optimizer = O.Adam(alpha=2e-4, beta1=0.5)
    discriminator_optimizer.setup(discriminator)

    if gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()
        generator.to_gpu()
        discriminator.to_gpu()
        xp = cuda.cupy
    else:
        xp = np

    for epoch in range(max_epochs):
        generator_epoch_loss = np.float32(0)
        discriminator_epoch_loss = np.float32(0)

        for i in range(0, train_size, batch_size):
            # Forward
            zs = xp.random.uniform(-1, 1, (batch_size, n_z)).astype(xp.float32)
            x_fake = generator(zs)
            y_fake = discriminator(x_fake, minibatch_discrimination=minibatch_discrimination)
            x_real = xp.zeros((batch_size, *im_shape), dtype=xp.float32)
            for xi in range(len(x_real)):
                x_real[xi] = xp.array(train[np.random.randint(train_size)])
            x_real = xp.expand_dims(x_real, 1)  # Grayscale channel axis
            y_real = discriminator(x_real, minibatch_discrimination=minibatch_discrimination)

            # Losses
            generator_loss = F.softmax_cross_entropy(y_fake, xp.ones(batch_size, dtype=xp.int32))
            discriminator_loss = F.softmax_cross_entropy(y_fake, xp.zeros(batch_size, dtype=xp.int32))
            discriminator_loss += F.softmax_cross_entropy(y_real, xp.ones(batch_size, dtype=xp.int32))
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

        generator_avg_loss = generator_epoch_loss / train_size
        discriminator_avg_loss = discriminator_epoch_loss / train_size

        print('Epoch {} Loss Generator: {} Loss Discriminator: {}'
              .format(epoch + 1, generator_avg_loss, discriminator_avg_loss))

    print('Saving model', out_generator_filename)
    serializers.save_hdf5(out_generator_filename, generator)

    print('Finished training')
