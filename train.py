import numpy as np
from PIL import Image
from chainer import datasets, cuda, serializers
from chainer import optimizers as O
from chainer import functions as F
from models import Generator, Discriminator

import config


if __name__ == '__main__':
    args = config.parse_args()
    gpu = args.gpu
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    out_generator_filename = args.out_generator_filename

    # Prepare the training data
    train, _ = datasets.get_mnist(withlabel=False, ndim=3)

    # Prepare the models
    generator = Generator(nz=args.nz)
    generator_optimizer = O.Adam(alpha=1e-3, beta1=0.5)
    generator_optimizer.setup(generator)

    discriminator = Discriminator(mbd=args.mbd)
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

    train_size = len(train)

    for epoch in range(max_epochs):
        generator_epoch_loss = np.float32(0)
        discriminator_epoch_loss = np.float32(0)

        for i in range(0, train_size, batch_size):
            # Forward
            zs = xp.random.uniform(-1, 1, (batch_size, args.nz)).astype(xp.float32)
            x_fake = generator(zs)
            y_fake = discriminator(x_fake)
            x_real = xp.zeros((batch_size, *train[0].shape), dtype=xp.float32)
            for xi in range(len(x_real)):
                x_real[xi] = xp.array(train[np.random.randint(train_size)])
            y_real = discriminator(x_real)

            # Losses
            """
            # GAN loss using softmax_cross_entropy
            generator_loss = F.softmax_cross_entropy(y_fake, xp.ones(batch_size, dtype=xp.int32))
            discriminator_loss = F.softmax_cross_entropy(y_fake, xp.zeros(batch_size, dtype=xp.int32))
            discriminator_loss += F.softmax_cross_entropy(y_real, xp.ones(batch_size, dtype=xp.int32))
            discriminator_loss /= 2
            """

            g_loss = F.softplus(-y_fake)
            g_loss = F.sum(g_loss) / g_loss.shape[0]
            d_loss = F.softplus(-y_real)
            d_loss += F.softplus(y_fake)
            d_loss = F.sum(d_loss) / d_loss.shape[0]
            d_loss /= 2.0

            print(g_loss.data)
            print(d_loss.data)

            # Backprop
            generator_optimizer.zero_grads()
            g_loss.backward()
            generator_optimizer.update()

            discriminator_optimizer.zero_grads()
            d_loss.backward()
            discriminator_optimizer.update()

            generator_epoch_loss += g_loss.data
            discriminator_epoch_loss += d_loss.data

        generator_avg_loss = generator_epoch_loss / train_size
        discriminator_avg_loss = discriminator_epoch_loss / train_size

        print('Epoch {} Loss Generator: {} Loss Discriminator: {}'
              .format(epoch + 1, generator_avg_loss, discriminator_avg_loss))

    print('Saving model', out_generator_filename)
    serializers.save_hdf5(out_generator_filename, generator)

    print('Finished training')
