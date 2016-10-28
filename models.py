import chainer
from functools import reduce
from chainer import Chain
from chainer import functions as F
from chainer import links as L


def lindim(dims, scale, n):
    d = map(lambda x: x // scale, dims)
    d = reduce(lambda x, y: x * y, d)
    return d * n


def convdim(dims, scale, n):
    return (n, dims[0] // scale, dims[1] // scale)


class MinibatchDiscrimination(Chain):
    def __init__(self, in_shape, n_kernels, kernel_dim):
        super(MinibatchDiscrimination, self).__init__(
            t=L.Linear(in_shape, n_kernels*kernel_dim)
        )
        self.n_kernels = n_kernels
        self.kernel_dim = kernel_dim

    def __call__(self, x):
        minibatch_size = x.shape[0]
        activation = F.reshape(self.t(x), (-1, self.n_kernels, self.kernel_dim))
        activation_ex = F.expand_dims(activation, 3)
        activation_ex_t = F.expand_dims(F.transpose(activation, (1, 2, 0)), 0)
        activation_ex, activation_ex_t = F.broadcast(activation_ex, activation_ex_t)
        diff = activation_ex - activation_ex_t

        xp = chainer.cuda.get_array_module(x.data)
        eps = F.expand_dims(xp.eye(minibatch_size, dtype=xp.float32), 1)
        eps = F.broadcast_to(eps, (minibatch_size, self.n_kernels, minibatch_size))
        sum_diff = F.sum(abs(diff), axis=2)
        sum_diff = F.broadcast_to(sum_diff, eps.shape)
        abs_diff = sum_diff + eps

        minibatch_features = F.sum(F.exp(-abs_diff), 2)
        return F.concat((x, minibatch_features), axis=1)


class Generator(Chain):
    def __init__(self, n_z, out_shape):
        super(Generator, self).__init__(
            fc0=L.Linear(n_z, lindim(out_shape, 2**4, 256)),
            dc1=L.Deconvolution2D(256, 128, 4, stride=2, pad=1),
            dc2=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            dc3=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
            dc4=L.Deconvolution2D(32, 1, 4, stride=2, pad=1),
            bn0=L.BatchNormalization(lindim(out_shape, 2**4, 256)),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(64),
            bn3=L.BatchNormalization(32)
        )
        self.out_shape = out_shape

    def __call__(self, z, test=False):
        h = F.relu(self.bn0(self.fc0(z), test=test))
        h = F.reshape(h, ((z.shape[0],) + convdim(self.out_shape, 2**4, 256)))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        h = F.sigmoid(self.dc4(h))
        return h


class Discriminator(Chain):
    def __init__(self, in_shape):
        super(Discriminator, self).__init__(
            c0=L.Convolution2D(1, 32, 4, stride=2, pad=1),
            c1=L.Convolution2D(32, 64, 4, stride=2, pad=1),
            c2=L.Convolution2D(64, 128, 4, stride=2, pad=1),
            c3=L.Convolution2D(128, 256, 4, stride=2, pad=1),
            fc4=L.Linear(lindim(in_shape, 2**4, 256), 512),
            mbd=MinibatchDiscrimination(512, 32, 8),
            fc5=L.Linear(512, 512+32),  # Alternative to minibatch discrimination
            fc6=L.Linear(512+32, 2),
            bn1=L.BatchNormalization(64),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(256)
        )

    def __call__(self, x, minibatch_discrimination=True, test=False):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h), test=test))
        h = F.leaky_relu(self.bn2(self.c2(h), test=test))
        h = F.leaky_relu(self.bn3(self.c3(h), test=test))
        h = self.fc4(h)

        if minibatch_discrimination:
            h = self.mbd(h)
        else:
            h = F.leaky_relu(self.fc5(h))

        h = self.fc6(h)
        return h
