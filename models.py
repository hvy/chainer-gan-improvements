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
            t=L.Linear(in_shape, n_kernels*kernel_dim))

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
    def __init__(self, nz):
        super(Generator, self).__init__(
            fc=L.Linear(nz, 7*7*64),
            dc1=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
            dc2=L.Deconvolution2D(32, 1, 4, stride=2, pad=1),
            bn=L.BatchNormalization(32))

    def __call__(self, z, test=False):
        h = F.relu(self.fc(z))
        h = F.reshape(h, (h.shape[0], 64, 7, 7))
        h = F.relu(self.bn(self.dc1(h), test=test))
        h = F.sigmoid(self.dc2(h))
        return h


class Discriminator(Chain):
    def __init__(self, mbd=True):
        super(Discriminator, self).__init__(
            c1=L.Convolution2D(1, 32, 4, stride=2, pad=1),
            c2=L.Convolution2D(32, 64, 4, stride=2, pad=1),
            fc1=L.Linear(7*7*64, 64),
            fc3=L.Linear(64+16, 1),
            bn1=L.BatchNormalization(64),
            bn2=L.BatchNormalization(64),
            bn3=L.BatchNormalization(64+16))

        if mbd:
            self.add_link('fc2', MinibatchDiscrimination(64, 16, 8))
        else:
            self.add_link('fc2', L.Linear(64, 64+16))

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c1(x))
        h = F.leaky_relu(self.bn1(self.c2(h), test=test))
        h = F.leaky_relu(self.bn2(self.fc1(h), test=test))
        h = F.leaky_relu(self.bn3(self.fc2(h), test=test))
        h = self.fc3(h)
        # Skip sigmoid() in case we are computing the loss with softplus
        # h = F.sigmoid(h)
        return h
