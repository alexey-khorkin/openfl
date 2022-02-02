import mxnet as mx
from mxnet.gluon import nn
from mxnet import init, nd

def create_net():
    ctx=mx.cpu(0)

    net = nn.Sequential()
    net.add(nn.Conv2D(channels=64, kernel_size=3, padding=1, activation='relu'),
            nn.BatchNorm(),
            nn.MaxPool2D(),
            nn.Conv2D(channels=128, kernel_size=3, padding=1, activation='relu'),
            nn.BatchNorm(),
            nn.MaxPool2D(),
            nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'),
            nn.BatchNorm(),
            nn.MaxPool2D(),
            nn.Flatten(),
            nn.Dense(64),
            nn.Activation('relu'),
            nn.Dropout(rate=0.005),
            nn.Dense(30))
    
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    net(nd.ones((1, 1, 96, 96), ctx=ctx)) # first forward pass for weight initialization
    print('Model has been returned!')
    return net
