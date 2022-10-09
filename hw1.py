import jittor as jt
import numpy as np
import matplotlib.pyplot as plt

from jittor import nn, Module, init
if jt.has_cuda:
    jt.flags.use_cuda = 1

class Model(Module):
    def __init__(self):
        self.layer1 = nn.Linear(1, 100)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(100, 1)
    def execute (self,x) :
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

np.random.seed(0)
jt.set_seed(3)
n = 1000
batch_size = 50

def get_data(n):
    for i in range(n):
        x = np.random.rand(batch_size, 1)-0.5
        y = x**3
        yield jt.float32(x), jt.float32(y)

model = Model()
learning_rate = 0.1
optim = nn.SGD(model.parameters(), learning_rate)

for epoch in range(20):

    for i,(x, y) in enumerate(get_data(n)):
        pred_y = model(x)
        loss = jt.sqr(pred_y-y)
        loss_mean = loss.mean()
        optim.step(loss_mean)
        print(f"step {i}, loss = {loss_mean.numpy().sum()}")

x = np.linspace(-0.5, 0.5, 100)
y = x**3
x = np.expand_dims(x, axis=1)
x = jt.float32(x)



pred_y = model(x).numpy().squeeze(-1)
x = x.numpy().squeeze(-1)

print(x.shape)
print(y.shape)
print(pred_y.shape)

plt.plot(x, y)
plt.plot(x, pred_y)

plt.savefig("res.png")



