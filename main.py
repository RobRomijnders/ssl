import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from model import Net
import dataloader

mnist = dataloader.read_data_sets("MNIST_data",norm=True)

cuda = True and torch.cuda.is_available()
bsz = 64
bsz_test = 100
epochs = 10
lr= 0.01
momentum = 0.5
seed = 1111
log_interval = 100
keep_prob = 0.8
cf = {'keep_prob'           :0.7,
      'conv1'               :10,
      'conv2'               :20,
      'fc1'                 :50}

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

mnist.cuda = cuda
mnist.bsz = bsz
data, targets = mnist.sample()

model = Net(cf)
if cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
self_ensemble_penalizer = nn.MSELoss()
batches = 0

def ramp_up():
    """Decay/ramp up function for the scaling of self_ensemble loss."""
    global batches
    batches += 1
    final = 1500.
    if batches < final:
      p = batches/final
      return float(np.exp(-5.*(p - 1) ** 2))
    else:
        if batches == final: print('Finished schedule for ramping up the lambda')
        return 1.0



def loss_calc(*,z1, z2, target):
    """Calculates the combined loss for semi-supervised learning. Note that on the validation
    set we calcalate only the nll_loss"""
    loss = F.nll_loss(z1[:5], target[:5])
    z2 = z2.detach()
    self_ensemble = self_ensemble_penalizer(z1,z2)
    loss += 0.05*ramp_up()*self_ensemble
    return loss


def train(epoch):
    """Train one epoch, which consists of a fixed number of batches"""
    model.train()
    for batch_idx in range(800):
        data, target = mnist.sample()
        optimizer.zero_grad()
        output1 = model(data)
        output2 = model(data)
        loss = loss_calc(z1=output1,z2=output2,target=target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: %3i [%6i/%6i (%.0f)] \tLoss: %.6f'%(epoch, batch_idx , 100, 100. * batch_idx / 100, loss.data[0]))

def test(epoch):
    """Test bsz_test number of batches from the validation set. Note that we calculate the normal nll_loss.
    The self_ensemble cost is not taken into account here"""
    model.eval()
    test_loss = 0
    correct = 0
    for k in range(bsz_test):
        data, target = mnist.sample(dataset='val')
        data.volatile = True
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    acc = correct/(bsz_test*bsz)
    test_loss /= bsz_test # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, bsz_test*bsz,
        100. * acc))

if __name__ == '__main__':
    try:
        for epoch in range(1, epochs + 1):
            train(epoch)
            test(epoch)
    except KeyboardInterrupt:
        #Except on keyboard interrupt so we can end training prematurely
        print('Finish training')
