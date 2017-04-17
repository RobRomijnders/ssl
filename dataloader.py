# I copy-pasted parts of this code from the Tensorflow source code. Credits to them!
import gzip
import os
import numpy
from six.moves import urllib
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

import numpy as np

from torch.autograd import Variable
import torch

def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels


class Dataloader(object):

  def __init__(self, data, sizes,norm, ssl_lbls):
    """Construct a DataSet.
    """
    self.data = {}
    self.sizes = sizes
    self._index_in_epoch = 0
    self._epochs_completed = 0

    self.logging_pol=None

    ds = ['train_ulbl','train','val','test']
    self.mean = None
    self.std = None

    self.bsz_lbl = ssl_lbls

    self.bsz = 32
    self.cuda = False
    for d in ds:
        X,y  = data[d]

        X = X.copy().astype(np.float32)
        y = y.copy().astype(np.int16)
        if norm:
            if not self.mean:
                self.mean = np.mean(X)
                self.std = np.sqrt(np.var(X)+1E-09)
            X -= self.mean
            X /= self.std
        self.data[d] = [X,y]
    return

  @property
  def im_size(self):
      _,H,W,_ = self.data['train'][0].shape
      return H, W

  def unnorm(self,X):
      return (X*self.std)+self.mean

  def next_batch(self, dataset,batch_size = None):
    """Return the next `batch_size` examples from this data set."""
    if not batch_size:
        batch_size = self.bsz
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self.sizes[dataset]:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self.sizes[dataset])
      numpy.random.shuffle(perm)
      self.data[dataset][0] = self.data[dataset][0][perm]
      self.data[dataset][1] = self.data[dataset][1][perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self.sizes[dataset]
    end = self._index_in_epoch
    return self.data[dataset][0][start:end], self.data[dataset][1][start:end]

  def sample(self, batch_size = None,dataset='ssl'):
      """Packs the data from self.next_batch() into Pytorch tensors"""
      if not batch_size:
          batch_size = self.bsz
      if dataset == 'ssl':
          X_lbl, y_lbl = self.next_batch(dataset='train',batch_size=self.bsz_lbl)
          X_ulbl, y_ulbl = self.next_batch(dataset='train_ulbl', batch_size=self.bsz-self.bsz_lbl)

          X = np.concatenate((X_lbl,X_ulbl))
          y = np.concatenate((y_lbl, y_ulbl))
      elif dataset in ['val', 'test']:
          X, y = self.next_batch(dataset=dataset,batch_size=batch_size)
      else:
          assert False, 'Expected a dataset of name (ssl) or (val) or (test)'
      assert X.shape[0] == self.bsz
      data = torch.FloatTensor(np.transpose(X,[0,3,1,2]))
      targets = torch.LongTensor(y.astype(np.int64))
      if self.cuda:
        data = data.cuda()
        targets = targets.cuda()
      return Variable(data), Variable(targets)




def read_data_sets(train_dir, one_hot=False,norm=False, ssl_lbls = 10, ssl_ratio = 0.1):
  """For the train set, only (ssl_ratio) labels are stored. For self.sample(), every first (ssl_lbls) are labelled"""
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000

  #TRAIN
  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)
  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)

  #TEST
  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)
  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)

  #VALIDATION
  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  #Make SSL
  N = train_images.shape[0]
  num_labelled = int(ssl_ratio*N)
  num_unlabelled = N - num_labelled
  print('SSL: We have %i train samples, of which %i labelled and %i unlabelled'%(N,num_labelled,num_unlabelled))
  train_images_lbl = train_images[:num_labelled]
  train_labels_lbl = train_labels[:num_labelled]
  train_images_ulbl = train_images[num_labelled:]
  train_labels_ulbl = np.zeros((num_unlabelled,))
  data = {  'train'     :(train_images_lbl, train_labels_lbl),
            'train_ulbl':(train_images_ulbl, train_labels_ulbl),
            'val'       :(validation_images, validation_labels),
            'test'      :(test_images, test_labels)}
  sizes = {'train'      :train_images_lbl.shape[0],
           'train_ulbl' :train_images_ulbl.shape[0],
            'val'       :validation_images.shape[0],
            'test'      :test_images.shape[0]}
  return Dataloader(data, sizes, norm, ssl_lbls)
