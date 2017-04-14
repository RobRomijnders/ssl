# Semi supervised learning with self ensembling
This repo implements the self ensembling technique in PyTorch. The code spans only 80 lines and about 50 lines of helper code. I wouldn't use the code in production, but it illustrates the concept.

[Here](https://arxiv.org/abs/1610.02242) is the paper.

# Quick summary
Self emsembling trains neural networks on partially labelled data. For some training batch, we run the network twice with different [dropouts](https://arxiv.org/pdf/1207.0580.pdf). On all samples, we penalize the squared difference between the predictions from both networks. On the labelled data, we also penalize the negative log likelihood of the labels. This technique beats other semi-supervised learners on the [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) and [SVHN](http://ufldl.stanford.edu/housenumbers/) datasets.

# What is semi-supervised learning (SSL)?
In semi supervised learning we use both labelled and unlabelled data. In many problems, we have data in abundance. Yet, it cost much time/effort/money to obtain labels. For example:

  * __Chest xrays__ We can have lots of scans in a database, but is cost much money to pay a radiologists labelling them.
  * __Computer vision__ We can scrape Google Images and get a huge dataset of images, but it costs effort to set up AMT and have people label them
  * __NLP__ We have many records of open books, emails and Twitter feed, but it cost mush effort to have people label the sentiment

# What links SSL and neural networks?
Neural networks learn feature representations. (Also named [distributed representations](http://www.nature.com/nature/journal/v521/n7553/full/nature14539.html), latent representations or hidden representations) This representations could be learned on both labelled and unlabelled data. With labelled data, we could think of the hidden neurons in a convolutional neural network. With unlabelled data, we could think of, for example, word vectors.

# Examples of SSL techniques
Self ensembling is not the first SSL technique. Other notable attempts:

  * [__Auto encoders__](https://arxiv.org/abs/1406.5298) Auto encoders learn to encode and decode data onto itself. Usually we obtain feature representations from a bottleneck between the encoder and decoder. In SSL, we train the auto encoder on all data. Secondly, we train a small classifier for the labelled data, using the feature representations as input. (You can use a variational auto encoder too)
  * [__Ladder networks__](https://arxiv.org/abs/1507.02672) Ladder networks extend the auto encoder to multiple levels. People visualize auto encoders as an up-side-down diagram. Ladder networks stack many such auto encoders. You can visualize it as a ladder.
  * [__Generative adversarial networks__](https://arxiv.org/abs/1606.03498) GAN's play a mini-max game. The generator learns to generate fake images; the discriminator learns to discriminate between fake and real images. In SSL, we view the discriminator as an extended classifier. Say you classify the 10 [MNIST](http://yann.lecun.com/exdb/mnist/) digits. The GAN discriminator would classify 11 classes: the 10 MNIST classes and the 11th class representing a fake. 

Personally, I like the self ensembler for its simplicity. In auto encoders, we train a decoder without using it; in ladder networks, we train multiple decoders without using them; and for GAN's, we train a generator without using it. Moreover, the reseach community still works on stable algorithms to train GAN's. In that perspective, self ensembling trains only a single network. So in the engineering, we need to monitor only one piece of code.

# The code
The code consists of three scripts 

  * ```main.py``` initializes data and the network
  * ```model.py``` contains the PyTorch module for a convnet
  * ```dataloader.py``` provides some sample functions for the MNIST data


As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com