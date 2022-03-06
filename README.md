# Deep Learning with Noisy Labelled Data Literature

|Year|Conf|Repo|Title|
|----|----|----|-----|
|2017|||[Decoupling "when to update" from "how to update"](https://arxiv.org/abs/1706.02613)|
|2018||[Pt](https://github.com/bhanML/Co-teaching)|[Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels](https://arxiv.org/abs/1804.06872)|
|2019||[Pt](https://github.com/xingruiyu/coteaching_plus)|[How does Disagreement Help Generalization against Label Corruption?](https://arxiv.org/abs/1901.04215)|
|2019|||[Deep Self-Learning From Noisy Labels](https://arxiv.org/abs/1908.02160)|
|2019|||[Probabilistic End-to-end Noise Correction for Learning with Noisy Labels](https://arxiv.org/abs/1903.07788)|
|2020|||[Combating noisy labels by agreement: A joint training method with co-regularization](https://arxiv.org/abs/2003.02752)|
|2020|||[Label Noise Types and Their Effects on Deep Learning](https://arxiv.org/abs/2003.10471)|
|2020|||[Identifying Mislabeled Data using the Area Under the Margin Ranking](https://arxiv.org/abs/2001.10528)|
|2020|ICLR|[Pt](https://github.com/LiJunnan1992/DivideMix)|[DivideMix: Learning with Noisy Labels as Semi-supervised Learning](https://arxiv.org/abs/2002.07394)|
|2021||[Pt](https://github.com/yingyichen-cyy/Nested-Co-teaching)|[Boosting Co-teaching with Compression Regularization for Label Noise](https://arxiv.org/abs/2104.13766)|
|2021|ICLR|[Pt](https://github.com/pxiangwu/PLC)|[LEARNING WITH FEATURE-DEPENDENT LABEL NOISE: A PROGRESSIVE APPROACH](https://arxiv.org/abs/2103.07756)|


## Summaries

### 1 - Decoupling "when to update" from "how to update"
Uses two networks. Looks disagreement region for updates. Does not requires any clean subset. Uses Large Faces in the Wild (LWF) and MNIST datasets in experiments.
Disagreement region always contains noisy samples and network start to memorizes these samples in later iterations of training.

### 2 - Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels
Uses two networks. Updates is done with small loss instances of peer network. Does not require any clean subset. Assumes that noise ratio is known. Uses MNIST, CIFAR10, CIFAR100 dasets in experiments. Apply symmetric and asymmetric syhentetic noises in experiments.

### 3 - How does Disagreement Help Generalization against Label Corruption?
Uses two networks. Updates is done with small loss instances of peer network which are in the disagreement region. It can be viewed as combination of Decoupling and Co-teaching. Does not require any clean subset. Assumes that noise ration is known. Uses MNIST, CIFAR10, CIFAR100, NEW and T-ImageNet datasets in experiments. Apply symmetric and asymmetric syhentetic noise.

### 4 - Deep Self-Learning From Noisy Labels

### 5 - Probabilistic End-to-end Noise Correction for Learning with Noisy Labels
Uses a single network. Try to model image labels as probabilistic distrubitions in soft-label space. Initially, train a neural network with noisy labels using classical cross entropy loss. At this step, learning rate is set to high value and they claim that high learning rate prevents overfitting to noisy labels. Then, soft labels are initialized with noisy labels. In the second step, they optimize both network parameters and soft labels to decrease loss value. At this step, loss function includes three terms: classification loss, compatibility loss and entropy loss as a regulazir. Classification loss is a modified version of KL divergence. Compatibility loss is cross entropy between soft labels and initial noisy labels. Entropy loss is applied to output of the network to avoid flat predictions. In the third step, they fine tune networks parameters over optimized soft labels using a small learning rate.

They used CIFAR10, CIFAR100, CUB-200 and Clothing1M datesets in experiments. They used symmetric and asymmetric syhentetic noises for clean sets. They used noise rate dependent hyperparameters in CIFAR10 case. They used a subset of Clothing1M by claiming there is a class imbalance problem in the dataset.

### 6 - Combating noisy labels by agreement: A joint training method with co-regularization

### 7 - Label Noise Types and Their Effects on Deep Learning

### 8 - Identifying Mislabeled Data using the Area Under the Margin Ranking

### 9 - DivideMix: Learning with Noisy Labels as Semi-supervised Learning

### 10 - Boosting Co-teaching with Compression Regularization for Label Noise

### 11 - LEARNING WITH FEATURE-DEPENDENT LABEL NOISE: A PROGRESSIVE APPROACH


## Some Classes

### Robuts Loss Functions

1 - Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon,Richard Nock, and Lizhen Qu. Making deep neural networksrobust to label noise: A loss correction approach. InCVPR,pages 1944–1952, 2017.

2 - Aritra Ghosh, Himanshu Kumar, and P. S. Sastry.  Robustloss functions under label noise for deep neural networks. InAAAI, pages 1919–1925, 2017.

3 - Zhilu Zhang and Mert R. Sabuncu. Generalized cross entropyloss for training deep neural networks with noisy labels.  InNIPS, 2018.


## Some References

- P. Welinder, S. Branson, T. Mita, C. Wah, F. Schroff, S. Be-longie, and P. Perona. Caltech-UCSD Birds 200. TechnicalReport CNS-TR-2010-001, California Institute of Technol-ogy, 2010.

    Expert knowledge is necessary for some datasets such as the fine-grained CUB-200, which demands knowledge from ornithologists.


- Robert Fergus, Fei-Fei Li, Pietro Perona, and Andrew Zisser-man. Learning object categories from Internet image searches.Proceedings of the IEEE, 98(8):1453–1466, 2010
- Jonathan Krause, Benjamin Sapp, Andrew Howard, HowardZhou, Alexander Toshev, Tom Duerig, James Philbin, andLi Fei-Fei. The unreasonable effectiveness of noisy data forfine-grained recognition.  InECCV, volume 9907 ofLNCS,pages 301–320. Springer, 2016.
- Sainbayar Sukhbaatar, Joan Bruna, Manohar Paluri, LubomirBourdev, and Rob Fergus. Training convolutional networkswith noisy labels.arXiv preprint arXiv:1406.2080, 2014.
    
    We can easily collect a large scale dataset with noisy annotations through image search engines


- Tong Xiao, Tian Xia, Yi Yang, Chang Huang, and XiaogangWang. Learning from massive noisy labeled data for image classification. InCVPR, pages 2691–2699, 2015.
    
    Noisy annotations canbe obtained by extracting labels from the surrounding textsor using the searching keywords


- Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht,and Oriol Vinyals. "Understanding deep learning requiresrethinking generalization". InICLR, 2017
    
    Shows that networks can memorize even completly random labels, so noise effect is an important problem for highly parametrized networks.


- Jan Larsen, Lars Nonboe Andersen, Mads Hintz-Madsen, andLars Kai Hansen. Design of robust neural network classifiers.InICASSP, pages 1205–1208, 1998.

    Modelling of symmetric noise.


- Sainbayar Sukhbaatar, Joan Bruna, Manohar Paluri, LubomirBourdev, and Rob Fergus. Training convolutional networkswith noisy labels.arXiv preprint arXiv:1406.2080, 2014.

    Modelling of asymmetric noise.

- Carla E. Brodley and Mark A. Friedl. Identifying mislabeledtraining data.J. Artif. Intell. Res., 11:131–167, 1999.

    Delete unreliable samples.

- Isabelle Guyon, Nada Matic, and Vladimir Vapnik. Discov-ering informative patterns and data cleaning. InKDD, pages181–203, 1996.

    Hard-diffucult samples are important for network accuracy.

