# Deep Learning with Noisy Labelled Data Literature

|Year|Conf|Repo|Title|
|----|----|----|-----|
|2017|||[Decoupling "when to update" from "how to update"](https://arxiv.org/abs/1706.02613)|
|2018|CVPR|[Pt](https://github.com/kuanghuei/clean-net)|[CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise](https://arxiv.org/abs/1711.07131)|
|2018||[Pt](https://github.com/bhanML/Co-teaching)|[Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels](https://arxiv.org/abs/1804.06872)|
|2019||[Pt](https://github.com/xingruiyu/coteaching_plus)|[How does Disagreement Help Generalization against Label Corruption?](https://arxiv.org/abs/1901.04215)|
|2019|||[Deep Self-Learning From Noisy Labels](https://arxiv.org/abs/1908.02160)|
|2019|CVPR||[Probabilistic End-to-end Noise Correction for Learning with Noisy Labels](https://arxiv.org/abs/1903.07788)|
|2020|ICLR||[SELF: Learning to Filter Noisy Labels with Self-Ensembling](https://arxiv.org/abs/1910.01842)|
|2020|||[Combating noisy labels by agreement: A joint training method with co-regularization](https://arxiv.org/abs/2003.02752)|
|2020|||[Label Noise Types and Their Effects on Deep Learning](https://arxiv.org/abs/2003.10471)|
|2020|||[Identifying Mislabeled Data using the Area Under the Margin Ranking](https://arxiv.org/abs/2001.10528)|
|2020|ICLR|[Pt](https://github.com/LiJunnan1992/DivideMix)|[DivideMix: Learning with Noisy Labels as Semi-supervised Learning](https://arxiv.org/abs/2002.07394)|
|2021|CVPR|[Pt](https://github.com/yingyichen-cyy/Nested-Co-teaching)|[Boosting Co-teaching with Compression Regularization for Label Noise](https://arxiv.org/abs/2104.13766)|
|2021|ICLR|[Pt](https://github.com/pxiangwu/PLC)|[LEARNING WITH FEATURE-DEPENDENT LABEL NOISE: A PROGRESSIVE APPROACH](https://arxiv.org/abs/2103.07756)|
|2021|CVPR|[Pt](https://github.com/KentoNishi/Augmentation-for-LNL)|[Augmentation Strategies for Learning with Noisy Labels](https://arxiv.org/abs/2103.02130)|

## Summaries

### 1 - Decoupling "when to update" from "how to update"
Uses two networks. Looks disagreement region for updates. Does not requires any clean subset. Uses Large Faces in the Wild (LWF) and MNIST datasets in experiments.
Disagreement region always contains noisy samples and network start to memorizes these samples in later iterations of training.

### 2 - CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise

### 3 - Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels
Uses two networks. Updates is done with small loss instances of peer network. Does not require any clean subset. Assumes that noise ratio is known. Uses MNIST, CIFAR10, CIFAR100 dasets in experiments. Apply symmetric and asymmetric syhentetic noises in experiments.

### 4 - How does Disagreement Help Generalization against Label Corruption?
Uses two networks. Updates is done with small loss instances of peer network which are in the disagreement region. It can be viewed as combination of Decoupling and Co-teaching. Does not require any clean subset. Assumes that noise ration is known. Uses MNIST, CIFAR10, CIFAR100, NEW and T-ImageNet datasets in experiments. Apply symmetric and asymmetric syhentetic noise.

### 5 - Deep Self-Learning From Noisy Labels
Uses single network. There is an iterative approach which consists training and label correction phases. Initially, a network starts its the first epoch using noisy label. Then, features of samples are extracted with that network. Using cosine similarity, similiarity between samples inside each class is calculated(it is classwise). Then, a similarity metric is defined to extract samples which are possibly clean and hard. For each class, paper extracts multiple prototypes for each class using this similarity metric. Then, each samples compared with prototypes to create a class score for each samples. The class where class score is maxiumum is assigned to sample as new-corrected label.

They used Food101-N and Clothing1M datasets in their experiments. There is not any clean subset or noise rate assumption before training.

### 6 - Probabilistic End-to-end Noise Correction for Learning with Noisy Labels
Uses a single network. Try to model image labels as probabilistic distrubitions in soft-label space. Initially, train a neural network with noisy labels using classical cross entropy loss. At this step, learning rate is set to high value and they claim that high learning rate prevents overfitting to noisy labels. Then, soft labels are initialized with noisy labels. In the second step, they optimize both network parameters and soft labels to decrease loss value. At this step, loss function includes three terms: classification loss, compatibility loss and entropy loss as a regulazir. Classification loss is a modified version of KL divergence. Compatibility loss is cross entropy between soft labels and initial noisy labels. Entropy loss is applied to output of the network to avoid flat predictions. In the third step, they fine tune networks parameters over optimized soft labels using a small learning rate.

They used CIFAR10, CIFAR100, CUB-200 and Clothing1M datesets in experiments. They used symmetric and asymmetric syhentetic noises for clean sets. They used noise rate dependent hyperparameters in CIFAR10 case. They used a subset of Clothing1M by claiming there is a class imbalance problem in the dataset.

### 7 - SELF: Learning to Filter Noisy Labels with Self-Ensembling

Uses two networks but in concept of mean teacher network. There is an student network and teacher network. Student network is trained over current dataset and then, teacher network is updated as moving average of student network. Student network is trained with labelled data using classification loss, while unlabelled data is used for consistency loss between student and teacher network predictions.
Here, paper idea is detecting possible noisy samples during training using an ensembling approach. At initial, all samples are assigned as possible clean samples. Network is trained over. Then, each sample prediction is used in moving average calculation. If the maxima of moving average prediction of a sample is consistent with its ground truth, it is tagged as clean for current iteration. In this way, a possible clean set is constructed.
In the next iteration, network is again trained over possible clean set. If the network accuracy over a validation set is higher than best network result, previous filtering process is repeated to construct a new possible clean set. This continues till the convergence of approach. 
Possible noisy samples are used in the consistency loss of mean teach network training.

Paper uses CIFAR10, CIFAR100 and ImageNet datasets in experiments. They used symmetric and asymmetric syhentetic noises for clean sets. There is not any clean subset or noise rate assumption before training.

### 8 - Combating noisy labels by agreement: A joint training method with co-regularization

### 9 - Label Noise Types and Their Effects on Deep Learning

### 10 - Identifying Mislabeled Data using the Area Under the Margin Ranking

### 11 - DivideMix: Learning with Noisy Labels as Semi-supervised Learning

Uses two networks. Initially, networks are warming up using cross entropy loss with penalty for confident predictions by adding a negative cross entropy term. They claim that adding this penalty term, makes clean and noisy samples more distinguisable during analyzing their loss. Then, for each training epoch, algorithm first uses GMM to model per-sample loss with each of the two networks. Using this and a clean probability threshold, the network then categorizes samples into a labelled set and an unlabelled set. Batches are pulled from each of these two sets and are first augmented. Predictions using the augmented samples are made and a sharpening function is applied to output to reduce entropy of label distribution. For labelled set, weighted sum of original label and network predictions are taken while for unlabelled set average of the two networks are taken before sharpening. Then, we obtained two new sets and these are fed to MixMatch algorithm.

### 12 - Boosting Co-teaching with Compression Regularization for Label Noise
Uses two networks. They apply two stages training. In the first step, they trained two networks using nested dropout approach. Nested dropout create an output at the end of network with importance oredered.  Then, they choose first k entry of this importance ordered output. They, fine tune their network using these k entry of output with Co-teaching algorithm. The first stage provides a reliable base for Co-teaching algorithm.

They used Food101-N and Clothing1M datasets in their experiments. They takes 260k images from Clothing1M dataset which is balanced. There is not any clean dataset or noise rate assumption before training.
### 13 - LEARNING WITH FEATURE-DEPENDENT LABEL NOISE: A PROGRESSIVE APPROACH

### 14 - Augmentation Strategies for Learning with Noisy Labels


## Some Classes

### Robuts Loss Functions

1 - Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon,Richard Nock, and Lizhen Qu. Making deep neural networksrobust to label noise: A loss correction approach. InCVPR,pages 1944–1952, 2017.

2 - Aritra Ghosh, Himanshu Kumar, and P. S. Sastry.  Robustloss functions under label noise for deep neural networks. InAAAI, pages 1919–1925, 2017.

3 - Zhilu Zhang and Mert R. Sabuncu. Generalized cross entropyloss for training deep neural networks with noisy labels.  InNIPS, 2018.

### Sample Selection
1 -  Bo  Han,  Quanming  Yao,  Xingrui  Yu,  Gang  Niu,  MiaoXu,  Weihua Hu,  Ivor Tsang,  and Masashi Sugiyama.   Co-teaching:  Robust training of deep neural networks with ex-tremely noisy labels. InNeurIPS, pages 8535–8545, 2018.

2 - Eran Malach and Shai Shalev-Shwartz. Decoupling “when to update” from “how to update”. In NIPS, 2017.

3 - Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In ICML, 2018.

4 - Pengfei Chen, Ben Ben Liao, Guangyong Chen, and Shengyu Zhang. Understanding and utilizing deep neural networks trained with noisy labels. In ICML, 2019.

### Loss Correction
1 - Daiki Tanaka,  Daiki Ikami,  Toshihiko Yamasaki,  and Kiy-oharu  Aizawa.   Joint  optimization  framework  for  learning with noisy labels. In Proceedings of the IEEE Conferenceon Computer Vision and Pattern Recognition, pages 5552–5560, 2018.

2 - Scott E. Reed, Honglak Lee, Dragomir Anguelov, Christian Szegedy, Dumitru Erhan, and Andrew Rabinovich. Training deep neural networks on noisy labels with bootstrapping. In ICLR, 2015.

3 - Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu. Making deep neural networks robust to label noise: a loss correction approach. In CVPR, 2017.

4 - Jacob Goldberger and Ehud Ben-Reuven. Training deep neural-networks using a noise adaptation layer. In ICLR, 2017.

5 - Xingjun Ma, Yisen Wang, Michael E. Houle, Shuo Zhou, Sarah M. Erfani, Shu-Tao Xia, Sudanthi Wijewickrema, and James Bailey. Dimensionality-driven learning with noisy labels. In ICML, 2018.

## Some References

- P. Welinder, S. Branson, T. Mita, C. Wah, F. Schroff, S. Be-longie, and P. Perona. Caltech-UCSD Birds 200. TechnicalReport CNS-TR-2010-001, California Institute of Technol-ogy, 2010.

    Expert knowledge is necessary for some datasets such as the fine-grained CUB-200, which demands knowledge from ornithologists.


- Robert Fergus, Fei-Fei Li, Pietro Perona, and Andrew Zisser-man. Learning object categories from Internet image searches.Proceedings of the IEEE, 98(8):1453–1466, 2010
- Jonathan Krause, Benjamin Sapp, Andrew Howard, HowardZhou, Alexander Toshev, Tom Duerig, James Philbin, andLi Fei-Fei. The unreasonable effectiveness of noisy data forfine-grained recognition.  InECCV, volume 9907 ofLNCS,pages 301–320. Springer, 2016.
- Sainbayar Sukhbaatar, Joan Bruna, Manohar Paluri, LubomirBourdev, and Rob Fergus. Training convolutional networkswith noisy labels.arXiv preprint arXiv:1406.2080, 2014.
    
    We can easily collect a large scale dataset with noisy annotations through image search engines


- Tong Xiao, Tian Xia, Yi Yang, Chang Huang, and XiaogangWang. Learning from massive noisy labeled data for image classification. InCVPR, pages 2691–2699, 2015.
    
    Noisy annotations canvbe obtained by extracting labels from the surrounding texts or using the searching keywords


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

-  Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc VLe. Self-training with noisy student improves imagenet clas-sification.  InProceedings of the IEEE/CVF Conference onComputer  Vision  and  Pattern  Recognition,  pages  10687–10698, 2020

    Data augmentation in classification

- Kihyuk  Sohn,  Zizhao  Zhang,  Chun-Liang  Li,  Han  Zhang,Chen-Yu Lee, and Tomas Pfister.  A simple semi-supervisedlearning  framework  for  object  detection.arXiv  preprintarXiv:2005.04757, 2020.

    Data augmentation in object detection

- Devansh Arpit, Stanisław Jastrz ̨ebski, Nicolas Ballas, DavidKrueger,  Emmanuel  Bengio,  Maxinder  S.  Kanwal,  TeganMaharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, andSimon Lacoste-Julien. A closer look at memorization in deepnetworks. InICML, 2017.

    correctly labeled data fit before incorrectly labeled data as discovered

- Eric Arazo,  Diego Ortego,  Paul Albert,  Noel E O’Connor,and Kevin McGuinness.  Unsupervised label noise modelingand loss correction.arXiv preprint arXiv:1904.11238, 2019.
- Junnan Li, Richard Socher, and Steven CH Hoi.  Dividemix:Learning  with  noisy  labels  as  semi-supervised  learning.arXiv preprint arXiv:2002.07394, 2020.

    Mix-up augmentation in noise labelled problem

- Eric Arazo, Diego Ortego, Paul Albert, Noel E O’Connor, and Kevin McGuinness. Unsupervised label noise modeling and loss correction. arXiv preprint arXiv:1904.11238, 2019.
- Junnan Li, Richard Socher, and Steven CH Hoi. Dividemix: Learning with noisy labels as semi-supervised learning. arXiv preprint arXiv:2002.07394, 2020.
- Xingrui Yu, Bo Han, Jiangchao Yao, Gang Niu, Ivor W Tsang, and Masashi Sugiyama. How does disagreement help generalization against label corruption? arXiv preprint arXiv:1901.04215, 2019.

    Warm-up for loss investigation of noisy labelled datasets