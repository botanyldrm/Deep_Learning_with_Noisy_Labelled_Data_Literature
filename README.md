# Deep Learning with Noisy Labelled Data Literature

|Year|Conf|Repo|Title|
|----|----|----|-----|
|2017|||[Decoupling "when to update" from "how to update"](https://arxiv.org/abs/1706.02613)|
|2018||[Pt](https://github.com/bhanML/Co-teaching)|[Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels](https://arxiv.org/abs/1804.06872)|
|2019||[Pt](https://github.com/xingruiyu/coteaching_plus)|[How does Disagreement Help Generalization against Label Corruption?](https://arxiv.org/abs/1901.04215)|
|2019|||[Deep Self-Learning From Noisy Labels](https://arxiv.org/abs/1908.02160)|
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

### 5 - Combating noisy labels by agreement: A joint training method with co-regularization

### 6 - Label Noise Types and Their Effects on Deep Learning

### 7 - Identifying Mislabeled Data using the Area Under the Margin Ranking

### 8 - DivideMix: Learning with Noisy Labels as Semi-supervised Learning

### 9 - Boosting Co-teaching with Compression Regularization for Label Noise

### 10 - LEARNING WITH FEATURE-DEPENDENT LABEL NOISE: A PROGRESSIVE APPROACH