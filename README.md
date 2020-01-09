## Universal Adversarial Audio Perturbations (UAP)
Python code for Universal adversarial audio perturbations [1] generation.
The target model which is used here is the combination of Sincnet [2] +VGG19. [Keras implementation](https://github.com/grausof/keras-sincnet) of SincNet (M. Ravanelli - Y. Bengio) [2] is used. The model is trained on [UrbanSound8k](https://urbansounddataset.weebly.com/urbansound8k.html) dataset [3]. For more information on data normalization please refer to our paper [1].

Two methods are used for UAP generation The first method is based on an iterative, greedy approach that is well-known in computer vision [4]: it aggregates small perturbations to the input so as to push it to the decision boundary. The second method, which is the main contribution of our paper [1], is a novel penalty formulation, which finds targeted and untargeted universal adversarial perturbations.

## Prerequisites
- tensorflow-gpu>=1.12.0
- keras>=2.2.4
- pysoundfile (``` pip install pysoundfile```)
- numpy
- pickle
- sklearn

## References

[1] Abdoli, Sajjad, et al. "Universal adversarial audio perturbations." [Arxiv](https://arxiv.org/pdf/1908.03173.pdf) preprint arXiv:1908.03173 (2019).

[2] Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” [Arxiv](http://arxiv.org/abs/1808.00158)

[3] Justin Salamon, Christopher Jacoby, and Juan Pablo Bello. "A dataset and taxonomy for urban sound research." Proceedings of the 22nd ACM international conference on Multimedia. ACM, 2014. 

[4] Moosavi-Dezfooli, Seyed-Mohsen, et al. "Universal adversarial perturbations." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
