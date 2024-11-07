# **Neural Fractional Differential Equations; C. Coelho, M. Fernanda P. Costa, L.L. Ferrás**

## Papers associated:
- ["Tracing Footprints: Neural Networks Meet Non-integer Order Differential Equations For Modelling Systems with Memory", Tiny Papers @ ICLR 2024](https://openreview.net/forum?id=8518dcW4hc&referrer=%5Bthe%20profile%20of%20C.%20Coelho%5D(%2Fprofile%3Fid%3D~C._Coelho2))
- ["Neural Fractional Differential Equations", arxiv preprint](https://arxiv.org/abs/2403.02737)
- ["Neural fractional differential equations: Optimising the order of the fractional derivative. Fractal and Fractional, 8(9), 2024"](https://www.mdpi.com/2504-3110/8/9/529)

### **If you use this code, please cite the papers above, respectively as:**

```
@inproceedings{Y_neuralFDE,
  author       = {C. Coelho and
                  M. Fernanda P. Costa and
                  L.L. Ferr{\'{a}}s},
  title        = {Tracing Footprints: Neural Networks Meet Non-integer Order Differential
                  Equations For Modelling Systems with Memory},
  booktitle    = {The Second Tiny Papers Track at {ICLR} 2024, Tiny Papers @ {ICLR}
                  2024, Vienna, Austria, May 11, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=8518dcW4hc},
  timestamp    = {Fri, 26 Jul 2024 13:58:33 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/0002CF24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```
@article{Y_coelho2024neuralFDEPreprint,
  title={Neural Fractional Differential Equations},
  author={C. Coelho and M. Fernanda P. Costa and L.L. Ferr{\'a}s},
  journal={arXiv preprint arXiv:2403.02737},
  year={2024},
  url={https://arxiv.org/pdf/2403.02737}
}
```

```
@article{Y_alphaNN,
  AUTHOR = {Coelho, C. and Costa, M. Fernanda P. and Ferrás, L.L.},
TITLE = {Neural Fractional Differential Equations: Optimising the Order of the Fractional Derivative},
JOURNAL = {Fractal and Fractional},
VOLUME = {8},
YEAR = {2024},
NUMBER = {9},
ARTICLE-NUMBER = {529},
URL = {https://www.mdpi.com/2504-3110/8/9/529},
ISSN = {2504-3110},
ABSTRACT = {Neural Fractional Differential Equations (Neural FDEs) represent a neural network architecture specifically designed to fit the solution of a fractional differential equation to given data. This architecture combines an analytical component, represented by a fractional derivative, with a neural network component, forming an initial value problem. During the learning process, both the order of the derivative and the parameters of the neural network must be optimised. In this work, we investigate the non-uniqueness of the optimal order of the derivative and its interaction with the neural network component. Based on our findings, we perform a numerical analysis to examine how different initialisations and values of the order of the derivative (in the optimisation process) impact its final optimal value. Results show that the neural network on the right-hand side of the Neural FDE struggles to adjust its parameters to fit the FDE to the data dynamics for any given order of the fractional derivative. Consequently, Neural FDEs do not require a unique α value; instead, they can use a wide range of α values to fit data. This flexibility is beneficial when fitting to given data is required and the underlying physics is not known.},
DOI = {10.3390/fractalfract8090529}
}
```


Having in mind the concepts of Neural ODEs and the role of fractional calculus in Neural systems , we extend Neural ODEs to Neural FDEs. These feature a Caputo fractional derivative of order $\alpha$ (with $0<\alpha<1$) on the left-hand side, representing the variation of the state of a dynamical system, and a NN ($\boldsymbol{f_\theta}$) on the right-hand side. When $\alpha=1$, we recover the Neural ODE. The parameter $\alpha$ is learnt by another NN, $\alpha_{\boldsymbol{\phi}}$, with parameters $\boldsymbol{\phi}$. This approach allows the Neural FDE to adjust to the training data independently, enabling it to find the best possible fit without user intervention.

During the Neural FDE training, we employ a specialised numerical solver for FDEs to compute the numerical solution. Subsequently, a loss function is used to compare the numerically predicted outcomes with the ground truth. Using autograd for backpropagation, we adjust the weights and biases within the NNs $\boldsymbol{f_\theta}$ and $\alpha_{\boldsymbol{\phi}}$ to minimise the loss.

### **Usage**

The Predictor-Corrector Fractional Numerical Method is available at ```PECE.py``` and at ```stepPECE.py``` if you desire to choose a time step different than the data points (see the [paper](https://arxiv.org/abs/2403.02737) for an explanation). It can be used to solve FDE Initial Value Problems. This can be used as a standalone as follows, for a simple example of solving $\dfrac{d^{\alpha}y(x)}{dt^{\alpha}}=1-x$:

```python
import torch
from PECE import *

true_y0 = torch.tensor([0.518629]).to(device) # Initial condition
t = torch.linspace(0., args.tf, args.data_size) # Discretised time
alpha = 0.9 # Fractional order
step_size = 0.01 # Step size

def f(t,x): # Right hand side of the FDE
    return 1-x

solutions = solve(torch.Tensor([alpha]).to(device), f, ic, t, h=torch.Tensor([step_size]))
```

A Neural FDE example is available at ```neuralFDE.py``` using the stepPECE. To use it:

```
python neuralFDE.py
```

Using the options you can choose to use a ODE or FDE generated dataset, the alpha-order value and some NN options.
