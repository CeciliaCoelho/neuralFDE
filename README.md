# **Neural Fractional Differential Equations; C. Coelho, M. Fernanda P. Costa, L.L. Ferrás**

## Papers associated:
- ["Tracing Footprints: Neural Networks Meet Non-integer Order Differential Equations For Modelling Systems with Memory", Tiny Papers @ ICLR 2024](https://openreview.net/forum?id=8518dcW4hc&referrer=%5Bthe%20profile%20of%20C.%20Coelho%5D(%2Fprofile%3Fid%3D~C._Coelho2))
- ["Neural Fractional Differential Equations", arxiv preprint](https://arxiv.org/abs/2403.02737)
- ["Neural fractional differential equations: Optimising the order of the fractional derivative. Fractal and Fractional, 8(9), 2024"](https://www.mdpi.com/2504-3110/8/9/529)


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
