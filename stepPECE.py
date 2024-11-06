from scipy import special
import torch
import sys
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

def solve(a, f, y0, t, h=0):
    y_out = torch.zeros((1,1))
    h = h.to(device)
    for i in range(len(t)-1):
        t0 = t[0].to(device)
        if h.to(device) == torch.Tensor([0]).to(device): h = (t[i+1] - t0).to(device)
        N = math.ceil(((t[i+1] - t0) / h.to(device)).item())

        y = torch.zeros((1,1))
        tspan = torch.arange(t0.item(), t[i+1].item() + h.item(), h.item()).unsqueeze(1).to(device)

        for n in range(0, N):
            temp = torch.zeros(1, dtype=torch.float, requires_grad=True).to(device)

            for k in range(0, int(torch.ceil(a))):
                temp.data += y0 * (tspan[n])**k / math.factorial(k)

            y_p = predictor(f, y, a, n, h, y0, t0, tspan)
            y_new = temp + h**a / math.gamma(a+2) *(f(tspan[n], y_p) + right(f, y, a, n, h)).unsqueeze(1)

            if n == 0: 
                y = y_new
            else:
                y = torch.cat([y, y_new], dim=0)
        
        y = linearInterp(t0, tspan[-1], y0, y_new, t[i+1])
        if i == 0:
            y_out = y[-1]
        else:
            y_out = torch.cat([y_out, y[-1]], dim=0)


    return y_out

def right(f, y, a, n, h):
    temp = torch.zeros((1)).to(device) 
    for j in range(0, n+1):
        if j == n:
            temp += A(j, n, a) * f((j*h).to(device), torch.zeros((1)).to(device))
        else:
            temp += A(j, n, a) * f((j*h).to(device), y[j].to(device))

    return temp

def A(j, n, a):
    if j == 0:
        return n**(a + 1) - (n - a) * (n + 1)**a
    elif 1 <= j <= n:
        return (n - j + 2)**(a + 1) + (n - j)**(a + 1) - 2*(n - j + 1)**(a+1)
    elif j == n + 1:
        return 1

def predictor(f, y, a, n, h, y0, t0, tspan):
    predict = torch.zeros((1)).to(device) 
    leftsum = 0.
    l = torch.ceil(a)
    for k in range(0, int(l)):
        leftsum += y0 * (tspan[n])**k / math.factorial(k)

    for j in range(0, n+1):
        if j == n: 
            predict += torch.mul(B(j, n, a), f(tspan[n], torch.zeros((1)).to(device)))
        else:
            predict += torch.mul(B(j, n, a), f(tspan[n], y[j].to(device)))

    return leftsum.add(torch.mul(h**a / a, predict))

def B(j, n, a):
    return ((n + 1 - j)**a - (n - j)**a)

def linearInterp(t0, t1, y0, y1, t):
    if t == t0:
        return y0
    elif t == t1:
        return y1
    else:
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)

