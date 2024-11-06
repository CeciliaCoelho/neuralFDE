import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse, time
import scipy

from torchdiffeq import odeint_adjoint as odeint
from stepPECE import * 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser('demo')
parser.add_argument('--diffeq', type=str, choices=['FDE', 'ODE'], default='FDE')
parser.add_argument('--dataset', type=str, choices=['population'], default='population')
parser.add_argument('--alpha', type=float, default=0.99)
parser.add_argument('--data_size', type=int, default=300)
parser.add_argument('--t_length', type=int, default=200)
parser.add_argument('--data_size_test', type=int, default=400)
parser.add_argument('--t_length_test', type=int, default=300)
parser.add_argument('--niters', type=int, default=200)
parser.add_argument('--step_size', type=float, default=0)
parser.add_argument('--savePlot', type=str)
parser.add_argument('--saveModel', type=str)
args = parser.parse_args()

np.random.seed(0)


t = torch.as_tensor(np.linspace(0,args.t_length,args.data_size)).unsqueeze(1).to(device)
t_test_rec = torch.as_tensor(np.linspace(0,args.t_length,args.data_size)).unsqueeze(1).to(device)
t_test_ex = torch.as_tensor(np.linspace(0,args.t_length_test,args.data_size)).unsqueeze(1).to(device)
t_test_c = torch.as_tensor(np.linspace(0,args.t_length,args.data_size_test)).unsqueeze(1).to(device)


if args.step_size == 0: args.step_size = t[1]-t[0]

#################FDE###################################
with torch.no_grad():
        
    ic = 100


    if args.alpha == 1: #use ODE solver to create training and testing sets
        class f(nn.Module):
            def forward(self, t, x):
                return 10*x*(1-x/1000)

        print("using ODE solver")
        DF = odeint(f(), torch.tensor([ic]).float(), t.squeeze(1), method="euler", options=dict(step_size=args.step_size)).to(device)
        DF = DF/max(DF)
        DF = torch.tensor(DF, dtype=torch.float64)



        ##RECONSTRUCTION EXPERIMENT##
        DF_test_rec = odeint(f(), torch.tensor([ic]).float(), t_test_rec.squeeze(1), method="euler", options=dict(step_size=args.step_size)).to(device)
        DF_test_rec = DF_test_rec / max(DF_test_rec)
        ##EXTRAPOLATION EXPERIMENT##
        DF_test_ex = odeint(f(), torch.tensor([ic]).float(), t_test_ex.squeeze(1), method="euler", options=dict(step_size=args.step_size)).to(device)
        DF_test_ex = DF_test_ex / max(DF_test_ex)

    else: #use FDE solver to create training and testing sets
        def f(t,x):
            return 0.1*x*(1-x/1000) #alpha=0.8 ic=100

        print("using FDE solver")
        DF = solve(torch.Tensor([args.alpha]).to(device), f, ic, t, h=torch.Tensor([args.step_size])).to(device)
        DF = DF/max(DF)

        ##RECONSTRUCTION EXPERIMENT##
        DF_test_rec = solve(torch.Tensor([args.alpha]).to(device), f, ic, t_test_rec, h=torch.Tensor([args.step_size])).to(device)
        DF_test_rec = DF_test_rec / max(DF_test_rec)
        ##EXTRAPOLATION EXPERIMENT##
        DF_test_ex = solve(torch.Tensor([args.alpha]).to(device), f, ic, t_test_ex, h=torch.Tensor([args.step_size])).to(device)
        DF_test_ex = DF_test_ex / max(DF_test_ex)


class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.l1 = nn.Linear(1, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64,1)
        self.tanh = nn.Tanh()

    def forward(self, t, x):
        x = self.l1(x)
        x = self.tanh(x)
        x = self.l2(x)
        x = self.tanh(x)
        x = self.l2(x)
        x = self.tanh(x)
        out = self.l3(x)

        return out


class alphaNN(nn.Module):
    def __init__(self):
        super(alphaNN, self).__init__()
        self.l1 = nn.Linear(1, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.tanh(x)
        x = self.l2(x)
        x = self.tanh(x)
        x = self.l3(x)
        out = self.sigmoid(x)

        return out

func = ODEFunc().to(device)
if args.diffeq == "FDE":
    alphaFunc = alphaNN().to(device)
    params = list(func.parameters()) + list(alphaFunc.parameters())
else:
    params = list(func.parameters())

optimizer = optim.Adam(params, lr=1e-3, foreach=False)

iters = args.niters 
alpha = torch.Tensor([0.99]).to(device)


t_training = time.process_time()

if args.alpha != 1: 
    DF = DF.unsqueeze(1)
    DF_test_rec = DF_test_rec.unsqueeze(1)
    DF_test_ex = DF_test_ex.unsqueeze(1)

if args.alpha == 1 and args.diffeq == "FDE":
    DF = DF[:-1]
    DF_test_rec = DF_test_rec[:-1]
    DF_test_ex = DF_test_ex[:-1]

for ii in range(1, iters+1):
        optimizer.zero_grad()

        if args.diffeq == "FDE":
            alpha_pred = alphaFunc(alpha).to(device)
            pred = solve(alpha_pred, func, DF[0], t, h=torch.Tensor([args.step_size])).unsqueeze(1).to(device)
        else:
            if args.alpha != 1: 
                pred = odeint(func, DF[0], t.squeeze(1), method="euler", options=dict(step_size=args.step_size))[:-1].to(device)
            else:
                pred = odeint(func, DF[0], t.squeeze(1), method="euler", options=dict(step_size=args.step_size)).to(device)

        loss = nn.MSELoss()(pred, DF) 
        loss.backward(retain_graph=True)
        optimizer.step()

        if ii % 1 == 0:
            if args.diffeq == "FDE":
                print("Iter: {:4d} | Loss: {:.6f} | alpha={:.4f}".format(ii, loss.item(), alpha_pred.item()))
            else:
                print("Iter: {:4d} | Loss: {:.6f}".format(ii, loss.item()))


        if ii == iters:
            elapsed_training_time = time.process_time() - t_training
            t_testing = time.process_time()

            with torch.no_grad():
                if args.diffeq == "FDE":
                    pred_test_rec = solve(alpha_pred, func, DF[0], t_test_rec, h=torch.Tensor([args.step_size])).unsqueeze(1).to(device)
                    pred_test_ex = solve(alpha_pred, func, DF[0], t_test_ex, h=torch.Tensor([args.step_size])).unsqueeze(1).to(device)
                else:
                    if args.alpha != 1 and args.diffeq == "ODE":
                        pred_test_rec = odeint(func, DF[0], t_test_rec.squeeze(1), method="euler", options=dict(step_size=args.step_size))[:-1].to(device)
                        pred_test_ex = odeint(func, DF[0], t_test_ex.squeeze(1), method="euler", options=dict(step_size=args.step_size))[:-1].to(device)
                    else:
                        pred_test_rec = odeint(func, DF[0], t_test_rec.squeeze(1), method="euler", options=dict(step_size=args.step_size)).to(device)
                        pred_test_ex = odeint(func, DF[0], t_test_ex.squeeze(1), method="euler", options=dict(step_size=args.step_size)).to(device)

                loss_test_rec = nn.MSELoss()(pred_test_rec, DF_test_rec) 
                loss_test_ex = nn.MSELoss()(pred_test_ex, DF_test_ex) 
                print("Reconstruction Test Loss: {:.6f}".format(loss_test_rec.item()))
                print("Extrapolation Test Loss: {:.6f}".format(loss_test_ex.item()))

            elapsed_testing_time = time.process_time() - t_testing
            print("Training time:", elapsed_training_time)
            print("Testing time:", elapsed_testing_time)
            torch.save(func, args.saveModel+".pt")


            names = ["reconstruction", "extrapolation"]
            true = [DF_test_rec, DF_test_ex]
            pred = [pred_test_rec, pred_test_ex]
            for i in range(len(names)):
                plt.figure()
                t_test = range(len(true[i]))
                if args.alpha != 1 and args.diffeq == "ODE":
                    #don't show the initial condition 
                    plt.plot(t_test, true[i].cpu().detach().numpy(), label='ground-truth', linestyle='dashed', color='black')
                    plt.plot(t_test, pred[i].cpu().detach().numpy(), label='prediction', color='blue')
                else:
                    #don't show the initial condition 
                    plt.plot(t_test, true[i].cpu().detach().numpy(), label='ground-truth', linestyle='dashed', color='black')
                    plt.plot(t_test, pred[i].cpu().detach().numpy(), label='prediction', color='blue')
                #plt.legend()
                #plt.savefig(args.savePlot+"_"+names[i]+".pdf")
                plt.show()



