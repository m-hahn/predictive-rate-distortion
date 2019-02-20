
import torch

with open("logBeta_EE.csv", "r") as inFile:
    data = [x.split(",")[1:] for x in inFile.read().strip().split("\n")[1:]]
    xs = torch.FloatTensor([float(x[0]) for x in data]) # log(beta)
    ys = torch.FloatTensor([float(x[1]) for x in data]) # EE

alpha = torch.FloatTensor([1.0])
beta = torch.FloatTensor([-1.0])
gamma = torch.FloatTensor([1.1])
delta = torch.FloatTensor([0.0])

alpha.requires_grad = True
beta.requires_grad = True
gamma.requires_grad = True
delta.requires_grad = True

optim = torch.optim.Adam([alpha, beta, gamma, delta], lr=0.001)
lossModule = torch.nn.MSELoss()
for i in range(100000):
    optim.zero_grad()
    predicted = alpha * torch.pow(xs, 0.0) * (torch.pow(beta+xs, -gamma)) + 1
    loss = lossModule(predicted  , ys).mean()
    loss.backward()
    optim.step()
    if i % 10 == 0:
        print(loss, [float(x) for x in [alpha, beta, gamma, delta]])

