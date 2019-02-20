
import torch

with open("neural-pos.csv", "r") as inFile:
    data = [x.split(",")[1:] for x in inFile.read().strip().split("\n")[1:]]
    xs = torch.FloatTensor([float(x[1]) for x in data]) # MI With Future
    ys = torch.FloatTensor([float(x[0]) for x in data]) # Memory

alpha = torch.FloatTensor([1.0])
beta = torch.FloatTensor([1.0])
gamma = torch.FloatTensor([1.0])
alpha.requires_grad = True
beta.requires_grad = True
gamma.requires_grad = True

optim = torch.optim.Adam([alpha, beta, gamma], lr=0.0001)
lossModule = torch.nn.MSELoss()
for i in range(100000):
    optim.zero_grad()
    predicted = alpha * (1-torch.pow(ys.div(beta), -gamma))
    loss = lossModule(predicted  , xs).mean()
    loss.backward()
    optim.step()
    if i % 10 == 0:
        print(loss, [float(x) for x in [alpha, beta, gamma]])

