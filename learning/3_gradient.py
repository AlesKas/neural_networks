from platform import win32_edition
import torch

x = torch.rand(3, requires_grad=True)
print(x)

y = x + 2
print(y)
z = y*y*2

# Pokud neudělám ze Z skalár, gradient se nespočítá
#z = z.mean()
print(z)

# Můžu ale přidat vektor stejné délky, a to pak spočítá gradient
v = torch.tensor([1.1, 0.0, 0.0001], dtype=torch.float32)
z.backward(v)
print(x.grad)

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights*3).sum()

    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()

optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()