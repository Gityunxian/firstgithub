import torch
from torch.autograd import Variable
def hook_our(grad):
    print grad


x = Variable(torch.ones(2, 2)*1, requires_grad=True)
y = x + 2
z = y * y * 3
h = y.register_hook(hook_our)
out = z.mean()
out.backward(torch.FloatTensor([[1, 2], [3, 4]]))#这里是默认情况，相当于out.backward(torch.Tensor([1.0]))
print(x.grad)
h.remove()


from torch.autograd import Variable
import torch
v = Variable(torch.Tensor([0, 0, 0]), requires_grad=True)
h = v.register_hook(lambda grad: grad * 2)  # double the gradient
v.backward(torch.Tensor([1, 2, 3]))
#先计算原始梯度，再进hook，获得一个新梯度。
print(v.grad.data)
h.remove()  # removes the hook

