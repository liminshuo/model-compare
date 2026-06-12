import torch


def custom_add(a, b):
    """Hello World 自定义 wrapper — 语义等价于 torch.add"""
    return a + b


class CustomAddModule(torch.nn.Module):
    def forward(self, x, y):
        return custom_add(x, y)
