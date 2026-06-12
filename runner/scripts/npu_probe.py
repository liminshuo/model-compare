import torch
tests = {}
def add_test():
    a = torch.randn(4)
    b = torch.randn(4)
    try:
        import torch_npu
        if hasattr(torch, 'npu') and torch.npu.is_available():
            d = 'npu:0'
            torch.add(a.to(d), b.to(d))
            return 'NPU OK'
    except Exception as e:
        torch.add(a, b)
        return f'CPU fallback: {e}'
    torch.add(a, b)
    return 'CPU OK'
print('torch.add =>', add_test())
