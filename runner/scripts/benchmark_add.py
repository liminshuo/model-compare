import sys, time
sys.path.insert(0, 'demo_project')
import torch
from src.ops.custom_add import custom_add
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
a = torch.randn(1024, 1024, device=dev)
b = torch.randn(1024, 1024, device=dev)
o1 = torch.add(a, b)
o2 = custom_add(a, b)
print('device', dev)
print('max_diff', (o1 - o2).abs().max().item())
for name, fn in [('torch.add', lambda: torch.add(a,b)), ('custom_add', lambda: custom_add(a,b))]:
    if dev == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200): fn()
    if dev == 'cuda': torch.cuda.synchronize()
    print(name, 'ms/iter', (time.perf_counter()-t0)/200*1000)
