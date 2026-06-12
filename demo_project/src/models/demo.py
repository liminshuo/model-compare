from src.ops.custom_add import custom_add, CustomAddModule


def forward_legacy(x, y):
    x = x.cuda()  # 迁移遗留示例
    return custom_add(x, y)


model = CustomAddModule()
