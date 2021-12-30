import torch
from train import *
from ptflops import get_model_complexity_info
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn
from dropnprune import get_modules_start_with_name


def complexity(net):
    macs, params = get_model_complexity_info(
        net, (3, 224, 224), as_strings=True, print_per_layer_stat=False
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))


ckpt_path = "lightning_logs/prune0.2-cosineWarm5-lambda100pow2/version_0/checkpoints/epoch=193-step=75659.ckpt"
# ckpt_path = "lightning_logs/resnet/version_13/checkpoints/epoch=199-step=77999.ckpt"
# ckpt_path = "lightning_logs/prune0.4-cosineWarm5-moreFixes2/version_1/checkpoints/epoch=196-step=76829.ckpt"
ckpt = torch.load(ckpt_path)
state_dict = ckpt["state_dict"]

for k in list(state_dict.keys()):
    print(k)
    if "masks_history" in k:
        del state_dict[k]
        continue
    state_dict[k[len("model.") :]] = state_dict[k]
    del state_dict[k]

model = resnet32()
model.load_state_dict(state_dict, strict=False)
model = model.eval().cpu()
complexity(model)
torch.save(model.state_dict(), "pre.pth")
dummy_input = torch.zeros([1, 3, 32, 32])
torch.jit.save(torch.jit.trace(model, dummy_input), "pre.trace")
print(model(dummy_input))

int_planes_all = [
    int(state_dict[k].sum()) for k in list(state_dict.keys()) if "enabled" in k
]
sels = [
    torch.where(state_dict[k][0, :, 0, 0])[0].cpu()
    for k in list(state_dict.keys())
    if "enabled" in k
]

f = lambda p: [[p[i * 2], p[i * 2 + 1]] for i in range(5)]
planes_l1 = f(int_planes_all[:10])
planes_l2 = f(int_planes_all[10:20])
planes_l3 = f(int_planes_all[20:30])

model = resnet32(planes_per_layer=[planes_l1, planes_l2, planes_l3])

sel2s = []

keys = list(state_dict.keys())
for layer in ["layer1.", "layer2.", "layer3."]:
    this_keys = [i for i in keys if i.startswith(layer)]
    for i in range(5):
        sel = sels.pop(0)
        for s in [
            "conv1.weight",
            "bn1.weight",
            "bn1.bias",
            "bn1.running_mean",
            "bn1.running_var",
        ]:
            k = layer + f"{i}.{s}"
            state_dict[k] = state_dict[k][sel]
        for s in [
            # "dropnprune1.enabled_params",
            "conv2.weight",
        ]:
            k = layer + f"{i}.{s}"
            state_dict[k] = state_dict[k][:, sel]
        sel = sels.pop(0)
        sel2s.append(sel)
        for s in [
            "conv2.weight",
            "bn2.weight",
            "bn2.bias",
            "bn2.running_mean",
            "bn2.running_var",
        ]:
            k = layer + f"{i}.{s}"
            state_dict[k] = state_dict[k][sel]
        # k = layer + f"{i}.dropnprune2.enabled_params"
        # where = torch.where(state_dict[k][0, :, 0, 0])[0]
        # state_dict[k] = state_dict[k][:, sel]
        # k = layer + f"{i}.dropnprune2.reupscale_layer.sel"
        # state_dict[k] = where
model = model.eval().cpu()
model.load_state_dict(state_dict, strict=False)

reups = get_modules_start_with_name(model, "ReUp")
dropnprunes = get_modules_start_with_name(model, "DropNPrune")
for l in dropnprunes:
    l.bypass = True
for r, s in zip(reups, sel2s):
    r.sel = s

complexity(model)
dummy_input = torch.zeros([1, 3, 32, 32])
print(model(dummy_input))

torch.save(model.state_dict(), "post.pth")
torch.jit.save(torch.jit.trace(model, dummy_input), "post.trace")


val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root=PATH_DATASETS, train=False, transform=test_transforms),
    batch_size=128,
    num_workers=8,
    pin_memory=True,
)

criterion = nn.CrossEntropyLoss().cuda()
model = model.cuda()
# evaluate on validation set
m = LitResnet(create_model_fn=lambda: model)
trainer = Trainer(gpus=1)
trainer.validate(m, dataloaders=[val_loader])
