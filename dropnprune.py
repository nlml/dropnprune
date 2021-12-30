import time
import torch
from torch import nn
from typing import Optional

from torch.autograd.grad_mode import F


def get_modules_start_with_name(model, name):
    l = []

    def recurse(model, l):
        for module in model.children():
            if str(module).startswith(name):
                l.append(module)
            recurse(module, l)

    recurse(model, l)
    return l


def get_dropout_mask(shape, prob, device="cpu"):
    return torch.bernoulli(torch.full(shape, 1 - prob, device=device))


class ReUpScaleLayer(nn.Module):
    def __init__(self, num_features_out):
        super().__init__()
        self.num_features_out = num_features_out
        self.sel = None

    def forward(self, x):
        if self.sel is None:
            return x
        with torch.no_grad():
            out = torch.zeros_like(x[:, :1])
            out = torch.tile(out, [1, self.num_features_out] + [1] * (len(x.shape) - 2))
        out[:, self.sel] += x
        return out


class DropNPrune(nn.Module):
    def __init__(self, p, n_channels, required_output_channels=None):
        super().__init__()
        self.p = p
        self.n_channels = n_channels
        self.remaining_channels = list(range(self.n_channels))
        self.required_output_channels = required_output_channels
        if self.required_output_channels is not None:
            self.reupscale_layer = ReUpScaleLayer(self.required_output_channels)

        self.recording = True
        self.enabled = True
        self.register_buffer("masks_history", torch.zeros([0, self.n_channels]))
        self.register_buffer("enabled_params", torch.ones([1, self.n_channels, 1, 1]))

    def forward(self, x):
        if self.training and self.enabled:
            with torch.no_grad():
                shape = [x.shape[0], len(self.remaining_channels)]  # (B, C)
                mask_small = get_dropout_mask(shape, self.p, x.device)  # (B, C)
                mask = torch.ones_like(x)
                rem = torch.LongTensor(self.remaining_channels)
                mask[:, rem] *= mask_small.unsqueeze(-1).unsqueeze(-1)
            out = x * mask * self.enabled_params * (1.0 / (1 - self.p))  # (B, C, 1, 1)
            if self.training and self.recording:
                with torch.no_grad():
                    self.masks_history = torch.cat([self.masks_history, mask_small], 0)
        else:
            out = x * self.enabled_params
        if self.required_output_channels is not None:
            if out.shape[1] != self.required_output_channels:
                if self.required_output_channels.sel is None:
                    sel = torch.where(self.enabled_params[0, :, 0, 0])[0].cpu()
                    self.reupscale_layer.sel = sel
                out = self.reupscale_layer(out)
        return out

    def prune_some_channels(self, channels):
        channels_in_orig = [self.remaining_channels[i] for i in channels]
        with torch.no_grad():
            self.enabled_params[0, torch.LongTensor(channels_in_orig)] = 0
        for c in channels_in_orig:
            self.remaining_channels.remove(c)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Overrides state_dict() to not save any masks_history"""
        d = super().state_dict(destination, prefix, keep_vars)
        d[prefix + "masks_history"] = d[prefix + "masks_history"][:0]
        return d


def linreg_torch(x, y, p=None, lamb=None):
    # betas = (X'X)^1 X'y
    N, D = x.shape[0], x.shape[1]
    if p is not None:
        # p should be probability entry == 1 (ie prob of not dropping out)
        print(f"Linear regression with matrix of shape: {list(x.shape)}")
        c = p ** 2
        xtransx = torch.eye(D, device=x.device) * (p - c)  # diagonal = p
        xtransx += c  # off-diagonals = p ** 2
        xtransx *= N
        if lamb is not None:
            xtransx += lamb * torch.eye(D, device=x.device)
        betas = torch.linalg.inv(xtransx).matmul(x.T)
    else:
        if lamb is None:
            betas = torch.linalg.pinv(x)
        else:
            xtransx = x.T.matmul(x) + lamb * torch.eye(D, device=x.device)
            betas = torch.linalg.inv(x).matmul(x.T)
    betas = betas.matmul(y)
    return betas


def estimate_trend(y, x):
    with torch.no_grad():
        x = x.float()
        x = torch.stack(
            [
                torch.ones([x.shape[0]], device=x.device),
                x,
                torch.log(x),
                torch.sqrt(x),
            ],
            1,
        )
        betas = linreg_torch(x, y)
    return x.matmul(betas)


class Pruner:
    def __init__(
        self,
        model,
        pruning_freq: Optional[int] = None,
        prune_on_batch_idx: Optional[int] = 0,
        pruning_warmup: int = 0,
        pct_to_prune: float = 0.4,
        sched_cfg: dict = {"type": "cosine"},
    ):

        self.pruning_freq = pruning_freq
        self.prune_on_batch_idx = prune_on_batch_idx
        self.pruning_warmup = pruning_warmup
        self.pct_to_prune = pct_to_prune
        self.sched_cfg = sched_cfg
        self._loss_history = []
        self.global_step = 0

        self.dropnprune_layers = get_modules_start_with_name(model, "DropNPrune")
        self.total_num_params = sum(
            [l.enabled_params.sum().item() for l in self.dropnprune_layers]
        )
        self.total_num_to_prune = int(self.pct_to_prune * self.total_num_params)

        with torch.no_grad():
            if self.sched_cfg["type"] == "cosine":
                warmup = self.sched_cfg.get("warmup", 5)
                x = torch.cos(torch.linspace(0, torch.pi, 200 - warmup)) / 2 + 0.5
                x = torch.cat([torch.linspace(0, 1, warmup), x])
            else:  # default pruning schedule
                x = torch.cat(
                    [
                        torch.tile(torch.LongTensor([1]), [100]),
                        torch.tile(torch.LongTensor([0]), [100]),
                    ],
                    0,
                )
            assert x.shape[0] == 200
            self.f = torch.cumsum(x / x.sum(), 0)

    @property
    def num_pruned_so_far(self):
        return self.total_num_params - sum(
            [l.enabled_params.sum().item() for l in self.dropnprune_layers]
        )

    def step(self, loss):
        self.global_step += 1
        self._loss_history.append(loss.detach())

    def clean_up(self):
        with torch.no_grad():
            for l in self.dropnprune_layers:
                l.masks_history = torch.zeros([0, len(l.remaining_channels)]).to(
                    l.masks_history.device
                )
            self._loss_history = []

    def pruning_scheduler(self, epoch):
        return self.f[epoch]

    def calc_num_to_prune(self, batch_idx, epoch):
        if self.prune_on_batch_idx is not None:
            if self.prune_on_batch_idx != batch_idx:
                return None
        elif self.pruning_freq is not None:
            if self.global_step % self.pruning_freq != 0:
                return None
        else:
            raise Exception(
                "Either pruning_freq or prune_on_batch_idx must be defined!"
            )
        return int(
            round(
                (
                    self.pruning_scheduler(epoch) * self.total_num_to_prune
                    - self.num_pruned_so_far
                ).item()
            )
        )

    def maybe_run_pruning(self, batch_idx, epoch):
        num_to_prune = self.calc_num_to_prune(batch_idx, epoch)
        if num_to_prune is not None:
            if num_to_prune > 0:
                if len([i.masks_history for i in self.dropnprune_layers][0]):
                    if len(self._loss_history):
                        print("num_to_prune", num_to_prune)
                        self.run_pruning(self._loss_history, num_to_prune)
            self.clean_up()

    def run_pruning(self, loss_history, n_channels_to_prune):
        if n_channels_to_prune <= 0:
            return None
        print("Running pruning...")
        now = time.time()
        all_mask_histories = [i.masks_history for i in self.dropnprune_layers]
        layer_idxs = torch.cat(
            [
                torch.full([i.shape[1]], idx, device=loss_history[0].device)
                for idx, i in enumerate(all_mask_histories)
            ]
        )
        # all_mask_histories: (N, featuresLayeri * L layers)
        all_mask_histories = torch.cat(all_mask_histories, 1)
        # loss_timestamps = [1, 1, 1, ..., 2, 2, 2, ..., ..., pruning_freq, pruning_freq, ...]
        loss_timestamps = torch.cat(
            [
                torch.full([i.shape[0]], idx + 1, device=loss_history[0].device)
                for idx, i in enumerate(loss_history)
            ]
        )
        all_losses = torch.cat(loss_history, 0)  # (N,)
        trend = estimate_trend(all_losses, loss_timestamps)
        # detrended_loss = all_losses - trend
        # hi score means dropping that param out tended to cause large increases in loss
        # lo score means dropping that param tended to have minimal impact or even help
        pct_diff_loss_to_trend = all_losses / trend - 1
        # import pdb; pdb.set_trace()

        scores = linreg_torch(
            all_mask_histories, pct_diff_loss_to_trend, 1 - self.dropnprune_layers[0].p
        )
        # torch.save((all_mask_histories.cpu().numpy(), all_losses, loss_timestamps), '/tmp/tt')

        # TODO: DELETE THIS
        # scores = torch.randn([len(scores)])
        lowest_score_idxs = torch.argsort(-scores)[:n_channels_to_prune]
        cum_layer_sizes = torch.cumsum(
            torch.LongTensor(
                [0] + [i.masks_history.shape[1] for i in self.dropnprune_layers]
            ),
            0,
        )
        to_prune = {}
        for idx in lowest_score_idxs:
            layer_idx = layer_idxs[idx]
            idx_in_layer = idx - cum_layer_sizes[layer_idx]
            if layer_idx not in to_prune:
                to_prune[layer_idx] = [idx_in_layer]
            else:
                to_prune[layer_idx].append(idx_in_layer)
        for layer_idx, channels_to_prune in to_prune.items():
            self.dropnprune_layers[layer_idx].prune_some_channels(channels_to_prune)
        print(f"Done pruning! in {time.time() - now:.2f} secs")

        print(
            [
                (i, l.enabled_params.sum().item())
                for i, l in enumerate(self.dropnprune_layers)
            ]
        )
