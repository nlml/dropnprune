import time
import torch
from torch import nn
from typing import Optional

from torch.autograd.grad_mode import F
from torch import Tensor


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
    def __init__(self, num_features_out, sel_dim=None):
        super().__init__()
        self.num_features_out = num_features_out
        if sel_dim is None:
            sel_dim = self.num_features_out
        self.sel_dim = sel_dim
        self.register_buffer("sel", torch.arange(self.sel_dim))

    def forward(self, x) -> Tensor:
        if self.sel is None:
            return x
        with torch.no_grad():
            out = torch.zeros(
                [x.shape[0], self.num_features_out] + [int(i) for i in x.shape[-2:]],
                device=x.device,
            )
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
            self.reupscale_layer = ReUpScaleLayer(
                self.required_output_channels, n_channels
            )

        self.lamb = 64 / n_channels

        self.bypass = False
        self.recording = True
        self.enabled = True
        self.register_buffer("masks_history", torch.zeros([0, self.n_channels]))
        self.register_buffer("enabled_params", torch.ones([1, self.n_channels, 1, 1]))

    def _forward(self, x) -> Tensor:
        if self.bypass:
            return x
        if self.training and self.enabled:
            with torch.no_grad():
                shape = [x.shape[0], len(self.remaining_channels)]  # (B, C)
                mask_small = get_dropout_mask(shape, self.p, x.device)  # (B, C)
                mask = torch.ones_like(x)
                rem = torch.LongTensor(self.remaining_channels)
                mask[:, rem] *= mask_small.unsqueeze(-1).unsqueeze(-1)
                if self.recording:
                    self.masks_history = torch.cat([self.masks_history, mask_small], 0)
            x = x * mask * (1.0 / (1 - self.p))  # (B, C, 1, 1)
        return x * self.enabled_params

    def forward(self, x) -> Tensor:
        x = self._forward(x)
        if hasattr(self, "reupscale_layer"):
            if x.shape[1] != self.required_output_channels:
                return self.reupscale_layer(x)
        return x

    def prune_some_channels(self, channels):
        channels_in_orig = torch.tensor(self.remaining_channels, device=channels.device)
        channels_in_orig = channels_in_orig[channels]
        with torch.no_grad():
            self.enabled_params[0, channels_in_orig] = 0
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
        pct_to_prune: float = 0.4,
        sched_cfg: dict = {"type": "cosine", "warmup": 10, "invert": False},
        detrending_on: bool = False,
        dropout_ratio_mode: bool = True,
        lambda_multiplier: float = 1,
        lambda_pow: float = 1,
    ):
        self.pruning_freq = pruning_freq
        self.prune_on_batch_idx = prune_on_batch_idx
        self.pct_to_prune = pct_to_prune
        self.sched_cfg = sched_cfg
        self.detrending_on = detrending_on
        self.dropout_ratio_mode = dropout_ratio_mode
        self.lambda_multiplier = lambda_multiplier
        self.lambda_pow = lambda_pow

        self._loss_history = []
        self.global_step = 0
        self._num_pruned_so_far = 0

        self.dropnprune_layers = get_modules_start_with_name(model, "DropNPrune")
        self.total_num_params = sum(
            [l.enabled_params.sum().item() for l in self.dropnprune_layers]
        )
        self._num_remaining_params = self.total_num_params
        self.total_num_to_prune = int(self.pct_to_prune * self.total_num_params)

        with torch.no_grad():
            if self.sched_cfg["type"] == "cosine":
                warmup = self.sched_cfg.get("warmup", 5)
                if sched_cfg.get("invert", False):
                    sched_cfg["warmup"] = 0
                    x = torch.cos(torch.linspace(torch.pi, 0, 200 - warmup)) / 2 + 0.5
                else:
                    x = torch.cos(torch.linspace(0, torch.pi, 200 - warmup)) / 2 + 0.5
                if sched_cfg.get("hard_warmup", False):
                    x = torch.cat([torch.zeros([warmup]), x])
                else:
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
        self._num_remaining_params = sum(
            [l.enabled_params.sum() for l in self.dropnprune_layers]
        ).item()
        self._num_pruned_so_far = self.total_num_params - self._num_remaining_params
        return self._num_pruned_so_far

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

    def set_dropout_mode(self, mode):
        assert mode in [True, False]
        for l in self.dropnprune_layers:
            l.enabled = mode

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
        ran_pruning = False
        num_to_prune = self.calc_num_to_prune(batch_idx, epoch)
        if num_to_prune is not None:
            if num_to_prune > 0:
                if len([i.masks_history for i in self.dropnprune_layers][0]):
                    if len(self._loss_history):
                        print("num_to_prune", num_to_prune)
                        self.run_pruning(self._loss_history, num_to_prune)
                        ran_pruning = True
            self.clean_up()
        return ran_pruning

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
        if self.detrending_on:
            trend = estimate_trend(all_losses, loss_timestamps)
            # detrended_loss = all_losses - trend
            # hi score means dropping that param out tended to cause large increases in loss
            # lo score means dropping that param tended to have minimal impact or even help
            pct_diff_loss_to_trend = all_losses / trend - 1
        else:
            pct_diff_loss_to_trend = all_losses - all_losses.mean(0, keepdim=True)

        lambdas = torch.cat(
            [
                torch.full(
                    [len(l.remaining_channels)], l.lamb, device=loss_history[0].device
                )
                for l in self.dropnprune_layers
            ]
        )
        lambdas = (lambdas ** self.lambda_pow) * self.lambda_multiplier
        scores = linreg_torch(
            all_mask_histories,
            pct_diff_loss_to_trend,
            1 - self.dropnprune_layers[0].p,
            lamb=lambdas,
        )

        # TODO: DELETE THIS
        # scores = torch.randn([len(scores)])
        self._last_scores = scores.detach().cpu()
        highest_score_idxs = torch.argsort(-scores)
        highest_score_idxs = highest_score_idxs[:n_channels_to_prune]
        cum_layer_sizes = torch.cumsum(
            torch.LongTensor(
                [0] + [int(i.masks_history.shape[1]) for i in self.dropnprune_layers]
            ),
            0,
        )
        to_prune = {}
        for idx in highest_score_idxs:
            layer_idx = layer_idxs[idx].item()
            idx_in_layer = idx - cum_layer_sizes[layer_idx]
            if layer_idx not in to_prune:
                to_prune[layer_idx] = [idx_in_layer]
            else:
                to_prune[layer_idx].append(idx_in_layer)
        for layer_idx in sorted(to_prune.keys()):
            channels_to_prune = to_prune[layer_idx]
            self.dropnprune_layers[layer_idx].prune_some_channels(
                torch.stack(channels_to_prune)
            )
        print(f"Done pruning! in {time.time() - now:.2f} secs")
        print(
            [
                (i, l.enabled_params.sum().item())
                for i, l in enumerate(self.dropnprune_layers)
            ]
        )
