import numpy as np
import time
import torch
from torch import nn
from typing import Optional

from torch.autograd.grad_mode import F
from torch import Tensor


def moving_average(a, n=3):
    if n == 1:
        return a
    ret = torch.cumsum(a.float(), dim=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


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
    def __init__(
        self, p, n_channels, required_output_channels=None, rescale_post_dropout=True
    ):
        super().__init__()
        self.p = p
        self.n_channels = n_channels
        self.remaining_channels = list(range(self.n_channels))
        self.required_output_channels = required_output_channels
        self.rescale_post_dropout = rescale_post_dropout
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
            x = x * mask
            if self.rescale_post_dropout:
                x = x * (1.0 / (1 - self.p))  # (B, C, 1, 1)
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


def linreg_torch(x, y, p=None, lamb=None, return_preds=False, return_xtxinv=False):
    # betas = (X'X)^1 X'y
    N, D = x.shape[0], x.shape[1]
    if p is not None:
        # p should be probability entry == 1 (ie prob of not dropping out)
        c = p ** 2
        xtx = torch.eye(D, device=x.device) * (p - c)  # diagonal = p
        xtx += c  # off-diagonals = p ** 2
        xtx *= N
        if lamb is not None:
            xtx += lamb * torch.eye(D, device=x.device)
        xtxinv = torch.linalg.inv(xtx)
        betas = xtxinv.matmul(x.T)
    else:
        if lamb is None:
            if return_xtxinv:
                xtx = x.T.matmul(x)
                xtxinv = torch.linalg.inv(xtx)
                betas = xtxinv.matmul(x.T)
            else:
                betas = torch.linalg.pinv(x)
        else:
            xtx = x.T.matmul(x) + lamb * torch.eye(D, device=x.device)
            xtxinv = torch.linalg.inv(xtx)
            betas = xtxinv.matmul(x.T)
    betas = betas.matmul(y)
    if return_preds:
        preds = x.matmul(betas).squeeze()
        if return_xtxinv:
            return betas, preds, xtxinv
        return betas, preds
    if return_xtxinv:
        return betas, xtxinv
    return betas


def estimate_trend(y, x, just_linear=False):
    with torch.no_grad():
        x = x.float()
        if just_linear:
            x = torch.stack(
                [
                    torch.ones([x.shape[0]], device=x.device),
                    x,
                ],
                1,
            )
        else:
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


def calc_scores(bsize, all_mask_histories, all_losses, p, ma):
    device = all_losses.device
    # all_losses = torch.cat(loss_history, 0)  # (N,)
    all_losses = -torch.exp(-all_losses)

    # Calc a simple linear trend over the training steps (batches) then subtract it
    # from the per-sample loss
    al = all_losses.view(-1, bsize).mean(1)
    trend_al = estimate_trend(
        al, torch.arange(1, len(al) + 1, device=device), just_linear=True
    )
    trend_all_losses = trend_al.unsqueeze(1).repeat([1, bsize]).view(-1)
    all_losses = all_losses - trend_all_losses

    if ma:
        # Calc a moving average over the training steps then subtract from per-sample loss
        al = all_losses.view(-1, bsize).mean(1)
        # mirror padding for centered moving average:
        alx = torch.cat(
            [
                al[: ma // 2][torch.arange(ma // 2 - 1, -1, -1)],
                al,
                al[-ma // 2 - 1 :][torch.arange(ma // 2 - 2, -1, -1)],
            ]
        )
        trend_al = moving_average(alx, ma)
        trend_all_losses = trend_al.unsqueeze(1).repeat([1, bsize]).view(-1)
        all_losses = all_losses - trend_all_losses
    all_losses -= all_losses.mean()

    betas = linreg_torch(all_mask_histories, all_losses, p)
    return betas


class Pruner:
    def __init__(
        self,
        model,
        pruning_freq: Optional[int] = None,
        prune_on_batch_idx: Optional[int] = 0,
        pct_to_prune: float = 0.4,
        sched_cfg: dict = {"type": "cosine", "warmup": 0, "finish": 0},
        detrending_on: bool = True,
        dropout_ratio_mode: bool = False,
        lambda_multiplier: float = 0,
        lambda_pow: float = 1,
        prune_every_epoch: Optional[int] = 5,
        ma: Optional[int] = 50,
        score_threshold: float = 0.005,
    ):
        self.pruning_freq = pruning_freq
        self.prune_on_batch_idx = prune_on_batch_idx
        self.pct_to_prune = pct_to_prune
        self.sched_cfg = sched_cfg
        self.detrending_on = detrending_on
        self.dropout_ratio_mode = dropout_ratio_mode
        self.lambda_multiplier = lambda_multiplier
        self.lambda_pow = lambda_pow
        self.prune_every_epoch = prune_every_epoch
        self.ma = ma
        self.score_threshold = score_threshold

        self._loss_history = []
        self.global_step = 0
        self._num_pruned_so_far = 0
        self._num_pruned_this_round = 0

        self.dropnprune_layers = get_modules_start_with_name(model, "DropNPrune")
        self.total_num_params = sum(
            [l.enabled_params.sum().item() for l in self.dropnprune_layers]
        )
        self._num_remaining_params = self.total_num_params
        self.total_num_to_prune = int(self.pct_to_prune * self.total_num_params)

        with torch.no_grad():
            if self.sched_cfg["type"] == "cosine":
                warmup = self.sched_cfg.get("warmup", 5)
                finish = self.sched_cfg.get("finish", 0)
                if sched_cfg.get("invert", False):
                    sched_cfg["warmup"] = 0
                    x = torch.cos(torch.linspace(torch.pi, 0, 200 - warmup - finish))
                else:
                    x = torch.cos(torch.linspace(0, torch.pi, 200 - warmup - finish))
                x = x / 2 + 0.5
                if sched_cfg.get("hard_warmup", False):
                    x = torch.cat([torch.zeros([warmup]), x])
                else:
                    x = torch.cat([torch.linspace(0, 1, warmup), x])
                x = torch.cat([x, torch.zeros([finish])])
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
        if self.prune_every_epoch is not None:
            if (epoch + 1) % self.prune_every_epoch != 0:
                return None
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

    def maybe_run_pruning(self, batch_idx, epoch, save_path=None):
        with torch.no_grad():
            ran_pruning = False
            num_to_prune = self.calc_num_to_prune(batch_idx, epoch)
            if num_to_prune is not None:
                if num_to_prune > 0 or self.score_threshold is not None:
                    if len([i.masks_history for i in self.dropnprune_layers][0]):
                        if len(self._loss_history):
                            print("num_to_prune", num_to_prune)
                            self.run_pruning(
                                self._loss_history, num_to_prune, save_path
                            )
                            ran_pruning = True
                self.clean_up()
        return ran_pruning

    def run_pruning(self, loss_history, n_channels_to_prune, save_path=None):
        if self.score_threshold is None and n_channels_to_prune <= 0:
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
        bsize = loss_history[0].shape[0]
        all_losses = torch.cat(loss_history, 0)  # (N,)
        scores = calc_scores(
            bsize,
            all_mask_histories,
            all_losses,
            1 - self.dropnprune_layers[0].p,
            self.ma,
        )

        # TODO: DELETE THIS
        # scores = -scores
        # scores = torch.randn([len(scores)])
        # scores = -torch.abs(scores)
        if 0:  # save_path is not None:
            torch.save(
                (
                    scores,
                    loss_history,
                    # all_mask_histories,
                    # pct_diff_loss_to_trend,
                    # lambdas,
                    self.lambda_pow,
                    self.lambda_multiplier,
                ),
                save_path,
            )
            print(f"Saved {save_path}")

        self._last_scores = scores.detach().cpu().numpy()
        highest_score_idxs = torch.argsort(-scores)
        cum_layer_sizes = torch.cumsum(
            torch.LongTensor(
                [0] + [int(i.masks_history.shape[1]) for i in self.dropnprune_layers]
            ),
            0,
        )
        to_prune = {}
        tmp_remaining_channels = {}
        num_pruned = 0
        i = -1
        while self.score_threshold is not None or num_pruned < n_channels_to_prune:
            i += 1
            idx = highest_score_idxs[i]
            if self.score_threshold is not None and scores[idx] < self.score_threshold:
                print(
                    f"Score {scores[idx]} below thresh {self.score_threshold}"
                    f" - exiting after pruning {num_pruned} "
                    f"(n_channels_to_prune was {n_channels_to_prune})"
                )
                break
            layer_idx = layer_idxs[idx].item()
            # Make sure all layers keep at least 1 parameter
            if layer_idx not in tmp_remaining_channels:
                tmp_remaining_channels[layer_idx] = len(
                    self.dropnprune_layers[layer_idx].remaining_channels
                )
            if tmp_remaining_channels[layer_idx] > 1:
                idx_in_layer = idx - cum_layer_sizes[layer_idx]
                if layer_idx not in to_prune:
                    to_prune[layer_idx] = [idx_in_layer]
                else:
                    to_prune[layer_idx].append(idx_in_layer)
                num_pruned += 1
                tmp_remaining_channels[layer_idx] -= 1
        self._num_pruned_this_round = num_pruned
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
