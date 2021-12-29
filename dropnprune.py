import time
import torch
from torch import nn


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
    return torch.bernoulli(torch.tensor([1 - prob]).repeat(shape)).to(device)


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
        self.required_output_channels = required_output_channels
        if self.required_output_channels is not None:
            self.reupscale_layer = ReUpScaleLayer(self.required_output_channels)

        self.recording = False
        self.enabled = True
        self.register_buffer("masks_history", torch.zeros([0, self.n_channels]))
        self.register_buffer("enabled_params", torch.ones([1, self.n_channels, 1, 1]))

    def reset_masks_history(self):
        self.masks_history = torch.zeros([0, self.n_channels]).to(
            self.masks_history.device
        )

    def forward(self, x):
        if self.training and self.enabled:
            shape = [x.shape[0], x.shape[1]]  # (B, C)
            mask = get_dropout_mask(shape, self.p, x.device)  # (B, C)
            out = (
                x
                * mask.unsqueeze(-1).unsqueeze(-1)
                * self.enabled_params
                * (1.0 / (1 - self.p))
            )  # (B, C, 1, 1)
            if self.training and self.recording:
                with torch.no_grad():
                    self.masks_history = torch.cat([self.masks_history, mask], 0)
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
        with torch.no_grad():
            self.enabled_params[0, torch.LongTensor(channels)] = 0

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Overrides state_dict() to not save any masks_history"""
        d = super().state_dict(destination, prefix, keep_vars)
        d[prefix + "masks_history"] = d[prefix + "masks_history"][:0]
        return d


def linreg_torch(x, y, p=None):
    # betas = (X'X)^1 X'y
    if p is not None:
        # p should be probability entry == 1 (ie prob of not dropping out)
        N = x.shape[0]
        c = p ** 2
        xtransx = torch.eye(int(x.shape[1])).to(x.device) * (p - c)  # diagonal = p
        xtransx += c  # off-diagonals = p ** 2
        xtransx *= N
        betas = torch.linalg.inv(xtransx).matmul(x.T)
    else:

        betas = torch.linalg.pinv(x)
    betas = betas.matmul(y)
    return betas


def estimate_trend(y, x):
    with torch.no_grad():
        x = x.float()
        x = torch.stack(
            [
                torch.ones([x.shape[0]], device=x.device),
                torch.log(x),
                # torch.sqrt(x),
            ],
            1,
        )
        betas = linreg_torch(x, y)
    return x.matmul(betas)


def run_pruning(loss_history, dropprunes, n_channels_to_prune):
    rrr = n_channels_to_prune - int(n_channels_to_prune)
    n_channels_to_prune = int(n_channels_to_prune) + int(torch.rand([1]).item() < rrr)
    if n_channels_to_prune <= 0:
        return None
    print("Running pruning...")
    now = time.time()
    dropprune_layers = [i for i in dropprunes]
    print(f"len(dropprune_layers): {len(dropprune_layers)}")
    all_mask_histories = [i.masks_history for i in dropprune_layers]
    layer_idxs = torch.cat(
        [
            torch.full([i.shape[1]], idx, device=loss_history[0].device)
            for idx, i in enumerate(all_mask_histories)
        ]
    )
    all_mask_histories = torch.cat(
        all_mask_histories, 1
    )  # (N, featuresLayeri * L layers)
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
        all_mask_histories, pct_diff_loss_to_trend, 1 - dropprunes[0].p
    )
    # torch.save((all_mask_histories.cpu().numpy(), all_losses, loss_timestamps), '/tmp/tt')

    # TODO: DELETE THIS
    # scores = torch.randn([len(scores)])
    lowest_score_idxs = torch.argsort(-scores)[:n_channels_to_prune]
    cum_layer_sizes = torch.cumsum(
        torch.LongTensor([0] + [i.masks_history.shape[1] for i in dropprune_layers]),
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
        dropprune_layers[layer_idx].prune_some_channels(channels_to_prune)
    print(f"Done pruning! in {time.time() - now:.2f} secs")

    print([(i, l.enabled_params.sum().item()) for i, l in enumerate(dropprune_layers)])
    return None
