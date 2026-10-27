"""Hybrid Muon + Adam optimizer (Muon: https://kellerjordan.github.io/posts/muon/).

Muon updates hidden weight matrices with an orthogonalized momentum step; tensors that are
not good Muon targets (biases, input/output layers, thin matrices) are updated with Adam.
Conv kernels (out, in, kh, kw) and the blind-spot parameter tensors (out, in, n_taps) are
flattened to (out, in*kh*kw) / (out, in*n_taps). The blind-spot layers store only the
unmasked taps, so orthogonalized updates cannot leak into the blind spot.
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate the orthogonal factor U V^T of G with a quintic Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    # bf16 is fine for the iteration; MPS falls back to float32
    X = G.to(torch.float32 if G.device.type == 'mps' else torch.bfloat16)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def use_muon(param, min_dim=32):
    """Muon for >=2D weights whose flattened matrix has both dimensions >= min_dim.

    This sends hidden convs to Muon and keeps the input layer (fan-in of a few frames),
    the 1-channel output layer and biases on Adam, for any of the SHINE/UDVD/UNet nets.
    """
    if param.ndim < 2:
        return False
    rows = param.shape[0]
    cols = param.numel() // rows
    return min(rows, cols) >= min_dim


def split_params(named_params, min_dim=32):
    muon, adam = [], []
    for _, p in named_params:
        if not p.requires_grad:
            continue
        (muon if use_muon(p, min_dim) else adam).append(p)
    return muon, adam


class MuonWithAuxAdam(torch.optim.Optimizer):
    """One optimizer with two kinds of param groups, so Lightning sees a single optimizer.

    Groups with use_muon=True take lr, momentum, nesterov, ns_steps, weight_decay (decoupled).
    Other groups take lr, betas, eps, weight_decay (coupled L2, as torch.optim.Adam).
    """

    def __init__(self, param_groups):
        for group in param_groups:
            if group['use_muon']:
                group.setdefault('lr', 0.02)
                group.setdefault('momentum', 0.95)
                group.setdefault('nesterov', True)
                group.setdefault('ns_steps', 5)
                group.setdefault('weight_decay', 0.0)
            else:
                group.setdefault('lr', 1e-3)
                group.setdefault('betas', (0.9, 0.999))
                group.setdefault('eps', 1e-8)
                group.setdefault('weight_decay', 0.0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group['use_muon']:
                self._muon_step(group)
            else:
                self._adam_step(group)
        return loss

    def _muon_step(self, group):
        lr, momentum, wd = group['lr'], group['momentum'], group['weight_decay']
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(p)
            buf = state['momentum_buffer']
            buf.lerp_(p.grad, 1 - momentum)
            g = p.grad.lerp(buf, momentum) if group['nesterov'] else buf
            g2d = g.reshape(g.size(0), -1)
            update = zeropower_via_newtonschulz5(g2d, steps=group['ns_steps'])
            # keep the per-element update size independent of the matrix aspect ratio
            update = update * max(1.0, g2d.size(0) / g2d.size(1)) ** 0.5
            if wd != 0:
                p.mul_(1 - lr * wd)
            p.add_(update.view_as(p), alpha=-lr)

    def _adam_step(self, group):
        lr, (beta1, beta2), eps, wd = group['lr'], group['betas'], group['eps'], group['weight_decay']
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad if wd == 0 else p.grad.add(p, alpha=wd)
            state = self.state[p]
            if 'step' not in state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            bias1 = 1 - beta1 ** state['step']
            bias2 = 1 - beta2 ** state['step']
            denom = (exp_avg_sq / bias2).sqrt_().add_(eps)
            p.addcdiv_(exp_avg, denom, value=-lr / bias1)
