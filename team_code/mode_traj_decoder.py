import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
  def __init__(self, dim, eps=1e-6):
    super().__init__()
    self.eps = eps
    self.scale = dim**-0.5
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x):
    norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
    return x / norm.clamp(min=self.eps) * self.weight


class SwishGLU(nn.Module):
  def __init__(self, dim, hidden_dim):
    super().__init__()
    self.project = nn.Linear(dim, hidden_dim * 2)

  def forward(self, x):
    value, gate = self.project(x).chunk(2, dim=-1)
    return value * F.silu(gate)


class ExpertMLP(nn.Module):
  def __init__(self, dim, dropout):
    super().__init__()
    self.net = nn.Sequential(
        SwishGLU(dim, dim * 4),
        nn.Dropout(dropout),
        nn.Linear(dim * 4, dim, bias=False),
    )

  def forward(self, x):
    return self.net(x)


class NoiseConditionedRouter(nn.Module):
  def __init__(self, dim, num_experts, top_k, normalize=True):
    super().__init__()
    self.num_experts = int(num_experts)
    self.top_k = max(1, min(int(top_k), self.num_experts))
    self.normalize = normalize
    self.router = nn.Sequential(
        nn.Linear(dim * 2, dim),
        nn.GELU(),
        nn.Linear(dim, self.num_experts),
    )
    self.last_probs = None
    self.last_router_mask = None

  def forward(self, x, noise_cond):
    if noise_cond.size(1) != x.size(1):
      noise_cond = noise_cond.expand(-1, x.size(1), -1)
    logits = self.router(torch.cat([x, noise_cond], dim=-1))
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probs = torch.softmax(logits, dim=-1).clamp(1e-9, 1.0)

    flat_probs = probs.reshape(-1, self.num_experts)
    if self.training:
      top_idx = torch.multinomial(flat_probs, self.top_k, replacement=False)
    else:
      top_idx = flat_probs.topk(self.top_k, dim=-1).indices

    flat_mask = torch.zeros_like(flat_probs).scatter_(1, top_idx, 1.0)
    flat_selected_probs = torch.zeros_like(flat_probs).scatter_(1, top_idx, flat_probs.gather(1, top_idx))
    router_mask = flat_mask.view_as(probs)
    router_probs = flat_selected_probs.view_as(probs)
    if self.normalize:
      router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    self.last_probs = probs
    self.last_router_mask = router_mask
    return router_mask, router_probs

  def load_balancing_loss(self):
    if self.last_probs is None or self.last_router_mask is None:
      return None
    probs = self.last_probs.reshape(-1, self.num_experts)
    mask = self.last_router_mask.reshape(-1, self.num_experts)
    density = mask.mean(dim=0)
    density_proxy = probs.mean(dim=0)
    return self.num_experts * torch.sum(density * density_proxy)


class NoiseMoEBlock(nn.Module):
  def __init__(self, dim, heads, dropout, num_experts, top_k):
    super().__init__()
    self.norm_attn = RMSNorm(dim)
    self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
    self.norm_moe = RMSNorm(dim)
    self.router = NoiseConditionedRouter(dim, num_experts, top_k)
    self.experts = nn.ModuleList([ExpertMLP(dim, dropout) for _ in range(num_experts)])

  @staticmethod
  def _causal_mask(length, device):
    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

  def forward(self, x, noise_cond):
    attn_in = self.norm_attn(x) + noise_cond.expand(-1, x.size(1), -1)
    attn_out, _ = self.attn(
        attn_in,
        attn_in,
        attn_in,
        attn_mask=self._causal_mask(x.size(1), x.device),
        need_weights=False,
    )
    x = x + attn_out

    moe_in = self.norm_moe(x)
    router_mask, router_probs = self.router(moe_in, noise_cond)
    moe_out = torch.zeros_like(moe_in)
    for expert_idx, expert in enumerate(self.experts):
      token_mask = router_mask[..., expert_idx].bool()
      if token_mask.any():
        probs = router_probs[..., expert_idx][token_mask].unsqueeze(-1)
        moe_out[token_mask] += probs * expert(moe_in[token_mask])
    return x + moe_out

  def load_balancing_loss(self):
    return self.router.load_balancing_loss()


class MoDETrajectoryDecoder(nn.Module):
  """
  MoDE-style sparse denoising decoder for TF++ trajectory anchors.

  Inputs:
    encoder_out: (B, 10, D) TF++ trajectory tokens.

  Outputs:
    pred_trajectories: (K, B, 20), matching the old anchor decoder.
    scores: (K, B), logits for anchor confidence.
  """

  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.hidden_dim = int(getattr(cfg, 'mode_decoder_dim', getattr(cfg, 'tf_de_dim', 256)))
    self.num_layers = int(getattr(cfg, 'mode_decoder_layers', getattr(cfg, 'tf_de_layers', 4)))
    self.num_heads = int(getattr(cfg, 'mode_decoder_heads', getattr(cfg, 'tf_de_heads', 8)))
    self.dropout = float(getattr(cfg, 'mode_decoder_dropout', getattr(cfg, 'tf_de_dropout', 0.05)))
    self.num_experts = int(getattr(cfg, 'mode_decoder_num_experts', 4))
    self.top_k = int(getattr(cfg, 'mode_decoder_top_k', 2))
    self.sigma_min = float(getattr(cfg, 'mode_sigma_min', 0.02))
    self.sigma_max = float(getattr(cfg, 'mode_sigma_max', 1.0))
    self.sigma_data = float(getattr(cfg, 'mode_sigma_data', 0.5))
    self.anchor_noise_scale = float(getattr(cfg, 'mode_anchor_noise_scale', 1.0))
    self.offset_scale = float(getattr(cfg, 'mode_offset_scale', 1.0))
    self.use_noisy_anchor_prior = bool(getattr(cfg, 'mode_use_noisy_anchor_prior', True))

    anchor_mu, anchor_var = self._load_anchor_stats(getattr(cfg, 'prior_traj_path', None))
    max_queries = int(getattr(cfg, 'mode_num_anchor_queries', 0))
    if max_queries > 0:
      anchor_mu = anchor_mu[:max_queries]
      anchor_var = anchor_var[:max_queries]
    self.register_buffer('anchor_mu', anchor_mu)
    self.register_buffer('anchor_var', anchor_var.clamp_min(1e-6))

    self.traj_token_proj = nn.Sequential(
        nn.Linear(int(getattr(cfg, 'gru_input_size', self.hidden_dim)), self.hidden_dim),
        RMSNorm(self.hidden_dim),
    )
    self.anchor_token_proj = nn.Sequential(
        nn.Linear(20, self.hidden_dim),
        RMSNorm(self.hidden_dim),
    )
    self.sigma_emb = nn.Sequential(
        nn.Linear(1, self.hidden_dim),
        nn.SiLU(),
        nn.Linear(self.hidden_dim, self.hidden_dim),
    )
    seq_len = 1 + int(getattr(cfg, 'predict_checkpoint_len', 10)) + anchor_mu.size(0)
    self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, self.hidden_dim))
    self.drop = nn.Dropout(self.dropout)
    self.blocks = nn.ModuleList([
        NoiseMoEBlock(self.hidden_dim, self.num_heads, self.dropout, self.num_experts, self.top_k)
        for _ in range(self.num_layers)
    ])
    self.norm = RMSNorm(self.hidden_dim)
    self.score_head = nn.Linear(self.hidden_dim, 1)
    self.offset_head = nn.Sequential(
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.GELU(),
        nn.Dropout(self.dropout),
        nn.Linear(self.hidden_dim, 20),
    )
    self.last_aux_losses = {}
    self._init_weights()

  @staticmethod
  def _fallback_anchors():
    steps = torch.arange(1, 11, dtype=torch.float32)
    straight = torch.stack([steps, torch.zeros_like(steps)], dim=-1).reshape(1, 20)
    return straight, torch.ones_like(straight) * 0.05

  def _load_anchor_stats(self, anchor_path):
    if anchor_path is None or str(anchor_path).strip() == '':
      return self._fallback_anchors()
    path = Path(anchor_path)
    if not path.is_absolute():
      path = Path.cwd() / path
    if not path.is_file():
      print(f'MoDETrajectoryDecoder: anchor file not found: {path}. Using a straight fallback anchor.')
      return self._fallback_anchors()

    with path.open('rt', encoding='utf-8') as f:
      data = json.load(f)
    mu = []
    var = []
    for entry in data:
      if 'mu' not in entry:
        continue
      mu.append(entry['mu'][:20])
      if 'var' in entry:
        var.append(entry['var'][:20])
      elif 'variance' in entry:
        var.append(entry['variance'][:20])
      else:
        var.append([0.05] * 20)
    if not mu:
      return self._fallback_anchors()
    anchor_mu = torch.tensor(mu, dtype=torch.float32)
    anchor_var = torch.tensor(var, dtype=torch.float32).clamp_min(1e-6)
    print(f'MoDETrajectoryDecoder: loaded {anchor_mu.size(0)} trajectory anchors from {path}')
    return anchor_mu, anchor_var

  def _init_weights(self):
    nn.init.normal_(self.pos_emb, std=0.01)
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

  def _sample_sigma(self, batch_size, device):
    if not self.training:
      return torch.full((batch_size,), self.sigma_min, device=device)
    log_min = math.log(max(self.sigma_min, 1e-6))
    log_max = math.log(max(self.sigma_max, self.sigma_min + 1e-6))
    return torch.exp(torch.empty(batch_size, device=device).uniform_(log_min, log_max))

  def _make_anchor_prior(self, batch_size, device, dtype):
    mu = self.anchor_mu.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    var = self.anchor_var.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    sigma = self._sample_sigma(batch_size, device).to(dtype=dtype)
    if self.training and self.use_noisy_anchor_prior and self.anchor_noise_scale > 0.0:
      noise = torch.randn_like(mu) * torch.sqrt(var)
      sigma_scale = (sigma / max(self.sigma_data, 1e-6)).view(batch_size, 1, 1)
      anchor_prior = mu + self.anchor_noise_scale * sigma_scale * noise
    else:
      anchor_prior = mu
    return anchor_prior, sigma

  def _collect_aux_losses(self, device):
    losses = []
    for block in self.blocks:
      loss = block.load_balancing_loss()
      if loss is not None:
        losses.append(loss)
    if losses:
      self.last_aux_losses = {'loss_mode_load_balance': torch.stack(losses).mean()}
    else:
      self.last_aux_losses = {'loss_mode_load_balance': torch.zeros((), device=device)}

  def forward(self, encoder_out):
    batch_size = encoder_out.size(0)
    anchor_prior, sigma = self._make_anchor_prior(batch_size, encoder_out.device, encoder_out.dtype)
    noise_token = self.sigma_emb((sigma.log() / 4.0).view(batch_size, 1)).unsqueeze(1)
    traj_tokens = self.traj_token_proj(encoder_out)
    anchor_tokens = self.anchor_token_proj(anchor_prior)
    tokens = torch.cat([noise_token, traj_tokens, anchor_tokens], dim=1)
    tokens = self.drop(tokens + self.pos_emb[:, :tokens.size(1)].to(tokens.dtype))

    noise_cond = noise_token
    for block in self.blocks:
      tokens = block(tokens, noise_cond)
    tokens = self.norm(tokens)

    num_anchors = anchor_prior.size(1)
    anchor_out = tokens[:, -num_anchors:]
    offsets = self.offset_head(anchor_out) * self.offset_scale
    pred_trajectories = anchor_prior + offsets
    scores = self.score_head(anchor_out).squeeze(-1)
    self._collect_aux_losses(encoder_out.device)

    return pred_trajectories.permute(1, 0, 2).contiguous(), scores.permute(1, 0).contiguous()
