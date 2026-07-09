"""The four loss framings (paper.md §2).

Common signature: loss_module(s, v, t, instance_ids) -> (loss, components)
with s, v, t L2-normalized (B, d) sketch/photo/text embeddings.

Every variant shares the same weighted pairwise-InfoNCE base so the geometric
terms are compared as *additions* to an identical contrastive core — the
comparison isolates the geometry, not a reweighting.
"""

import torch
import torch.nn as nn

from . import sphere


def info_nce(a, b, temperature, ids=None):
    """Symmetric InfoNCE. If ids given, same-instance off-diagonal pairs
    (e.g. two sketches of the same photo in one batch) are masked out of the
    denominator — they are false negatives, common under grouped sampling."""
    logits = a @ b.t() / temperature
    n = logits.shape[0]
    labels = torch.arange(n, device=logits.device)
    if ids is not None:
        same = ids.unsqueeze(0) == ids.unsqueeze(1)
        mask = same & ~torch.eye(n, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(mask, float("-inf"))
    return 0.5 * (nn.functional.cross_entropy(logits, labels)
                  + nn.functional.cross_entropy(logits.t(), labels))


class _PairwiseBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.temp = cfg["temperature"]
        self.w = (cfg["w_sv"], cfg["w_st"], cfg["w_vt"])

    def pairwise(self, s, v, t, ids):
        l_sv = info_nce(s, v, self.temp, ids)
        l_st = info_nce(s, t, self.temp, ids)
        l_vt = info_nce(v, t, self.temp, ids)
        total = self.w[0] * l_sv + self.w[1] * l_st + self.w[2] * l_vt
        return total, dict(nce_sv=l_sv.item(), nce_st=l_st.item(),
                           nce_vt=l_vt.item())


class BaselineLoss(_PairwiseBase):
    """Baseline to beat: pairwise InfoNCE sum + adaptive-margin sketch-photo
    triplet. margin_ij = m0 * (1 - cos(t_i, t_j)): semantically close pairs
    (per the text bridge) get a smaller required margin (paper.md A6 — one
    simple instantiation of 'adaptive margin')."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.m0 = cfg["triplet_margin0"]
        self.w_trip = cfg["w_triplet"]

    def forward(self, s, v, t, ids):
        total, comp = self.pairwise(s, v, t, ids)
        sim = s @ v.t()                                   # (B, B)
        pos = sim.diag()
        diff = ids.unsqueeze(0) != ids.unsqueeze(1)       # valid negatives
        margin = self.m0 * (1 - t @ t.t())
        viol = (margin + sim - pos.unsqueeze(1)).clamp_min(0)
        trip = (viol * diff).sum() / diff.sum().clamp_min(1)
        comp["triplet"] = trip.item()
        return total + self.w_trip * trip, comp


class FrechetLoss(_PairwiseBase):
    """Pairwise base + pull s, v, t toward their own Frechet/Karcher mean.
    Anchor detached (paper.md §2.1 / A3): 'pull toward current consensus';
    contrastive base supplies the between-concept repulsion."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lam = cfg["lambda_frechet"]
        self.exact = cfg["karcher_exact"]
        self.iters = cfg["karcher_iters"]

    def forward(self, s, v, t, ids):
        total, comp = self.pairwise(s, v, t, ids)
        triple = torch.stack([s, v, t], dim=1)            # (B, 3, d)
        with torch.no_grad():
            m = sphere.karcher_mean(triple, iters=self.iters, exact=self.exact)
        d2 = sphere.geodesic_distance(triple, m.unsqueeze(1)) ** 2
        fre = d2.mean()
        comp["frechet"] = fre.item()
        return total + self.lam * fre, comp


class TriangleLoss(_PairwiseBase):
    """Pairwise base + spherical-excess (Girard area) regularizer. Couples
    the three angles the pairwise sum optimizes independently. Shape-only
    term (E=0 <=> near-a-great-circle, NOT close) — see paper.md §2.2."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lam = cfg["lambda_triangle"]

    def forward(self, s, v, t, ids):
        total, comp = self.pairwise(s, v, t, ids)
        excess = sphere.spherical_triangle_excess(s, v, t).mean()
        comp["triangle_excess"] = excess.item()
        return total + self.lam * excess, comp


class VMFLoss(_PairwiseBase):
    """Pairwise base + vMF NLL with a LEARNABLE per-modality concentration
    kappa_m (softplus-parameterized). mu per concept = normalized mean of the
    instance's in-batch members across modalities (detached — same rationale
    as the Frechet anchor). Use with GroupedInstanceSampler so instances have
    several members in-batch (Sketchy's >=5 sketches/photo structure).

    kappa acts as the learned uncertainty weight: gradient on x scales with
    kappa_m, and d(NLL)/d(kappa_m) = A_d(kappa_m) - mean alignment, so
    kappa_sketch converges low iff sketches really are more dispersed.
    """

    MODALITIES = ("sketch", "photo", "text")

    def __init__(self, cfg, embed_dim):
        super().__init__(cfg)
        self.lam = cfg["lambda_vmf"]
        self.d = embed_dim
        # kappa_mode (EXP3): "learned" = gradient-trained free parameter (the
        # original framing); "ema" = kappa is a detached EMA of the per-modality
        # Banerjee kappa-hat, i.e. a STATISTIC of the data's concentration, not a
        # parameter — only the mean direction drives gradients.
        self.mode = cfg.get("vmf_kappa_mode", "learned")
        self.ema = cfg.get("vmf_kappa_ema", 0.9)
        k0 = torch.tensor(float(cfg["kappa_init"]), dtype=torch.float64)
        # stable inverse softplus (log(expm1(k)) overflows for k >~ 89):
        # log(e^k - 1) = k + log1p(-e^-k)
        raw0 = (k0 + torch.log1p(-torch.exp(-k0))).float()
        self.raw_kappa = nn.Parameter(raw0.repeat(3))
        self.register_buffer("ema_kappa", torch.full((3,), float(cfg["kappa_init"])))

    def kappas(self):
        if self.mode == "ema":
            return self.ema_kappa
        return nn.functional.softplus(self.raw_kappa) + 1e-4

    @staticmethod
    def _banerjee(rbar, d):
        rbar = rbar.clamp(1e-4, 1 - 1e-4)
        return rbar * (d - rbar ** 2) / (1 - rbar ** 2)

    def forward(self, s, v, t, ids):
        total, comp = self.pairwise(s, v, t, ids)
        members = torch.cat([s, v, t], dim=0)              # (3B, d)
        mids = ids.repeat(3)
        uniq, inv = torch.unique(mids, return_inverse=True)
        with torch.no_grad():                              # detached mu_c
            sums = torch.zeros(len(uniq), members.shape[1], device=members.device)
            sums.index_add_(0, inv, members)
            mu = sphere.normalize(sums)
        mu_per = mu[inv]                                   # (3B, d)
        B = s.shape[0]

        if self.mode == "ema":
            # kappa as a statistic: per-modality mean alignment to the concept
            # mean -> Banerjee kappa-hat -> EMA. Detached; no grad through kappa.
            with torch.no_grad():
                align = (members * mu_per).sum(-1)         # (3B,)
                for i in range(3):
                    rbar = align[i * B:(i + 1) * B].mean()
                    khat = self._banerjee(rbar, self.d)
                    self.ema_kappa[i].mul_(self.ema).add_((1 - self.ema) * khat)
            kap = self.ema_kappa
        else:
            kap = self.kappas()

        kap_per = torch.cat([kap[i].expand(B) for i in range(3)])
        nll = sphere.vmf_nll(members, mu_per, kap_per, d=self.d).mean()
        comp["vmf_nll"] = nll.item()
        for name, k in zip(self.MODALITIES, kap):
            comp[f"kappa_{name}"] = k.item()
        return total + self.lam * nll, comp


def build_loss(name: str, cfg, embed_dim: int) -> nn.Module:
    return dict(
        baseline=lambda: BaselineLoss(cfg),
        frechet=lambda: FrechetLoss(cfg),
        triangle=lambda: TriangleLoss(cfg),
        vmf=lambda: VMFLoss(cfg, embed_dim),
    )[name]()
