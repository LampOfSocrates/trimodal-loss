"""Tri-modal encoder: frozen CLIP + LoRA adapters on the sketch path only.

Design (paper.md §2.4, A7): one shared visual tower. LoRA modules are toggled
ON for sketch forwards and OFF for photo forwards, so photo/text embeddings
remain exactly frozen-CLIP's — text stays a valid zero-shot semantic bridge.

LoRA is injected into the ViT block MLP linears (c_fc / c_proj). We skip
attention projections because open_clip uses nn.MultiheadAttention, whose
forward reads in_proj/out_proj *parameters* functionally — wrapping those
modules would silently no-op.

backbone="toy": tiny frozen random encoders + a trainable residual sketch
adapter. Zero downloads, seconds to run — smoke-test path only.
"""

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)  # B stays 0 => identity at init
        self.enabled = False

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return y


def inject_lora(visual: nn.Module, rank: int, alpha: float) -> int:
    n = 0
    for block in visual.transformer.resblocks:
        for name in ("c_fc", "c_proj"):
            base = getattr(block.mlp, name)
            if isinstance(base, nn.Linear):
                setattr(block.mlp, name, LoRALinear(base, rank, alpha))
                n += 1
    return n


def set_lora_enabled(model: nn.Module, flag: bool):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = flag


# ------------------------------------------------------------- toy ---------

class _ToyImage(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(64 * 16, dim),
        )

    def forward(self, x):
        return self.net(x)


class _ToyText(nn.Module):
    """Deterministic hashed bag-of-words -> frozen random projection."""

    def __init__(self, dim=128, vocab=4096):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.EmbeddingBag(vocab, dim, mode="mean")

    def forward(self, texts: list[str]):
        idx, offsets = [], [0]
        for t in texts:
            idx += [int(hashlib.md5(w.encode()).hexdigest(), 16) % self.vocab
                    for w in t.lower().split()]
            offsets.append(len(idx))
        device = self.emb.weight.device
        return self.emb(torch.tensor(idx, device=device),
                        torch.tensor(offsets[:-1], device=device))


# ------------------------------------------------------- tri-encoder -------

class TriEncoder(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.backbone = cfg["backbone"]
        if self.backbone == "clip":
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                cfg["clip_model"], pretrained=cfg["clip_pretrained"])
            self.clip = model
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer(cfg["clip_model"])
            self.embed_dim = model.visual.output_dim
            for p in self.clip.parameters():
                p.requires_grad_(False)  # frozen backbone
            n = inject_lora(self.clip.visual, cfg["lora_rank"], cfg["lora_alpha"])
            print(f"[model] CLIP {cfg['clip_model']} frozen; "
                  f"LoRA r={cfg['lora_rank']} injected into {n} MLP linears")
        else:  # toy smoke-test backbone — random features, no downloads
            from torchvision import transforms
            self.embed_dim = 128
            self.img_enc = _ToyImage(self.embed_dim)
            self.txt_enc = _ToyText(self.embed_dim)
            for p in list(self.img_enc.parameters()) + list(self.txt_enc.parameters()):
                p.requires_grad_(False)
            self.sketch_adapter = nn.Linear(self.embed_dim, self.embed_dim)
            nn.init.zeros_(self.sketch_adapter.weight)  # residual, identity at init
            nn.init.zeros_(self.sketch_adapter.bias)
            size = cfg["image_size"]
            self.preprocess = transforms.Compose(
                [transforms.Resize((size, size)), transforms.ToTensor()])
            print("[model] TOY backbone (random frozen nets) — smoke test only")
        self.to(device)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _clip_image(self, imgs, lora: bool, grad: bool):
        set_lora_enabled(self.clip, lora)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            z = self.clip.encode_image(imgs.to(self.device))
        set_lora_enabled(self.clip, False)
        return F.normalize(z, dim=-1)

    def encode_photo(self, imgs):
        if self.backbone == "clip":
            return self._clip_image(imgs, lora=False, grad=False)
        with torch.no_grad():
            return F.normalize(self.img_enc(imgs.to(self.device)), dim=-1)

    def encode_sketch(self, imgs, grad=True):
        if self.backbone == "clip":
            return self._clip_image(imgs, lora=True, grad=grad)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            z = self.img_enc(imgs.to(self.device))
            z = z + self.sketch_adapter(z)
        return F.normalize(z, dim=-1)

    def encode_text(self, texts: list[str]):
        with torch.no_grad():
            if self.backbone == "clip":
                toks = self.tokenizer(texts).to(self.device)
                z = self.clip.encode_text(toks)
            else:
                z = self.txt_enc(texts)
        return F.normalize(z, dim=-1)
