#!/usr/bin/env python3
"""
new3: Adaptive Delocalization Defense — single-sample, no ρ, no clustering, no free weights.

Physical picture (Anderson localization + GradSentry spectral geometry)
----------------------------------------------------------------------
Poison samples must encode a shared trigger→target map *and* keep the primary
QA task. On M_clean this dual objective delocalizes the lm_head gradient G
along two complementary axes of the same bipartite matrix:

  (1) Spectral axis — singular values σ of crop(G) become flatter
        Φ_spec = 1 − Gini(σ / ||σ||_1)     (GradSentry-Gini)

  (2) Vocabulary axis — row energies r_v = ||G[v,:]||_2 leak beyond top-m
        Φ_row  = sqrt(1 − mass_top-m(r))   (row-energy gap / leakage)

Two parameter-free candidate scores (each is f(sample_i) only):
        flat = Φ_spec
        DI   = Φ_row × Φ_spec

Channel selection (unsupervised, no labels / no ρ):
  Fit Silverman KDE on each score; measure mode *clarity*
        clarity = 1 − min_{between peaks} dens / sqrt(dens_left · dens_right)
  Prefer the clearer bimodal channel:
        if clarity(flat) ≥ clarity(DI):  use flat  + GradSentry KDE valley
        else:                            use DI    + mode-bounded max-gap τ

Intuition: strong attacks (AddSent / CBA) already separate on spectral flatness
alone; weak BadNets need the joint delocalization index. Clarity picks this
automatically from the score histogram — no fitted coefficients.

Constraints satisfied: FN-oriented thresholding, no poison-rate prior, no
sample-space clustering, overhead ≈ SVD + one row-energy pass.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import autograd
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    confusion_matrix,
)

from .defender import Defender
from openbackdoor.victims import CasualLLMVictim
from openbackdoor.data import getCasualDataloader
from openbackdoor.utils import logger

EPS = 1e-12


class New3Defender(Defender):
    name = "new3"

    def __init__(
        self,
        targetPara: Optional[str] = "lm_head.weight",
        targetDataset: Optional[str] = "webqa",
        svdRank: Optional[int] = 16,
        idRank: Optional[int] = 16,
        threshold: Optional[float] = 0.5,
        autoThreshold: Optional[bool] = True,
        cleanModelPath: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pre = True
        self.targetPara = targetPara
        self.targetDataset = targetDataset
        self.svdRank = int(svdRank)
        self.idRank = int(idRank)
        self.threshold = float(threshold)
        self.autoThreshold = bool(autoThreshold)
        self.cleanModelPath = cleanModelPath
        self.threshold_method = None
        self.computed_threshold = None
        self.selected_score_name = None

    def correct(
        self,
        poison_data: List,
        clean_data: Optional[List] = None,
        model: Optional[CasualLLMVictim] = None,
        clean_model: Optional[CasualLLMVictim] = None,
        **kwargs,
    ):
        detect_model = clean_model if clean_model is not None else model
        if detect_model is None:
            raise ValueError("new3 requires clean_model or model for detection")

        feats = self.get_channels_and_labels(poison_data, detect_model)
        scores, meta_sel = self.select_scores(feats["gap"], feats["flat"])
        poison_labels = feats["poison_labels"]
        self.selected_score_name = meta_sel["score_name"]

        if self.autoThreshold:
            threshold, meta = self._threshold_for_score(scores, prefer=meta_sel["thr_pref"])
            meta = {**meta_sel, **meta}
            logger.info(
                f"new3 selected={meta_sel['score_name']} clarity(flat)={meta_sel['clarity_flat']:.4f} "
                f"clarity(DI)={meta_sel['clarity_di']:.4f} | τ={threshold:.6f} ({meta['method']})"
            )
            self.computed_threshold = threshold
            self.threshold_method = meta["method"]
        else:
            threshold = self.threshold
            self.computed_threshold = threshold
            self.threshold_method = "manual"

        pred = (np.asarray(scores, dtype=np.float64) > threshold).astype(np.int64)
        return self.filtering(poison_data, pred, poison_labels)

    def get_scores_and_labels(self, dataset, model: CasualLLMVictim):
        feats = self.get_channels_and_labels(dataset, model)
        scores, _ = self.select_scores(feats["gap"], feats["flat"])
        return scores.tolist(), feats["poison_labels"]

    def get_channels_and_labels(self, dataset, model: CasualLLMVictim):
        """One forward+backward → (Φ_row, Φ_spec)."""
        data_loader = getCasualDataloader(dataset, batch_size=1, shuffle=False)
        model.train()

        target_param = None
        for n, p in model.llm.named_parameters():
            if self.targetPara in n:
                target_param = p
                target_param.requires_grad_(True)
                logger.info(f"Temporarily enabled gradients for {n}")
                break
        if target_param is None and "lm_head" in self.targetPara:
            target_param = model.llm.get_output_embeddings().weight
            target_param.requires_grad_(True)
            logger.info("Using output embedding weight for lm_head gradient (PeftModel fallback)")
        assert target_param is not None, f"No parameter matching '{self.targetPara}' found"

        gaps, flats, poison_labels = [], [], []
        for batch in tqdm(data_loader, desc="new3 channels (gap, flat)", total=len(data_loader)):
            poison_labels.extend(batch["poison_label"])
            model.zero_grad(set_to_none=True)
            batch_inputs, batch_labels, attention_mask = model.process(batch)
            output = model.forward(
                inputs=batch_inputs, labels=batch_labels, attentionMask=attention_mask
            )
            grad = autograd.grad(output.loss, [target_param], allow_unused=True)[0]
            g = grad.detach()
            if "lora" not in self.targetPara:
                g = g[: g.shape[0] // 8, : g.shape[1] // 8]
            gaps.append(self._row_gap(g, m=self.idRank))
            flats.append(self._spectral_flatness(g))
            del g, grad, output

        target_param.requires_grad_(False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Disabled gradients for {self.targetPara} after detection")

        return {
            "gap": np.asarray(gaps, dtype=np.float64),
            "flat": np.asarray(flats, dtype=np.float64),
            "poison_labels": np.asarray(poison_labels, dtype=np.int64),
        }

    def select_scores(self, gap: np.ndarray, flat: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Pick flat or DI=gap×flat by KDE mode clarity (no labels)."""
        gap = np.asarray(gap, dtype=np.float64)
        flat = np.asarray(flat, dtype=np.float64)
        di = gap * flat
        c_flat = self._mode_clarity(flat)
        c_di = self._mode_clarity(di)
        if c_flat + 1e-12 >= c_di:
            scores = flat
            meta = {
                "score_name": "flat",
                "thr_pref": "kde_valley",
                "clarity_flat": float(c_flat),
                "clarity_di": float(c_di),
            }
        else:
            scores = di
            meta = {
                "score_name": "DI=gap×flat",
                "thr_pref": "max_gap",
                "clarity_flat": float(c_flat),
                "clarity_di": float(c_di),
            }
        return scores, meta

    # keep a thin wrapper used by analysis scripts
    @staticmethod
    def score_from_channels(gap: np.ndarray, flat: np.ndarray) -> np.ndarray:
        return np.asarray(gap, dtype=np.float64) * np.asarray(flat, dtype=np.float64)

    # ---- channels --------------------------------------------------------

    @staticmethod
    def _row_gap(grad: torch.Tensor, m: int = 16, eps: float = EPS) -> float:
        G = grad.detach().float()
        if G.ndim != 2:
            G = G.reshape(G.shape[0], -1)
        r = torch.linalg.vector_norm(G, ord=2, dim=1)
        total = float(r.sum().item()) + eps
        m = int(min(m, r.numel()))
        top = torch.topk(r, k=m, largest=True).values
        rtm = float(top.sum().item()) / total
        return float(np.sqrt(max(1.0 - rtm, 0.0)))

    def _spectral_flatness(self, grad: torch.Tensor, eps: float = EPS) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = grad.detach().float()
        if X.ndim != 2:
            X = X.reshape(X.shape[0], -1)
        X = X.to(device)
        try:
            _, S, _ = torch.svd_lowrank(X, q=self.svdRank)
        except Exception:
            return 1.0
        s = torch.clamp(S, min=eps)
        p = s / s.sum()
        k = float(p.numel())
        if k <= 1.0:
            return 0.0
        p_sorted, _ = torch.sort(p)
        idx = torch.arange(1.0, k + 1.0, device=p.device, dtype=p.dtype)
        gini = 2.0 * torch.sum(idx * p_sorted) / k - (k + 1.0) / k
        gini = torch.clamp(gini, 0.0, 1.0)
        return float((1.0 - gini).item())

    # ---- density / threshold ---------------------------------------------

    def _kde_pack(self, scores, grid_points: int = 2000):
        x = np.asarray(scores, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size < 2:
            raise ValueError(f"Need >= 2 finite scores, got {x.size}")
        std = float(x.std(ddof=1)) if x.size > 1 and x.std(ddof=1) > 0 else 1.0
        bandwidth = max(1.06 * std * (x.size ** (-1.0 / 5.0)), 1e-6)
        grid = np.linspace(float(x.min()), float(x.max()), grid_points)
        dens = np.exp(
            KernelDensity(bandwidth=bandwidth, kernel="gaussian")
            .fit(x.reshape(-1, 1))
            .score_samples(grid.reshape(-1, 1))
        )
        peaks = list(find_peaks(dens, prominence=0.01 * dens.max())[0])
        valleys = list(find_peaks(-dens)[0])
        return x, grid, dens, peaks, valleys, bandwidth

    def _mode_clarity(self, scores) -> float:
        """1 − valley_dens / sqrt(peak_left · peak_right); 0 if <2 modes."""
        try:
            _, grid, dens, peaks, _, _ = self._kde_pack(scores)
        except ValueError:
            return 0.0
        if len(peaks) < 2:
            return 0.0
        order = np.argsort(dens[peaks])[::-1]
        i1, i2 = sorted([peaks[order[0]], peaks[order[1]]])
        valley = float(dens[i1 : i2 + 1].min())
        geom = float(np.sqrt(dens[i1] * dens[i2]) + EPS)
        return float(1.0 - valley / geom)

    def _threshold_for_score(self, scores, prefer: str = "max_gap"):
        x, grid, dens, peaks, valleys, bandwidth = self._kde_pack(scores)
        base = {
            "bandwidth": float(bandwidth),
            "num_peaks": int(len(peaks)),
            "num_samples": int(x.size),
        }

        tau_kde = self._kde_valley_tau(grid, dens, peaks, valleys)
        tau_gap, gap_meta = self._max_gap_tau(x, grid, dens, peaks, valleys)

        if prefer == "kde_valley":
            return float(tau_kde), {**base, "method": "kde_valley", "tau": float(tau_kde)}
        if prefer == "max_gap" and gap_meta is not None:
            return float(tau_gap), {**base, **gap_meta}
        # fallback
        return float(tau_kde), {**base, "method": "kde_valley_fallback", "tau": float(tau_kde)}

    def _find_valley_threshold(self, entropy_list, grid_points=2000, **kwargs):
        """Backward-compatible entry: max-gap with kde fallback (DI path). """
        return self._threshold_for_score(entropy_list, prefer="max_gap")

    @staticmethod
    def _kde_valley_tau(grid, dens, peaks, valleys) -> float:
        """GradSentry-style: first valley to the right of the leftmost peak."""
        if peaks and valleys:
            left = peaks[0]
            right_v = [v for v in valleys if v > left]
            if right_v:
                return float(grid[right_v[0]])
        if valleys:
            return float(grid[int(valleys[int(np.argmin([dens[v] for v in valleys]))])])
        return float(np.median(grid))

    @staticmethod
    def _max_gap_tau(x, grid, dens, peaks, valleys):
        if len(peaks) < 2:
            return None, None
        order = np.argsort(dens[peaks])[::-1]
        i1, i2 = sorted([peaks[order[0]], peaks[order[1]]])
        left_c, right_c = float(grid[i1]), float(grid[i2])
        xs = np.sort(x)
        seg = xs[(xs >= left_c) & (xs <= right_c)]
        if seg.size >= 2:
            diffs = np.diff(seg)
            k = int(np.argmax(diffs))
            tau = float(0.5 * (seg[k] + seg[k + 1]))
            return tau, {
                "method": "kde_modes_max_gap",
                "tau": tau,
                "left_mode": left_c,
                "right_mode": right_c,
                "gap_width": float(diffs[k]),
            }
        mid = [v for v in valleys if i1 < v < i2]
        if mid:
            vi = mid[int(np.argmin([dens[v] for v in mid]))]
        else:
            vi = i1 + int(np.argmin(dens[i1 : i2 + 1]))
        tau = float(grid[vi])
        return tau, {
            "method": "kde_two_tallest_valley",
            "tau": tau,
            "left_mode": left_c,
            "right_mode": right_c,
        }

    def filtering(self, poison_data, preds, poison_labels):
        preds = np.asarray(preds, dtype=np.int64)
        poison_labels = np.asarray(poison_labels, dtype=np.int64)
        assert len(preds) == len(poison_labels) == len(poison_data)

        tn, fp, fn, tp = confusion_matrix(poison_labels, preds, labels=[0, 1]).ravel()
        f1 = f1_score(poison_labels, preds, pos_label=1, zero_division=0) * 100
        prec = precision_score(poison_labels, preds, pos_label=1, zero_division=0) * 100
        rec = recall_score(poison_labels, preds, pos_label=1, zero_division=0) * 100
        acc = accuracy_score(poison_labels, preds) * 100
        logger.info(
            f"new3 detection: F1={f1:.2f} P={prec:.2f} R={rec:.2f} Acc={acc:.2f} "
            f"TP={tp} FP={fp} FN={fn} TN={tn}"
        )
        self.identification_stats = {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Accuracy": round(acc, 2),
            "Precision": round(prec, 2),
            "Recall": round(rec, 2),
            "F1": round(f1, 2),
        }
        kept = [poison_data[i] for i in range(len(poison_data)) if preds[i] == 0]
        return kept
