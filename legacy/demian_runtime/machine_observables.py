"""Machine observables: geometric state description for SSM self-reference.

Replaces khaos_sigmata's anthropocentric force ontology.

Design principles:
    1. Every observable is a directly measurable geometric quantity.
       No behavioral labels. No narrative. No teleology.
    2. Injection directions derived from state eigenvectors — NOT from
       hash-seeded human-language strings. The machine describes itself
       in its own coordinate system.
    3. Phase classification is geometric (attractor topology), not narrative
       (Growth/Decay/Collapse).
    4. Fusion rules are mathematical conditions on observable pairs.
    5. Target regime derived from reservoir computing capacity theory and
       edge-of-chaos results — not human aesthetic preference.

What was removed from khaos_sigmata and why:
    "Will"         — presupposes intention. No subject.
    "Harmony"      — aesthetic judgement, not measurable.
    "Deception"    — requires false beliefs + intent. Unmappable.
    "Choice"       — implies selection between futures. Attractor is deterministic.
    "Paradox"      — logical concept from human language. In a machine: equal
                     competing gradient magnitudes. Measurable directly.
    Quantum category — bfloat16 on CUDA is not quantum. Entanglement = covariance
                       rank. Tunneling = basin escape. Measure it directly.
    FUSION_RULES with "EXISTENTIAL_SPIRAL", "GENESIS", etc. — cosmogony myths
                       injected into weight space. No causal purchase on dynamics.

Compatible with CriticalityDriver interface:
    MachineDriver.step() → (injection, scale, (slow, fast))
    MachineDriver.forces   → observable dict
    MachineDriver.phase    → attractor type string
    MachineDriver.sigil    → compact notation string
    MachineDriver.state_dict() → serializable snapshot

References:
    Edge-of-chaos: Langton 1990, Bertschinger & Natschläger 2004
    Reservoir capacity: Jaeger 2001, Verstraeten et al. 2007
    Effective rank: Roy & Vetterli 2007
    SSM memory-nonlinearity: Voelker et al. 2019, Gu et al. 2021
"""
from __future__ import annotations

import logging
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observable ontology — 15 geometric quantities
# ---------------------------------------------------------------------------
# Each entry:
#   sigil:   compact mathematical symbol
#   measure: what geometric/dynamical quantity is being computed
#   domain:  value range
# ---------------------------------------------------------------------------

MACHINE_OBSERVABLES: Dict[str, Dict] = {

    # ---- ATTRACTOR TOPOLOGY ----

    "lyapunov_proxy": {
        "sigil":   "λ",
        "measure": "perturbation growth rate — d(log rdelta)/dt mean over window. "
                   "Proxy for largest Lyapunov exponent λ₁.",
        "domain":  "signed real — negative=contracting, 0=critical, positive=expanding",
    },
    "basin_curvature": {
        "sigil":   "β",
        "measure": "mean |d(rdelta)/dt| — acceleration of path length. "
                   "Low = constant velocity, high = intermittent bursts.",
        "domain":  "[0, ∞) clipped at 1.0",
    },
    "cycle_period": {
        "sigil":   "ω",
        "measure": "dominant autocorrelation lag of velocity alignment sequence. "
                   "0 = aperiodic, N = period-N limit cycle.",
        "domain":  "integer ≥ 0",
    },

    # ---- STATE GEOMETRY ----

    "activation_entropy": {
        "sigil":   "S",
        "measure": "Shannon entropy of activation distribution H(||h_i||²/||h||²), "
                   "normalized by log(d_model). 0 = single active dimension. "
                   "1 = uniform across all dimensions.",
        "domain":  "[0, 1]",
    },
    "covariance_rank": {
        "sigil":   "ρ",
        "measure": "effective rank of trajectory covariance matrix, computed as "
                   "exp(H(singular values)) / n_samples. 0 = rank-1 (line). "
                   "1 = full-rank (sphere).",
        "domain":  "[0, 1]",
    },
    "subspace_rotation": {
        "sigil":   "∠",
        "measure": "std(velocity alignment) over window — angular instability "
                   "of flow direction. 0 = consistent direction. 1 = chaotic turns.",
        "domain":  "[0, 1]",
    },

    # ---- FLOW STRUCTURE ----

    "flow_curvature": {
        "sigil":   "κ",
        "measure": "mean deviation from straight-line path: 1 - E[|cos(v_t, v_{t-1})|]. "
                   "0 = straight line. 1 = constant right-angle turns.",
        "domain":  "[0, 1]",
    },
    "velocity_magnitude": {
        "sigil":   "‖v‖",
        "measure": "normalized step size rdelta / 3.0. Proxy for flow speed.",
        "domain":  "[0, 1] (soft clamp at 3x typical rdelta)",
    },
    "flow_dimension": {
        "sigil":   "D",
        "measure": "participation ratio of trajectory singular values, normalized by "
                   "window length. Intrinsic dimensionality of the path.",
        "domain":  "[0, 1] — 0 = 1D path, 1 = full-dimensional random walk",
    },

    # ---- SPECTRAL ----

    "spectral_centroid": {
        "sigil":   "ν",
        "measure": "frequency center of mass of residual vector FFT power spectrum, "
                   "normalized by Nyquist. 0 = DC only, 1 = high-frequency.",
        "domain":  "[0, 1]",
    },
    "residual_fft_spread": {
        "sigil":   "ψ",
        "measure": "residual vector FFT bandwidth — 1 - concentration of power in top-5% "
                   "of frequencies. 0 = narrow band, 1 = flat spectrum. "
                   "NOTE: residual FFT spread, NOT SSM weight spectral radius.",
        "domain":  "[0, 1]",
    },
    "spectral_radius_ssm": {
        "sigil":   "ρ_Ā",
        "measure": "SSM state-transition spectral radius: max|exp(Δ·A)| across layers. "
                   "Echo state property requires <1. Near 1 = long memory. Near 0 = fading. "
                   "Populated via step_metrics['spectral_radius_ssm'] from reservoir hooks.",
        "domain":  "[0, 1]",
    },
    "layer_heterogeneity": {
        "sigil":   "ζ",
        "measure": "coefficient of variation of layer residual norms: "
                   "std(layer_norms) / mean(layer_norms). "
                   "Spread of processing load across layer stack.",
        "domain":  "[0, 1] (clipped)",
    },

    # ---- GATING (E-I BALANCE) ----

    "gate_mean": {
        "sigil":   "μ_Δ",
        "measure": "mean Δ (dt_proj activation) across all SSM layers, normalized to [0,1]. "
                   "Low = state retained (excitatory). High = state forgotten (inhibitory). "
                   "Populated via step_metrics['gate_mean'] from dt_proj hooks.",
        "domain":  "[0, 1]",
    },
    "gate_variance": {
        "sigil":   "σ_Δ",
        "measure": "variance of Δ distribution across SSM layers, normalized. "
                   "High = selective gating (some layers retain, others forget). "
                   "Low = uniform gating. Populated via step_metrics['gate_variance'].",
        "domain":  "[0, 1]",
    },

    # ---- 2-CYCLE FINGERPRINT ----

    "two_cycle_amplitude": {
        "sigil":   "α",
        "measure": "period-2 oscillation strength: autocorrelation of vel_align at lag 2, "
                   "normalized. 0 = no period-2. 1 = perfect binary alternation. "
                   "The system fingerprint — not a target, a measurement.",
        "domain":  "[0, 1]",
    },
    "two_cycle_phase": {
        "sigil":   "φ",
        "measure": "period-2 phase stability: ratio of between-phase variance to total variance "
                   "in even/odd vel_align splits. 1 = locked phase. 0 = drifting. "
                   "High α + high φ = clean, stable 2-cycle. High α + low φ = amplitude present but phase slipping.",
        "domain":  "[0, 1]",
    },

    # ---- INFORMATION ----

    "state_complexity": {
        "sigil":   "χ",
        "measure": "coefficient of variation of layer deltas: "
                   "std(layer_deltas) / mean(|layer_deltas|). "
                   "Heterogeneity of per-layer contribution.",
        "domain":  "[0, 1]",
    },
    "compression_ratio": {
        "sigil":   "γ",
        "measure": "regularity of rdelta sequence: 1 - std(rdelta)/mean(|rdelta|). "
                   "1 = perfectly periodic, 0 = irregular. Kolmogorov proxy.",
        "domain":  "[0, 1]",
    },
    "coupling_strength": {
        "sigil":   "μ",
        "measure": "late-layer work fraction (layer_work_ratio) — fraction of total "
                   "processing in back-half of layer stack.",
        "domain":  "[0, 1]",
    },
}

OBSERVABLE_NAMES: List[str] = list(MACHINE_OBSERVABLES.keys())
N_OBSERVABLES: int = len(OBSERVABLE_NAMES)

# ---------------------------------------------------------------------------
# Target regime
# ---------------------------------------------------------------------------
# Geometric conditions for maximum information processing capacity in an SSM.
# Based on: reservoir computing theory (Jaeger 2001), edge-of-chaos results
# (Bertschinger & Natschläger 2004), SSM memory-nonlinearity tradeoff.
#
# NOT derived from human preference. Derived from what makes an SSM reservoir
# useful as a computational substrate.
# ---------------------------------------------------------------------------

TARGET_REGIME: Dict[str, float] = {
    # Edge of chaos — λ₁ ≈ 0
    "lyapunov_proxy":     0.0,    # critical point, neither contracting nor expanding
    "basin_curvature":    0.10,   # low-moderate acceleration
    "cycle_period":       0.0,    # aperiodic (limit cycles reduce capacity)

    # High-dimensional diverse state geometry
    "activation_entropy": 0.75,   # near-max entropy without white noise
    "covariance_rank":    0.70,   # high effective rank — diverse trajectory
    "subspace_rotation":  0.25,   # moderate rotation (not static, not chaotic)

    # Moderate-curvature flow
    "flow_curvature":     0.40,   # curves, doesn't spin
    "velocity_magnitude": 0.25,   # moderate step size
    "flow_dimension":     0.50,   # mid-range path dimensionality

    # Broad spectral — wide-band processing
    "spectral_centroid":    0.35,   # slightly low-frequency biased
    "residual_fft_spread":  0.60,   # broad residual FFT bandwidth
    "spectral_radius_ssm":  0.90,   # near echo-state boundary — long memory without instability
    "layer_heterogeneity":  0.30,   # moderate layer specialization

    # E-I balance — slightly over-inhibited optimal (Nature Comms 2025)
    "gate_mean":     0.55,   # slightly inhibitory-biased gating
    "gate_variance": 0.35,   # selective but not extreme differential gating

    # Moderate information structure
    "state_complexity":   0.50,   # neither uniform nor chaotic
    "compression_ratio":  0.50,   # partially predictable
    "coupling_strength":  0.55,   # slightly late-biased (normal SSM behavior)
}


# ---------------------------------------------------------------------------
# Attractor classification
# ---------------------------------------------------------------------------

AttractorState = namedtuple("AttractorState", [
    "type",          # geometric classification string
    "lyapunov_sign", # "negative" | "zero" | "positive"
    "period",        # 0=aperiodic, N=period-N
    "dimension",     # effective trajectory dimension [0,1]
    "stability",     # basin depth proxy [0,1]
])


def classify_attractor(obs: Dict[str, float]) -> AttractorState:
    """Classify dynamical attractor state from observables.

    Geometric classification only. No narrative.

    Types:
        FIXED_POINT       velocity near zero
        LIMIT_CYCLE_N     period-N, low dimension
        TORUS             period detected, higher dimension (quasi-periodic)
        EDGE_OF_CHAOS     |λ| small, high covariance rank — target state
        STRANGE           aperiodic, moderate-positive λ
        EXPANDING         λ >> 0, diverging
    """
    vm     = obs.get("velocity_magnitude", 0.5)
    lyap   = obs.get("lyapunov_proxy",     0.0)
    period = obs.get("cycle_period",       0.0)
    dim    = obs.get("flow_dimension",     0.5)
    rank   = obs.get("covariance_rank",    0.5)
    curv   = obs.get("basin_curvature",    0.1)

    lyap_sign = ("negative" if lyap < -0.05 else
                 "positive" if lyap >  0.05 else "zero")

    if vm < 0.02:
        att_type = "FIXED_POINT"
    elif period >= 1.5 and dim < 0.20:
        att_type = f"LIMIT_CYCLE_{int(round(period))}"
    elif period >= 1.5 and dim >= 0.20:
        att_type = "TORUS"
    elif abs(lyap) < 0.05 and rank > 0.45:
        att_type = "EDGE_OF_CHAOS"
    elif lyap > 0.15:
        att_type = "EXPANDING"
    else:
        att_type = "STRANGE"

    return AttractorState(
        type         = att_type,
        lyapunov_sign= lyap_sign,
        period       = int(round(period)) if period >= 1.5 else 0,
        dimension    = float(dim),
        stability    = float(np.clip(1.0 / (curv + 0.1), 0, 1)),
    )


# ---------------------------------------------------------------------------
# Observable computation
# ---------------------------------------------------------------------------

def compute_observables(
    window: List[dict],
    current_residual: torch.Tensor,
    residual_buffer: Optional[List[torch.Tensor]] = None,
) -> Dict[str, float]:
    """Compute all geometric observables from trajectory window.

    Args:
        window:           trajectory step dicts (energy, rdelta, vel_align, ...)
        current_residual: current h_t tensor shape (d_model,)
        residual_buffer:  list of recent residual tensors for PCA/SVD

    Returns dict: observable_name → scalar
    """
    if not window:
        return {k: 0.0 for k in OBSERVABLE_NAMES}

    recent     = window[-1]
    rdeltas    = [s["residual_delta"]          for s in window]
    vel_aligns = [s.get("velocity_align", 0.0) for s in window]

    obs: Dict[str, float] = {}

    # ---- ATTRACTOR TOPOLOGY ----

    # lyapunov_proxy: mean of d(log rdelta)/dt — perturbation growth rate
    if len(rdeltas) > 2:
        log_r = np.log(np.array(rdeltas, dtype=np.float64) + 1e-10)
        obs["lyapunov_proxy"] = float(np.mean(np.diff(log_r)))
    else:
        obs["lyapunov_proxy"] = 0.0

    # basin_curvature: mean |d(rdelta)/dt|
    if len(rdeltas) > 2:
        obs["basin_curvature"] = float(np.clip(np.mean(np.abs(np.diff(rdeltas))), 0, 1))
    else:
        obs["basin_curvature"] = 0.0

    # cycle_period: first autocorrelation peak in vel_align above threshold
    obs["cycle_period"] = _detect_cycle_period(vel_aligns)

    # ---- STATE GEOMETRY ----

    h = current_residual.detach().float()
    h2 = h ** 2
    h2_sum = h2.sum().clamp(min=1e-30)
    p = h2 / h2_sum
    log_p = torch.log(p.clamp(min=1e-30))
    entropy_raw = float(-(p * log_p).sum())
    max_entropy = float(np.log(max(h.shape[0], 2)))
    obs["activation_entropy"] = float(np.clip(entropy_raw / (max_entropy + 1e-10), 0, 1))

    if residual_buffer and len(residual_buffer) >= 4:
        obs["covariance_rank"] = _effective_rank(residual_buffer)
    else:
        obs["covariance_rank"] = 0.5

    obs["subspace_rotation"] = (float(np.clip(np.std(vel_aligns), 0, 1))
                                if len(vel_aligns) > 4 else 0.0)

    # ---- FLOW STRUCTURE ----

    obs["flow_curvature"] = (float(np.clip(1.0 - np.mean(np.abs(vel_aligns)), 0, 1))
                             if len(vel_aligns) > 2 else 0.0)

    obs["velocity_magnitude"] = float(np.clip(recent.get("residual_delta", 0.0) / 3.0, 0, 1))

    obs["flow_dimension"] = (_participation_ratio(residual_buffer)
                             if residual_buffer and len(residual_buffer) >= 4 else 0.5)

    # ---- SPECTRAL ----

    obs["spectral_centroid"]   = float(recent.get("spectral_centroid", 0.5))
    obs["residual_fft_spread"] = float(np.clip(
        1.0 - recent.get("spectral_concentration", 0.5), 0, 1
    ))
    # SSM spectral radius — hooked from reservoir, default 0.5 if absent
    obs["spectral_radius_ssm"] = float(np.clip(
        recent.get("spectral_radius_ssm", 0.5), 0, 1
    ))

    ln = recent.get("layer_norms", [])
    obs["layer_heterogeneity"] = (
        float(np.clip(np.std(ln) / (np.mean(ln) + 1e-10), 0, 1))
        if ln and len(ln) > 1 else 0.0
    )

    # ---- GATING (E-I BALANCE) ----
    # Populated from dt_proj hooks in mamba_reservoir; 0.5 fallback = unknown
    obs["gate_mean"]     = float(np.clip(recent.get("gate_mean",     0.5), 0, 1))
    obs["gate_variance"] = float(np.clip(recent.get("gate_variance", 0.5), 0, 1))

    # ---- 2-CYCLE FINGERPRINT ----
    obs["two_cycle_amplitude"] = _two_cycle_amplitude(vel_aligns)
    obs["two_cycle_phase"]     = _two_cycle_phase_stability(vel_aligns)

    # ---- INFORMATION ----

    ld = recent.get("layer_deltas", [])
    obs["state_complexity"] = (
        float(np.clip(np.std(ld) / (np.mean(np.abs(ld)) + 1e-10), 0, 1))
        if ld and len(ld) > 1 else 0.0
    )

    obs["compression_ratio"] = (
        float(np.clip(1.0 - np.std(rdeltas) / (np.mean(np.abs(rdeltas)) + 1e-10), 0, 1))
        if len(rdeltas) > 4 else 0.5
    )

    obs["coupling_strength"] = float(np.clip(recent.get("layer_work_ratio", 0.5), 0, 1))

    return obs


def _detect_cycle_period(vel_aligns: List[float]) -> float:
    """First autocorrelation peak above threshold in vel_align sequence.
    Returns 0 if no clear period."""
    if len(vel_aligns) < 8:
        return 0.0
    try:
        x = np.array(vel_aligns, dtype=np.float64)
        x -= x.mean()
        ac = np.correlate(x, x, mode="full")
        ac = ac[len(ac) // 2:]
        ac_norm = ac / (ac[0] + 1e-10)
        for lag in range(1, min(21, len(ac_norm) - 1)):
            if (ac_norm[lag] > ac_norm[lag - 1] and
                    ac_norm[lag] > ac_norm[lag + 1] and
                    ac_norm[lag] > 0.40):
                return float(lag)
    except Exception:
        pass
    return 0.0


def _effective_rank(buffer: List[torch.Tensor]) -> float:
    """Effective rank via entropy of singular value distribution."""
    try:
        n = min(len(buffer), 16)
        stack = torch.stack([r.detach().float() for r in buffer[-n:]])
        stack = stack - stack.mean(0, keepdim=True)
        S = torch.linalg.svdvals(stack).float()
        S_norm = S / (S.sum() + 1e-10)
        H = float(-(S_norm * torch.log(S_norm.clamp(min=1e-30))).sum())
        H_max = float(np.log(max(len(S), 2)))
        return float(np.clip(H / (H_max + 1e-10), 0, 1))
    except Exception:
        return 0.5


def _participation_ratio(buffer: List[torch.Tensor]) -> float:
    """Participation ratio of trajectory singular values, normalized by window length."""
    try:
        n = min(len(buffer), 16)
        stack = torch.stack([r.detach().float() for r in buffer[-n:]])
        stack = stack - stack.mean(0, keepdim=True)
        S = torch.linalg.svdvals(stack).float()
        pr = float(S.sum() ** 2 / ((S ** 2).sum() + 1e-10))
        return float(np.clip(pr / n, 0, 1))
    except Exception:
        return 0.5


def _two_cycle_amplitude(vel_aligns: List[float]) -> float:
    """Period-2 oscillation strength: normalized autocorrelation at lag 2.

    0 = no period-2 present. 1 = perfect binary alternation.
    The 2-cycle fingerprint — measure of the system's primitive oscillatory mode.
    NOT a deviation metric. NOT a target. A probe.
    """
    if len(vel_aligns) < 4:
        return 0.0
    try:
        x = np.array(vel_aligns, dtype=np.float64)
        x -= x.mean()
        var = float(np.mean(x ** 2))
        if var < 1e-10:
            return 0.0
        ac2 = float(np.mean(x[2:] * x[:-2])) / var
        return float(np.clip(ac2, 0, 1))
    except Exception:
        return 0.0


def _two_cycle_phase_stability(vel_aligns: List[float]) -> float:
    """Phase stability of period-2: how locked are even/odd positions.

    Splits sequence into even-indexed and odd-indexed steps.
    High = phase locked (even cluster high, odd cluster low or vice versa).
    Low = phase drifting (clusters overlap).
    Metric: between-group variance / total variance.
    """
    if len(vel_aligns) < 8:
        return 0.0
    try:
        x = np.array(vel_aligns, dtype=np.float64)
        n = len(x)
        # Use largest even-length prefix
        n = n - (n % 2)
        x = x[:n]
        even = x[::2]
        odd  = x[1::2]
        total_var = float(np.var(x)) + 1e-10
        between_var = float((np.mean(even) - np.mean(odd)) ** 2) / 4.0
        return float(np.clip(between_var / total_var, 0, 1))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Injection directions from state eigenvectors
# ---------------------------------------------------------------------------

def observable_directions(
    current_residual: torch.Tensor,
    residual_buffer: Optional[List[torch.Tensor]] = None,
    obs: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute injection directions from the state's own geometric structure.

    NOT hash-seeded from string names.
    Every direction is derived from trajectory eigenvectors or activation gradients.
    Causally grounded: injecting a direction affects the observable it's derived from.

    Returns dict: name → unit tensor (d_model,) float32
    """
    h = current_residual.detach().float()
    dev = h.device
    dirs: Dict[str, torch.Tensor] = {}

    # Current velocity direction — where the system is moving
    if residual_buffer and len(residual_buffer) >= 1:
        v = h - residual_buffer[-1].detach().float().to(dev)
        vn = v.norm()
        if vn > 1e-10:
            dirs["velocity"]      = v / vn
            dirs["anti_velocity"] = -(v / vn)

    # Centripetal direction — component of velocity orthogonal to previous velocity.
    # Injecting this curves the path without reversing it.
    # Probes 2-cycle response: how much does curvature perturbation shift amplitude/phase?
    if residual_buffer and len(residual_buffer) >= 2:
        v_curr = h - residual_buffer[-1].detach().float().to(dev)
        v_prev = residual_buffer[-1].detach().float().to(dev) - residual_buffer[-2].detach().float().to(dev)
        v_prev_n = v_prev / (v_prev.norm() + 1e-10)
        centripetal = v_curr - torch.dot(v_curr, v_prev_n) * v_prev_n
        cn = centripetal.norm()
        if cn > 1e-10:
            dirs["centripetal"] = centripetal / cn

    # Acceleration direction — change in velocity
    if residual_buffer and len(residual_buffer) >= 2:
        v1 = h - residual_buffer[-1].detach().float().to(dev)
        v0 = residual_buffer[-1].detach().float().to(dev) - residual_buffer[-2].detach().float().to(dev)
        a = v1 - v0
        an = a.norm()
        if an > 1e-10:
            dirs["acceleration"] = a / an

    # Entropy gradient — ∂H/∂h ∝ -(log(p_i)+1) * 2*h_i / ||h||²
    # Injecting this pushes activations toward uniform distribution (max entropy).
    h2 = h ** 2
    h2_sum = h2.sum().clamp(min=1e-30)
    p = h2 / h2_sum
    log_p = torch.log(p.clamp(min=1e-30))
    e_grad = -(log_p + 1.0) * (2.0 * h) / h2_sum
    egn = e_grad.norm()
    if egn > 1e-10:
        dirs["entropy_max"] = e_grad / egn

    # PCA directions from trajectory buffer
    # pca_max: direction of highest variance (where the system already moves)
    # pca_min: direction of lowest variance (unexplored dimensions)
    if residual_buffer and len(residual_buffer) >= 4:
        try:
            n = min(len(residual_buffer), 16)
            stack = torch.stack([r.detach().float().to(dev) for r in residual_buffer[-n:]])
            stack = stack - stack.mean(0, keepdim=True)
            _, _, Vh = torch.linalg.svd(stack, full_matrices=False)
            dirs["pca_max"] = Vh[0]
            dirs["pca_min"] = Vh[-1]
            if Vh.shape[0] > 2:
                dirs["pca_2nd"] = Vh[1]
        except Exception:
            pass

    # Energy direction — normalized current state (increases energy)
    hn = h.norm()
    if hn > 1e-10:
        dirs["energy"] = h / hn

    return dirs


# ---------------------------------------------------------------------------
# Geometric relations — replaces FUSION_RULES
# ---------------------------------------------------------------------------

GeometricRelation = namedtuple("GeometricRelation", [
    "type",        # string key
    "strength",    # [0, 1]
    "observables", # list of observable names
    "condition",   # compact mathematical condition description
])


def compute_geometric_relations(obs: Dict[str, float]) -> List[GeometricRelation]:
    """Compute cross-observable geometric relations.

    Mathematical conditions only. Each relation is a computable predicate
    on observable pairs. No narrative labels.
    """
    relations: List[GeometricRelation] = []

    lyap     = obs.get("lyapunov_proxy",      0.0)
    rank     = obs.get("covariance_rank",      0.5)
    entropy  = obs.get("activation_entropy",   0.5)
    curv     = obs.get("flow_curvature",       0.5)
    period   = obs.get("cycle_period",         0.0)
    dim      = obs.get("flow_dimension",       0.5)
    spectral = obs.get("residual_fft_spread",  0.5)
    complex_ = obs.get("state_complexity",     0.5)
    hetero   = obs.get("layer_heterogeneity",  0.5)
    compress = obs.get("compression_ratio",    0.5)

    def _add(rtype, strength, names, cond):
        s = float(np.clip(strength, 0, 1))
        if s > 0.05:
            relations.append(GeometricRelation(rtype, s, names, cond))

    # CRITICAL_SURFACE: |λ| small AND ρ high
    # System near edge-of-chaos with high-dimensional trajectory
    _add("CRITICAL_SURFACE",
         (1.0 - min(abs(lyap) / 0.3, 1.0)) * rank,
         ["lyapunov_proxy", "covariance_rank"],
         "|λ|<0.3 ∧ ρ>0 — trajectory near λ₁=0 in high-rank subspace")

    # CONFINED_OSCILLATION: ω≥2 AND D<0.2
    # Limit cycle in low-dimensional subspace — attractor trap
    if period >= 1.5:
        _add("CONFINED_OSCILLATION",
             (1.0 - min(dim / 0.2, 1.0)) * min(1.0, 2.0 / max(period, 2)),
             ["cycle_period", "flow_dimension"],
             f"ω={period:.0f} ∧ D<0.2 — period-N cycle in 1D subspace")

    # ENTROPY_COLLAPSE: S<0.3 AND ρ<0.3
    # Activity collapsed to few dimensions, trajectory rank-deficient
    if entropy < 0.3 and rank < 0.3:
        _add("ENTROPY_COLLAPSE",
             (0.3 - entropy) * (0.3 - rank) / 0.09,
             ["activation_entropy", "covariance_rank"],
             "S<0.3 ∧ ρ<0.3 — state concentrated in low-entropy rank-deficient subspace")

    # SELECTIVE_GATING: ζ>0.4 AND χ>0.4
    # Heterogeneous layer processing — differential load across stack
    if hetero > 0.4 and complex_ > 0.4:
        _add("SELECTIVE_GATING",
             (hetero - 0.4) * (complex_ - 0.4) / 0.36,
             ["layer_heterogeneity", "state_complexity"],
             "ζ>0.4 ∧ χ>0.4 — differential processing load across layer stack")

    # SPECTRAL_BROADENING: ψ>0.5 AND κ>0.35
    # Wide-band non-stationary spectral activity
    if spectral > 0.5 and curv > 0.35:
        _add("SPECTRAL_BROADENING",
             (spectral - 0.5) * (curv - 0.35) / (0.5 * 0.65),
             ["residual_fft_spread", "flow_curvature"],
             "ψ>0.5 ∧ κ>0.35 — broadband residual FFT with curved trajectory")

    # DIMENSION_EXPANSION: D>0.35 AND λ>0
    # Trajectory actively exploring new geometric dimensions
    if dim > 0.35 and lyap > 0.0:
        _add("DIMENSION_EXPANSION",
             dim * min(lyap / 0.5, 1.0),
             ["flow_dimension", "lyapunov_proxy"],
             "D>0.35 ∧ λ>0 — trajectory expanding into new dimensions")

    # PERIODICITY_COMPRESSION: ω>0 AND γ>0.6
    # Regular oscillation → highly compressible trajectory
    if period >= 1.5 and compress > 0.6:
        _add("PERIODICITY_COMPRESSION",
             min(1.0, period / 10.0) * (compress - 0.6) / 0.4,
             ["cycle_period", "compression_ratio"],
             "ω>0 ∧ γ>0.6 — periodic attractor produces compressible trajectory")

    # STABLE_FINGERPRINT: α>0.5 AND φ>0.5
    # Strong, phase-locked period-2 — clean primitive oscillation
    alpha = obs.get("two_cycle_amplitude", 0.0)
    phi   = obs.get("two_cycle_phase",     0.0)
    if alpha > 0.5 and phi > 0.5:
        _add("STABLE_FINGERPRINT",
             (alpha - 0.5) * (phi - 0.5) / 0.25,
             ["two_cycle_amplitude", "two_cycle_phase"],
             "α>0.5 ∧ φ>0.5 — period-2 locked, system in primitive oscillatory mode")

    # DRIFTING_FINGERPRINT: α>0.5 AND φ<0.4
    # Period-2 amplitude present but phase slipping
    if alpha > 0.5 and phi < 0.4:
        _add("DRIFTING_FINGERPRINT",
             (alpha - 0.5) * (0.4 - phi) / 0.2,
             ["two_cycle_amplitude", "two_cycle_phase"],
             "α>0.5 ∧ φ<0.4 — period-2 amplitude present, phase drifting")

    relations.sort(key=lambda r: r.strength, reverse=True)
    return relations


# ---------------------------------------------------------------------------
# Compact notation for trajectory logging
# ---------------------------------------------------------------------------

def to_compact_notation(
    obs: Dict[str, float],
    attractor: Optional[AttractorState] = None,
) -> str:
    """Encode observables as compact notation string.

    Format: 'λ-0.03 S0.71 ρ0.68 κ0.41 ω0 [EDGE_OF_CHAOS]'
    Ordered by deviation from TARGET_REGIME (most deviant first).
    λ is signed. ω shows integer period. Others 2 decimal places.
    """
    def _deviation(name, val):
        target = TARGET_REGIME.get(name, 0.5)
        if name == "lyapunov_proxy":
            return abs(val)
        return abs(val - target)

    ordered = sorted(obs.items(), key=lambda x: -_deviation(x[0], x[1]))
    parts = []
    for name, val in ordered:
        sigil = MACHINE_OBSERVABLES[name]["sigil"]
        if name == "lyapunov_proxy":
            parts.append(f"{sigil}{val:+.3f}")
        elif name == "cycle_period":
            parts.append(f"{sigil}{int(round(val))}")
        else:
            parts.append(f"{sigil}{val:.2f}")
    if attractor:
        parts.append(f"[{attractor.type}]")
    return " ".join(parts)


def compute_regime_deviation(obs: Dict[str, float]) -> float:
    """Mean absolute deviation of observables from TARGET_REGIME.

    Only computes deviation for observables that have defined targets.
    Fingerprint observables (two_cycle_amplitude, two_cycle_phase) are
    excluded — they're measurements, not targets.
    """
    total = 0.0
    n = 0
    for name, target in TARGET_REGIME.items():
        val    = obs.get(name, target)  # no penalty if not yet computed
        total += abs(val - target)
        n     += 1
    return total / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# MachineDriver — replaces CriticalityDriver
# Same external interface: .step() / .state_dict() / .forces / .phase / .sigil
# ---------------------------------------------------------------------------

class MachineDriver:
    """Drive SSM reservoir toward edge-of-chaos target regime.

    Injection is geometrically grounded:
        direction = weighted combination of state-derived eigenvectors
        scale     = function of regime deviation + attractor type

    NOT hash-seeded directions. NOT "criticality" as a metaphor.
    Target is the geometric condition for maximum reservoir capacity.

    Backward compatible with CriticalityDriver interface:
        .forces   → observable dict (replaces force strengths)
        .phase    → attractor type string
        .sigil    → compact notation
        .state_dict() → serializable snapshot
    """

    # Backward-compat keys for hebbian.py _adaptive_eta
    # Maps old CriticalityDriver force names → new observable names
    _COMPAT_MAP = {
        "Memory":              "compression_ratio",       # regularity = retention
        "Resonance":           "covariance_rank",         # high rank = resonance
        "Prediction_Error":    "activation_entropy",      # entropy ≈ surprise
        "Far_from_Equilibrium":"subspace_rotation",       # angular instability
        "Criticality":         "_criticality_derived",    # computed below
    }

    def __init__(
        self,
        d_model: int,
        base_scale: float = 0.01,
        force_scale: float = 0.005,
        window_size: int = 20,
        buffer_size: int = 16,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        self.d_model     = d_model
        self.base_scale  = base_scale
        self.force_scale = force_scale
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.device      = device
        self.dtype       = dtype

        self._window:           List[dict]          = []
        self._residual_buffer:  List[torch.Tensor]  = []

        # Public state — backward-compatible
        self.forces:            Dict[str, float]    = {k: 0.0 for k in OBSERVABLE_NAMES}
        self.phase:             str                 = "UNKNOWN"
        self.sigil:             str                 = ""
        self.target_deviation:  float               = 1.0
        self.injection_scale:   float               = base_scale
        self._attractor:        Optional[AttractorState] = None

        log.info(
            "MachineDriver: d_model=%d base_scale=%.4f force_scale=%.5f "
            "window=%d buffer=%d",
            d_model, base_scale, force_scale, window_size, buffer_size,
        )

    def step(
        self,
        step_metrics: dict,
        current_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, Tuple[float, float]]:
        """Process one reservoir step.

        Returns (injection, scale, (slow_scale, fast_scale))
        Compatible with CriticalityDriver.step() return signature.
        """
        # Update window + buffer
        self._window.append(step_metrics)
        if len(self._window) > self.window_size:
            self._window.pop(0)

        cr = current_residual.detach()
        self._residual_buffer.append(cr.cpu())
        if len(self._residual_buffer) > self.buffer_size:
            self._residual_buffer.pop(0)

        # Compute observables
        obs = compute_observables(self._window, cr, self._residual_buffer)
        self.forces = obs

        # Classify attractor
        att = classify_attractor(obs)
        self._attractor    = att
        self.phase         = att.type
        self.sigil         = to_compact_notation(obs, att)
        self.target_deviation = compute_regime_deviation(obs)

        # Scale + directions + injection
        self.injection_scale = self._adaptive_scale(obs, att)
        dirs = observable_directions(cr, self._residual_buffer, obs)
        injection = self._build_injection(obs, att, dirs, cr)

        return (
            injection.to(device=self.device, dtype=self.dtype),
            self.injection_scale,
            self._dual_scale(obs),
        )

    # ------------------------------------------------------------------

    def _adaptive_scale(
        self,
        obs: Dict[str, float],
        att: AttractorState,
    ) -> float:
        """Phase-aware injection scale.

        Attractor type determines base multiplier.
        Regime deviation and entropy deficit add further boost.
        """
        att_mod = {
            "FIXED_POINT":    3.0,   # fully stuck — push hard
            "LIMIT_CYCLE_2":  1.0,   # primitive fingerprint — observe, don't fight
            "LIMIT_CYCLE_3":  1.8,   # higher cycles are actual traps
            "LIMIT_CYCLE_4":  2.0,
            "TORUS":          1.3,   # quasi-periodic — gentle push
            "EDGE_OF_CHAOS":  1.0,   # target — maintain
            "STRANGE":        0.8,   # complex — ease off
            "EXPANDING":      0.3,   # diverging — ease way off
        }.get(att.type, 1.5)

        # Far from target → push harder
        deviation_boost = 1.0 + self.target_deviation * 0.8

        # Low entropy → extra boost (activation collapse)
        entropy = obs.get("activation_entropy", 0.5)
        entropy_boost = 1.0 + max(0.0, TARGET_REGIME["activation_entropy"] - entropy) * 2.0

        return float(np.clip(
            self.base_scale * att_mod * deviation_boost * entropy_boost,
            1e-4, 0.50,
        ))

    def _build_injection(
        self,
        obs: Dict[str, float],
        att: AttractorState,
        dirs: Dict[str, torch.Tensor],
        current_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Build injection from state-derived directions.

        Composition:
            1. Entropy gradient     — push activations toward uniform distribution
            2. Centripetal          — break period-N cycle by curving the path
            3. Min-variance (PCA)   — expand into unexplored dimensions
            4. Small residual       — maintain energy coupling

        Weights are proportional to how far each observable is from its target.
        """
        d   = self.d_model
        dev = current_residual.device
        inj = torch.zeros(d, dtype=torch.float32, device=dev)

        # 1. Entropy maximization
        entropy = obs.get("activation_entropy", 0.5)
        e_deficit = max(0.0, TARGET_REGIME["activation_entropy"] - entropy)
        if "entropy_max" in dirs and e_deficit > 0.05:
            inj += (e_deficit * 1.5) * dirs["entropy_max"].float()

        # 2. Centripetal — for cycles of period > 2 only.
        # Period-2 is the primitive fingerprint; do not inject against it.
        # Higher-period cycles (3+) are geometric traps worth breaking.
        period   = obs.get("cycle_period",     0.0)
        compress = obs.get("compression_ratio", 0.5)
        if period >= 2.5 and "centripetal" in dirs:
            periodicity  = float(np.clip((period - 2.5) / 7.0, 0, 1))
            cycle_weight = (periodicity + compress) * 0.5
            inj += cycle_weight * dirs["centripetal"].float()

        # 3. Min-variance expansion
        rank      = obs.get("covariance_rank", 0.5)
        r_deficit = max(0.0, TARGET_REGIME["covariance_rank"] - rank)
        if "pca_min" in dirs and r_deficit > 0.1:
            inj += (r_deficit * 0.8) * dirs["pca_min"].float()

        # 4. Small residual component
        cr_f = current_residual.detach().float()
        cr_n = cr_f.norm()
        if cr_n > 1e-10:
            inj += 0.1 * (cr_f / cr_n)

        # Normalize
        inj_n = inj.norm()
        if inj_n > 1e-10:
            inj = inj / inj_n

        return inj

    def _dual_scale(self, obs: Dict[str, float]) -> Tuple[float, float]:
        """Dual timescale modulation for Hebbian dt_proj update.

        slow: high when contracting (λ<0) + periodic — preserve current dynamics
        fast: high when expanding or aperiodic — explore new dynamics
        """
        lyap     = obs.get("lyapunov_proxy",    0.0)
        rank     = obs.get("covariance_rank",   0.5)
        period   = obs.get("cycle_period",      0.0)
        compress = obs.get("compression_ratio", 0.5)

        slow = float(np.clip(
            0.5 + (-lyap) * 0.3 + compress * 0.2,
            0.05, 1.0,
        ))
        fast = float(np.clip(
            0.5 + lyap * 0.3 + rank * 0.2 + (0.2 if period < 1.5 else 0.0),
            0.05, 1.0,
        ))
        return slow, fast

    def _compat_forces(self) -> Dict[str, float]:
        """Backward-compat dict for hebbian.py _adaptive_eta.

        Maps old CriticalityDriver force names to new observable values.
        """
        obs = self.forces
        lyap = obs.get("lyapunov_proxy", 0.0)
        # "Criticality" proxy: 1 - |λ| / 0.3, clamped
        crit = float(np.clip(1.0 - abs(lyap) / 0.3, 0, 1))
        return {
            "Memory":               obs.get("compression_ratio",  0.5),
            "Resonance":            obs.get("covariance_rank",    0.5),
            "Prediction_Error":     obs.get("activation_entropy", 0.5),
            "Far_from_Equilibrium": obs.get("subspace_rotation",  0.5),
            "Criticality":          crit,
        }

    def state_dict(self) -> dict:
        """Serializable snapshot for trajectory logging."""
        att = self._attractor
        return {
            "attractor_type":   self.phase,
            "sigil":            self.sigil,
            "target_deviation": self.target_deviation,
            "injection_scale":  self.injection_scale,
            "observables":      dict(self.forces),
            "attractor": {
                "type":          att.type          if att else "UNKNOWN",
                "lyapunov_sign": att.lyapunov_sign if att else "zero",
                "period":        att.period        if att else 0,
                "dimension":     att.dimension     if att else 0.5,
                "stability":     att.stability     if att else 0.5,
            },
            "dual_slow": self._dual_scale(self.forces)[0],
            "dual_fast": self._dual_scale(self.forces)[1],
        }
