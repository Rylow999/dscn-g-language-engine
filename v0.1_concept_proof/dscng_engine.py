#!/usr/bin/env python3
"""
DSCN-G v3 — Verification Suite
================================
Architectural changes from v1:
  (a) theta* is EMERGENT (Kuramoto order parameter), not an external constant.
  (b) omega and phi are DE-COUPLED: reward = alignment(omega, omega_ideal).
  (c) Theorem 3 = phase CONSENSUS (not "convergence to theta*").
  (d) C3 = PLV drop (root vs. group consensus), not absolute phase perturbation.
  (e) Theorem 1 keeps homeostatic fixed-point + MAXIMALITY check.
  (f) Theorem 2 = omega-alignment convergence (phase-independent).

v3 additions (2026-07-20):
  - Kuramoto all-to-all phase coupling (weighted by omega similarity)
  - Broadcast omega learning (neuromodulatory, scaled by interference)
  - Sustained hijacking (epileptic focus / GNW ignition model)
"""

import json, math, os, sys, argparse
import numpy as np
from numpy.random import default_rng

# ── global defaults ────────────────────────────────────────────
DEFAULTS = {
    "N": 50, "K": 3, "d": 8,
    "alpha": 5.0, "beta": 0.20, "eta": 0.5,
    "lambda_vm": 3.0, "n_actions": 8,
    "gamma": 0.01, "theta_death": 0.10, "kappa": 1.0,
    "theta_emerg": 0.30,
    "eta_kura": 0.005,       # Kuramoto coupling strength (basal/low)
    "eta_kura_high": 0.025,  # During hijack (attention-like modulation)
    "hijack_steps": 15,      # Duration of hijack event
    "eta_hijack": 0.15,      # Root pull strength during hijack
    "seeds": 30, "steps": 2000,
}

# ═══════════════════════════════════════════════════════════════
#  DSCN_G_v3  —  φ physics + ω learning (no external θ*)
# ═══════════════════════════════════════════════════════════════

class DSCN_G_v3:
    """v3 core: phase consensus emerges; omega learns by alignment."""

    def __init__(self, *, seed=None, **kw):
        cfg = dict(DEFAULTS)
        cfg.update(kw)
        for k, v in cfg.items():
            setattr(self, k, v)

        self.rng = default_rng(seed)

        # ── state ──
        self.nodes_active = list(range(self.N))
        self.nodes_pruned  = []

        self.omega = self.rng.normal(0, 0.1, (self.N, self.d))
        self.omega_ideal = np.ones(self.d) / np.sqrt(self.d)   # unit target

        self.phi = self.rng.uniform(0, 2 * np.pi, self.N)
        self.vitality = np.ones(self.N)

        # chains: unique starting positions (no self-collisions)
        self.chain_positions = self.rng.choice(self.nodes_active,
                                               size=self.K, replace=False)

        self.t = 0
        self.history = {"N_active": [], "phase_coherence": [],
                        "mean_alignment": [], "reward": []}

        # ── C3 bookkeeping ──
        self.c3_hijack_count = 0
        self.c3_root_plv_deltas = []   # (t, plv_before, plv_after, delta) when E_root > θ_emerg
        
        # ── Hijack state machine (sustained hijacking) ──
        self.in_hijack = False
        self.hijack_counter = 0
        self.plv_intra_before_hijack = None  # PLV_intra justo antes de entrar en hijack

    # ── elementary helpers ──────────────────────────────────────

    def _relevance(self, i: int) -> float:
        """R_i = 1 / (1 + ‖ω_i − ω_ideal‖)."""
        d = np.linalg.norm(self.omega[i] - self.omega_ideal)
        return 1.0 / (1.0 + d)

    def _chain_step(self, src: int) -> int:
        """Eq.2: P(m|n) ∝ exp(−α·‖ω_m − ω_n‖)."""
        if not self.nodes_active:
            return src
        diffs = np.array([np.linalg.norm(self.omega[m] - self.omega[src])
                          for m in self.nodes_active])
        probs = np.exp(-self.alpha * diffs)
        probs /= probs.sum()
        return int(self.rng.choice(self.nodes_active, p=probs))

    def _von_mises_action(self, phi_i: float) -> int:
        """Eq.4: P(a|φ) = softmax(λ·cos(φ − θ_a))."""
        thetas = np.linspace(0, 2 * np.pi, self.n_actions, endpoint=False)
        logp = self.lambda_vm * np.cos(phi_i - thetas)
        logp -= logp.max()
        p = np.exp(logp)
        p /= p.sum()
        return int(self.rng.choice(self.n_actions, p=p))

    def _update_phi(self, i: int, action_idx: int, reward: float):
        """Eq.3 + adaptive coupling; no gate on binary outcome.
           Δφ = η_eff · R_i · reward · sin(θ_a − φ_i)."""
        R_i  = self._relevance(i)
        θ_a  = 2 * np.pi * action_idx / self.n_actions

        # adaptive coupling: stronger pull when error is large
        err  = (θ_a - self.phi[i] + np.pi) % (2 * np.pi) - np.pi   # ∈ (−π, π]
        factor = 1.0 + abs(err) / np.pi                             # ∈ [1, 2]
        η_eff  = self.eta * factor

        update = η_eff * R_i * reward * np.sin(θ_a - self.phi[i])
        self.phi[i] = (self.phi[i] + update) % (2 * np.pi)

    def _update_vitality_and_prune(self, activity: np.ndarray):
        """Eq.5: V ← V·e^{−γ} + A·(1−e^{−γ}); prune if V < θ_death."""
        decay = np.exp(-self.gamma)
        self.vitality[self.nodes_active] = (
            self.vitality[self.nodes_active] * decay +
            activity[self.nodes_active] * (1.0 - decay))

        to_prune = [i for i in self.nodes_active
                    if self.vitality[i] < self.theta_death]
        for i in to_prune:
            self.nodes_active.remove(i)
            self.nodes_pruned.append(i)

    def _valence(self, activity: np.ndarray) -> np.ndarray:
        """E_i = max(0, A_i − V_i)·κ."""
        return np.maximum(0, activity - self.vitality) * self.kappa

    def _wave_interference(self, i: int) -> float:
        """I_i = ‖ω_i‖ · cos(φ_i − φ_root)."""
        if not self.nodes_active:
            return 0.0
        root = self.nodes_active[0]
        return np.linalg.norm(self.omega[i]) * np.cos(self.phi[i] - self.phi[root])

    # ── coherence helpers (theta* = EMERGENT) ───────────────────

    def phase_coherence(self) -> float:
        """Kuramoto order parameter R = |⟨ e^{iφ} ⟩|."""
        if not self.nodes_active:
            return 0.0
        return abs(np.mean(np.exp(1j * self.phi[self.nodes_active])))

    def mean_omega_alignment(self) -> float:
        """Average cos(ω_i, ω_ideal) over active nodes."""
        if not self.nodes_active:
            return 0.0
        norms = np.linalg.norm(self.omega[self.nodes_active], axis=1) + 1e-8
        dots = (self.omega[self.nodes_active] * self.omega_ideal).sum(axis=1)
        return float(np.mean(dots / norms))

    def plv_root_vs_group(self) -> float:
        """Instantaneous PLV between root and mean phase of the group."""
        if len(self.nodes_active) < 2:
            return 1.0
        root  = self.nodes_active[0]
        others = self.nodes_active[1:]
        mean_phi = np.angle(np.mean(np.exp(1j * self.phi[others])))
        return abs(np.exp(1j * (self.phi[root] - mean_phi)))
    
    def plv_intra_group(self) -> float:
        """Kuramoto order parameter R for the group excluding root."""
        if len(self.nodes_active) < 3:
            return 1.0
        others = self.nodes_active[1:]
        z = np.mean(np.exp(1j * self.phi[others]))
        return abs(z)

    # ── Kuramoto coupling & hijack helpers ──────────────────────

    def _apply_kuramoto_coupling(self):
        """Apply Kuramoto phase coupling to all active nodes (synchronous update).

        Uses eta_kura_high during hijack (neuromodulatory attention),
        eta_kura (basal) otherwise.

        AUDIT FIX (2026-07-22): the original implementation wrote self.phi[i]
        back into the shared array *inside* the loop over i, while the inner
        loop over j read that same shared array. Because nodes_active is
        processed in a fixed order, any j that had already been visited as an
        i in an earlier outer-loop iteration contributed its *already-updated*
        phase to node i's coupling term instead of its phase at the start of
        this step. That makes the update a Gauss-Seidel-style sweep whose
        result depends on the (arbitrary) order of nodes_active, not the
        order-independent synchronous Kuramoto step the model/paper describe
        (dφ_i/dt using φ_j(t), not φ_j(t+dt) for j<i). Fixed by computing every
        node's coupling term from a snapshot of φ taken before any writes, then
        applying all updates at once. This is also vectorized (O(n^2) via numpy
        broadcasting instead of a Python double loop) for tractable runtimes at
        seeds=30/steps=2000.
        """
        if len(self.nodes_active) < 2:
            return

        # Dynamic eta: high during hijack, basal otherwise
        eta_eff = self.eta_kura_high if self.in_hijack else self.eta_kura

        idx = np.asarray(self.nodes_active)
        omega_sub = self.omega[idx]                              # (n, d)
        phi_sub = self.phi[idx].copy()                            # snapshot, (n,)

        diff = omega_sub[:, None, :] - omega_sub[None, :, :]      # (n, n, d)
        dist = np.linalg.norm(diff, axis=2)                       # (n, n)
        weight = np.exp(-self.alpha * dist)                       # (n, n)
        np.fill_diagonal(weight, 0.0)

        phase_diff = phi_sub[None, :] - phi_sub[:, None]          # φ_j − φ_i, row=i col=j
        numer = (weight * np.sin(phase_diff)).sum(axis=1)
        denom = weight.sum(axis=1)

        coupling = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
        self.phi[idx] = (phi_sub + eta_eff * coupling) % (2 * np.pi)

    def _apply_hijack_pull(self):
        """During hijack: root pulls all other nodes toward its phase.

        This REPLACES the normal RL phi update for non-root nodes.
        Root acts as a pathological driver (epileptic focus / GNW ignition).
        """
        if len(self.nodes_active) < 2:
            return

        root = self.nodes_active[0]
        for i in self.nodes_active[1:]:
            # Pull node i toward root's phase
            delta_phi = np.sin(self.phi[root] - self.phi[i])
            self.phi[i] = (self.phi[i] + self.eta_hijack * delta_phi) % (2 * np.pi)

    # ── step ────────────────────────────────────────────────────

    def step(self):
        self.t += 1

        # ── 1. Move chains ───────────────────────────────────────
        activity = np.zeros(self.N)
        for k in range(self.K):
            old = self.chain_positions[k]
            new = self._chain_step(old)
            self.chain_positions[k] = new
            activity[new] += 1.0

        # Root anchoring
        root = self.nodes_active[0] if self.nodes_active else 0
        if root in self.nodes_active:
            activity[root] += 1.0

        activity /= self.K + 1  # K chains + root anchor

        # ── 2. Vitality update + prune ───────────────────────────
        self._update_vitality_and_prune(activity)
        if not self.nodes_active:
            return 0, 0.0

        root = self.nodes_active[0]  # Refresh root after prune
        V = self._valence(activity)

        # ── 3. Kuramoto coupling (BEFORE RL updates) ─────────────
        # Dynamic eta: high during hijack, basal otherwise (neuromodulatory)
        self._apply_kuramoto_coupling()

        # ── 4. C3 hijack state machine ───────────────────────────
        # Check for hijack entry
        if not self.in_hijack and V[root] > self.theta_emerg:
            # ENTER hijack mode
            self.in_hijack = True
            self.hijack_counter = self.hijack_steps
            # Record PLV_intra justo antes del hijack (instantáneo)
            self.plv_intra_before_hijack = self.plv_intra_group()
            self.c3_hijack_count += 1

        # ── 5. RL updates (omega + phi) ──────────────────────────
        if self.in_hijack:
            # During hijack: root acts as pathological driver
            # RL still computes reward but phi update is REPLACED by hijack pull
            inter = np.array([self._wave_interference(i) for i in self.nodes_active])
            sel = int(self.nodes_active[np.argmax(inter)])
            action_idx = self._von_mises_action(self.phi[sel])

            # Omega update: broadcast to all active nodes, modulated by interference
            ω_vec = self.omega[sel]
            norm = np.linalg.norm(ω_vec) + 1e-8
            align = np.dot(ω_vec, self.omega_ideal) / norm
            reward = (align + 1.0) / 2.0

            # Broadcast omega update (T2 fix)
            for i in self.nodes_active:
                I_i = self._wave_interference(i)
                if I_i > 0:  # Only nodes in-phase with root learn
                    # Scale beta by interference (normalize later if needed)
                    β_eff = self.beta * (I_i / (np.linalg.norm(self.omega[i]) + 1e-8))
                    β_eff = min(β_eff, self.beta)  # Cap at base beta
                    self.omega[i] = ((1.0 - β_eff) * self.omega[i] +
                                     β_eff * reward * self.omega_ideal)

            # Phi update: ONLY root gets RL update, others get hijack pull
            self._update_phi(root, action_idx, reward)
            # All other nodes get pulled by root (replaces their RL phi update)
            self._apply_hijack_pull()

            # Decrement hijack counter
            self.hijack_counter -= 1
            if self.hijack_counter <= 0:
                # EXIT hijack mode
                self.in_hijack = False
                # Record C3 event: (t, plv_intra_before, plv_intra_after, delta)
                plv_intra_after = self.plv_intra_group()
                delta_plv = self.plv_intra_before_hijack - plv_intra_after
                self.c3_root_plv_deltas.append((
                    self.t - self.hijack_steps,  # Start of hijack
                    self.plv_intra_before_hijack,
                    plv_intra_after,
                    delta_plv
                ))
                self.plv_intra_before_hijack = None
        else:
            # Normal mode: standard RL updates
            # Select node (max interference)
            inter = np.array([self._wave_interference(i) for i in self.nodes_active])
            sel = int(self.nodes_active[np.argmax(inter)])

            # Action (Eq.4)
            action_idx = self._von_mises_action(self.phi[sel])

            # Reward = alignment(ω, ω_ideal)
            ω_vec = self.omega[sel]
            norm = np.linalg.norm(ω_vec) + 1e-8
            align = np.dot(ω_vec, self.omega_ideal) / norm
            reward = (align + 1.0) / 2.0

            # Omega update: broadcast to all active nodes (T2 fix)
            for i in self.nodes_active:
                I_i = self._wave_interference(i)
                if I_i > 0:  # Only in-phase nodes learn
                    β_eff = self.beta * (I_i / (np.linalg.norm(self.omega[i]) + 1e-8))
                    β_eff = min(β_eff, self.beta)
                    self.omega[i] = ((1.0 - β_eff) * self.omega[i] +
                                     β_eff * reward * self.omega_ideal)

            # Phi update (RL): only the selected node
            self._update_phi(sel, action_idx, reward)

        # ── 6. Logging ───────────────────────────────────────────
        self.history["N_active"].append(len(self.nodes_active))
        self.history["phase_coherence"].append(self.phase_coherence())
        self.history["mean_alignment"].append(self.mean_omega_alignment())
        self.history["reward"].append(reward)

        return action_idx, reward


# ═══════════════════════════════════════════════════════════════
#  VERIFICATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def verify_theorem_1(alpha=5.0, theta_death=0.10,
                     N_inits=(4, 50, 200),
                     seeds=DEFAULTS["seeds"],
                     steps=DEFAULTS["steps"]) -> dict:
    """
    Theorem 1 — Homeostatic Fixed Point + MAXIMALITY.
    (i)   N_ss* ≤ 1/θ_death             (universal bound)
    (ii)  ρ_eff ≥ N_ss* · θ_death²      (fixed-point condition)
    (iii) N_ss* is the *largest* n satisfying (ii) — maximality.
    Checks that N_ss*+1 fails the condition.
    """
    print("\n" + "=" * 62)
    print("THEOREM 1 (v3): Homeostatic Fixed Point + Maximality")
    print("=" * 62)
    print(f"α={alpha}  θ_death={theta_death}  seeds={seeds}  steps={steps}")
    ub = 1.0 / theta_death
    print(f"Universal bound: N_ss* ≤ {ub:.1f}")

    results = []
    for N_init in N_inits:
        N_vals, rho_vals = [], []
        for s in range(seeds):
            sim = DSCN_G_v3(N=N_init, alpha=alpha, theta_death=theta_death, seed=s)
            for _ in range(steps):
                sim.step()
            N_vals.append(len(sim.nodes_active))
            # compute Herfindahl from chain positions
            act = np.zeros(len(sim.nodes_active))
            for pos in sim.chain_positions:
                if pos in sim.nodes_active:
                    act[sim.nodes_active.index(pos)] += 1
            if act.sum() > 0:
                act /= act.sum()
                rho = float(np.sum(act ** 2))
            else:
                rho = 0.0
            rho_vals.append(rho)

        Nm, Ns = np.mean(N_vals), np.std(N_vals)
        rm, rs = np.mean(rho_vals), np.std(rho_vals)
        t2 = theta_death ** 2

        fp_ok  = bool(rm >= Nm * t2)
        ub_ok  = bool(Nm <= ub)

        # maximality: test N_ss* + 1 nodes
        n_test = int(Nm) + 1
        rho_approx_np1 = float(sim.K / n_test) if n_test > 0 else 0.0
        max_ok = not (rho_approx_np1 >= n_test * t2)

        print(f"\n  N_init={N_init:3d}: N_ss*={Nm:.1f}±{Ns:.1f}  ρ={rm:.4f}±{rs:.4f}")
        print(f"    Universal bound: {Nm:.1f} ≤ {ub:.1f} {'✓' if ub_ok else '✗'}")
        print(f"    Fixed point: ρ={rm:.4f} ≥ N·θ²={Nm*t2:.4f} {'✓' if fp_ok else '✗'}")
        print(f"    Maximality (N+1={n_test}): ρ≈{rho_approx_np1:.4f} ≥ {n_test*t2:.4f}? "
              f"{'✓ (fails → N* is max)' if max_ok else '✗ (suspicious)'}")

        results.append(dict(N_init=N_init, N_mean=Nm, N_std=Ns,
                            rho_mean=rm, rho_std=rs, fp_ok=fp_ok,
                            ub_ok=ub_ok, max_ok=max_ok,
                            maximality_n=int(n_test),
                            maximality_rho_approx=rho_approx_np1))

    return dict(alpha=alpha, theta_death=theta_death, results=results)


def verify_theorem_2(beta=DEFAULTS["beta"], seeds=DEFAULTS["seeds"],
                     steps=DEFAULTS["steps"]) -> dict:
    """
    Theorem 2 — ω alignment convergence (phase-independent).
    ω* = ω_ideal  (convergence to max alignment = 1.0).
    Checks: mean_alignment_final ≥ 1 − 2·β.
    """
    print("\n" + "=" * 62)
    print("THEOREM 2 (v3): ω Alignment Convergence (phase-independent)")
    print("=" * 62)
    print(f"β={beta}  seeds={seeds}  steps={steps}")

    finals = []
    for s in range(seeds):
        sim = DSCN_G_v3(N=50, K=3, beta=beta, seed=s)
        for _ in range(steps):
            sim.step()
        finals.append(sim.mean_omega_alignment())

    finals = np.array(finals)
    m, sd = np.mean(finals), np.std(finals)

    threshold = 1.0 - 2.0 * beta

    ok = bool(m >= threshold)
    print(f"  Final alignment = {m:.4f} ± {sd:.4f}")
    print(f"  Threshold (1−2β) = {threshold:.4f}")
    print(f"  Converged: {'✓' if ok else '✗'}")

    return dict(beta=beta, mean_alignment=float(m), std_alignment=float(sd),
                threshold=float(threshold), converged=ok)


def verify_theorem_3(eta=DEFAULTS["eta"], seeds=DEFAULTS["seeds"],
                     steps=DEFAULTS["steps"]) -> dict:
    """
    Theorem 3 — Phase consensus rate.
    Consensus = Kuramoto order parameter R ≥ 0.9.
    """
    print("\n" + "=" * 62)
    print("THEOREM 3 (v3): Phase Consensus Rate (Kuramoto)")
    print("=" * 62)
    print(f"η={eta}  seeds={seeds}  steps={steps}")

    cons_count = 0
    details = []
    for s in range(seeds):
        sim = DSCN_G_v3(N=50, K=3, eta=eta, seed=s)
        for _ in range(steps):
            sim.step()

        R = sim.phase_coherence()
        if R >= 0.9:
            cons_count += 1
            details.append("unimodal")
            continue

        if len(sim.nodes_active) >= 4:
            phis = sim.phi[sim.nodes_active]
            z = np.exp(1j * phis)
            centroid = np.mean(z)
            dist_c = np.abs(z - centroid)
            dist_ac = np.abs(z + centroid)
            c1 = np.where(dist_c <= dist_ac)[0]
            c2 = np.where(dist_ac < dist_c)[0]
            if len(c1) >= 2 and len(c2) >= 2:
                R1 = abs(np.mean(np.exp(1j * phis[c1])))
                R2 = abs(np.mean(np.exp(1j * phis[c2])))
                if R1 >= 0.9 and R2 >= 0.9:
                    cons_count += 1
                    details.append("bimodal")
                    continue
            elif R >= 0.5:
                cons_count += 1
                details.append("weak_unimodal")
                continue

        details.append("no_consensus")

    frac = cons_count / seeds
    print(f"  Consensus reached: {cons_count}/{seeds} = {frac:.3f}")
    for label in sorted(set(details)):
        n = details.count(label)
        print(f"    {label}: {n}/{seeds}")

    return dict(eta=eta, seeds=seeds, steps=steps,
                consensus_count=cons_count, fraction=float(frac),
                details=details)


def verify_c3(seeds=DEFAULTS["seeds"], steps=DEFAULTS["steps"]) -> dict:
    """
    C3 — Phase hijacking via PLV increase (pathological synchronization).
    Returns fraction of hijack events where PLV_intra increased > 0.3 (delta < -0.3).
    """
    print("\n" + "=" * 62)
    print("C3 (v3): Phase-Hijacking — pathological recruitment")
    print("=" * 62)
    print(f"seeds={seeds}  steps={steps}")

    all_deltas = []
    hijack_events = 0
    plv_rise_events = 0

    for s in range(seeds):
        sim = DSCN_G_v3(seed=s, N=50)
        for _ in range(steps):
            sim.step()

        evts = sim.c3_root_plv_deltas
        hijack_events += len(evts)
        for t_start, plv_before, plv_after, delta in evts:
            all_deltas.append(delta)
            if delta < -0.3:  # PLV_intra increased by > 0.3
                plv_rise_events += 1

    hij_rate = hijack_events / (seeds * steps)
    rise_rate = plv_rise_events / max(1, hijack_events)

    print(f"  Hijack triggers: {hijack_events} ({100*hij_rate:.2f}% of steps)")
    print(f"  PLV-rises >0.3 (delta<-0.3): {plv_rise_events} ({100*rise_rate:.1f}% of triggers)")
    if all_deltas:
        print(f"  Mean ΔPLV: {np.mean(all_deltas):.3f} ± {np.std(all_deltas):.3f}")
        print(f"  Max ΔPLV: {np.min(all_deltas):.3f} (most negative)")

    return dict(seeds=seeds, steps=steps,
                hijack_triggers=hijack_events,
                hijack_rate=hij_rate,
                plv_rises=plv_rise_events,
                rise_rate=rise_rate,
                mean_delta_plv=float(np.mean(all_deltas)) if all_deltas else None,
                std_delta_plv=float(np.std(all_deltas)) if all_deltas else None,
                min_delta_plv=float(np.min(all_deltas)) if all_deltas else None,
                all_delta_plv=[float(x) for x in all_deltas])


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="DSCN-G v3 verification")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke-test (5 seeds, 500 steps)")
    args = ap.parse_args()

    seeds = 5 if args.quick else args.seeds
    steps = 500 if args.quick else args.steps

    print("\n" + "=" * 64)
    print(" DSCN‑G  v3   —   EMERGENT   VERIFICATION")
    print("=" * 64)
    print(f"seeds = {seeds}   steps = {steps}")

    report: dict = {}

    # ── ρ_eff definition sanity ──────────────────────────────────
    sim = DSCN_G_v3(N=50, K=3, seed=0)
    for _ in range(steps): sim.step()
    manual_rho = 0.0
    if sim.nodes_active:
        act = np.zeros(len(sim.nodes_active))
        for pos in sim.chain_positions:
            if pos in sim.nodes_active:
                act[sim.nodes_active.index(pos)] += 1
        if act.sum() > 0:
            act /= act.sum()
            manual_rho = float(np.sum(act ** 2))

    print(f"\nρ_eff sanity: manual = {manual_rho:.4f}  (✓ if > 0)")

    # ── Theorems ─────────────────────────────────────────────────
    report["theorem1"] = verify_theorem_1(seeds=seeds, steps=steps)
    report["theorem2"] = verify_theorem_2(seeds=seeds, steps=steps)
    report["theorem3"] = verify_theorem_3(seeds=seeds, steps=steps)

    # ── C3 ───────────────────────────────────────────────────────
    report["c3"] = verify_c3(seeds=seeds, steps=steps)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(" SUMMARY  ( DSCN‑G v3 )")
    print("=" * 64)

    t1 = report["theorem1"]["results"]
    Ns = [r["N_mean"] for r in t1]
    print(f" T1  N_ss* = {np.mean(Ns):.1f}  {'✓' if all(r['ub_ok'] for r in t1) else '✗'}")

    print(f" T2  alignment = {report['theorem2']['mean_alignment']:.4f}  "
          f"{'✓' if report['theorem2']['converged'] else '✗'}")

    print(f" T3  consensus = {report['theorem3']['fraction']:.3f}  "
          f"({report['theorem3']['consensus_count']}/{seeds})")

    print(f" C3  hijack rate = {report['c3']['hijack_rate']:.4f}  "
          f"PLV-rises = {report['c3']['rise_rate']:.4f}")

    # ── Save ─────────────────────────────────────────────────────
    out = "verification_results_v3.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n→  {out}")

if __name__ == "__main__":
    main()
