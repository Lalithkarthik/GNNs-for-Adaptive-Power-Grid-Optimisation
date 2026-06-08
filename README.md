# Hybrid GNN–Solver Framework for Adaptive Power Grid Optimisation

> A physics-informed warm-start approach to Optimal Power Flow  
> **Lanka Sree Lalith Karthik** · Roll No. 23035010505  
> B.Sc. (Hons) Data Science and Artificial Intelligence · IIT Guwahati  
> DA378 Term Project, Trimester 8

---

## Overview

Optimal Power Flow (OPF) is a non-convex, computationally expensive optimisation problem that grid operators must solve thousands of times per day to dispatch generators safely and economically. As renewable penetration increases - with solar and wind outputs fluctuating minute-by-minute - traditional interior-point solvers (PIPS, IPOPT) that take 280–540 ms per solve are no longer adequate for sub-minute variability.

This project presents a **hybrid GNN–Newton-Raphson pipeline** that combines the speed of learned approximation with the safety guarantees of mathematical optimisation:

1. A **Graph Neural Network** is trained offline on AC-OPF ground-truth labels to predict bus voltages and generator setpoints from a load scenario in under 1 ms.
2. The GNN prediction is used as a **warm start** for pandapower's Newton–Raphson solver, which refines it to full AC-feasibility.
3. The GNN is trained with a **physics-informed loss** that combines supervised MSE with a DC power-balance penalty, encouraging predictions that respect Kirchhoff's Current Law before the NR step.

The pipeline achieves **9–14× speedup** over the full PIPS solver, **100% NR feasibility** across all test cases, and voltage-magnitude MAPE of **1.41–1.44%** - well below the 2% planning threshold.

---

## Results at a Glance

| Metric | Value | Verdict |
|---|---|---|
| Voltage magnitude MAPE (GCN) | 1.405% | Below 2% threshold |
| NR feasibility rate | 100% | All warm starts converge |
| Speedup vs. PIPS (case14) | 9.0× | Good speedup compared to PIPS |
| Speedup vs. PIPS (case118) | 13.8× | Speedup grows with network size |
| Wall-clock saving on case14 (GCN) | 49% vs. cold NR | Works well on the case 14 |
| Iteration savings | None observed | Limited to small cases |
| Large-grid validation | Absent | Not performed |

**Recommended architecture: GCN** - matches GATv2 accuracy at lower inference cost with no multi-head attention overhead.

---

## Method

### Problem: AC Optimal Power Flow

$$\min_{P_g, V, \theta} \sum_{i \in \mathcal{G}} C_i(P_{g,i})$$

subject to nonlinear AC power-balance equations, voltage bounds $V_i^{\min} \leq V_i \leq V_i^{\max}$, and thermal limits $|S_{ij}| \leq S_{ij}^{\max}$. The non-convex trigonometric constraints make every solve iterative and expensive.

### Graph Construction

Power grids map naturally to graphs $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

| Element | Representation | Features |
|---|---|---|
| Bus (node) | $v \in \mathcal{V}$ | $[P_d, Q_d, V^{\min}, V^{\max}, \text{type}]$ - dim 7 |
| Line (edge) | $e \in \mathcal{E}$ | $[r, x, b, S^{\max}]$ normalised - dim 4 |
| Target (per node) | - | $[v_m^{\text{pu}}, \theta_n, p_{g,n}]$ - dim 3 |

### Three GNN Architectures

All three share: 3 message-passing layers · hidden dim 128 · residual connections · LayerNorm · 10% dropout · 2-layer MLP decoder.

**GCN** (Kipf & Welling, ICLR 2017) - spectral aggregation via normalised adjacency:
$$\mathbf{H}^{(l+1)} = \sigma\!\left(\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$

**GraphSAGE** (Hamilton et al., NeurIPS 2017) - inductive mean-aggregation with concatenation, enabling generalisation to unseen topologies.

**GATv2** (Brody et al., ICLR 2022) - dynamic attention with explicit edge features, 4 parallel heads averaged per layer.

### Physics-Informed Loss

Standard MSE treats each node independently. A DC power-balance penalty is added to enforce network-level physical consistency:

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \underbrace{\|\mathbf{B}\hat{\boldsymbol{\theta}} - \hat{\mathbf{P}}_{\text{net}}\|_2^2}_{\mathcal{L}_{\text{phys}}}, \qquad \lambda = 0.10$$

where $\mathbf{B}$ is the DC susceptance matrix (assembled per batch; slack bus row excluded). The penalty enforces $\mathbf{B}\theta = \mathbf{P}$ - Kirchhoff's Current Law linearised at unity voltage. At convergence, $\mathcal{L}_{\text{phys}}$ contributes ≈26% of total loss.

**Why λ = 0.10:** Too high destabilises training (physics gradient dominates at epoch 1 when random angles catastrophically violate KCL). Too low effectively ignores the constraint. λ = 0.10 is the stable sweet spot - MSE-dominated but physics-guided.

### Warm-Start Pipeline

At inference:
1. GNN forward pass produces $(\hat{V}_m, \hat{\theta}, \hat{P}_g)$ in under 1 ms.
2. Voltages clamped to $[0.7, 1.3]$ p.u.; generator outputs clamped to $[P_{\min}, P_{\max}]$.
3. Predictions injected into pandapower via `net.res_bus` and `net.gen`.
4. `runpp(init='results')` refines to full AC-feasibility.

---

## Dataset

Ground-truth labels come exclusively from `pandapower.runopp()` (PIPS AC-OPF solver) - no synthetic labels.

| Case | Buses | Lines | Scenarios | Split |
|---|---|---|---|---|
| case14 | 14 | 15 | 500 | 80 / 10 / 10 |
| case30 | 30 | 41 | 500 | 80 / 10 / 10 |
| case118 | 118 | 173 | 500 | 80 / 10 / 10 |

Load scenarios are generated by Gaussian noise (σ = 0.25, clipped ±40%) on base $P_d / Q_d$; infeasible scenarios are discarded. Cases 57, 145, and 300 were excluded due to `runopp()` instability under heavy perturbation, consistent with known solver sensitivity in highly-meshed networks.

---

## Dependencies
```
torch==2.1.0
torch-geometric==2.4.0
pandapower>=2.13
numpy
scipy
matplotlib
```

Hardware used: Google Colab T4 GPU. Training completes in under 2 hours per architecture on this hardware.

---

## Limitations

These are documented to guide any future work building over this repository:

**L1 - Only 3 small IEEE test cases.** All accuracy and speedup figures come from case14, case30, case118. No evidence on stressed or meshed large grids (300+ buses) where warm-starting yields the iteration savings reported in prior literature.

**L2 - No iteration savings observed.** Flat NR start converges in 3–4 iterations on these well-conditioned cases - near the theoretical minimum. The 49% wall-clock saving on case14 comes from reduced Jacobian initialisation overhead in pandapower's `init='results'` code path, not from fewer iterations. Iteration benefits are expected at 300+ bus systems; this work cannot demonstrate them.

**L3 - DC-only physics penalty.** $\mathcal{L}_{\text{phys}}$ enforces only the DC linearisation, ignoring reactive power $Q$, voltage magnitudes, and line losses. No explicit incentive to respect voltage bounds or reactive power balance during training.

**L4 - MAPE undefined for two targets.** See note in Evaluation section above.

**L5 - No objective gap metric.** A 1.4% voltage MAPE could correspond to near-zero or meaningful extra generation cost relative to the true OPF optimum - impossible to determine without computing the dispatch cost gap. This is the most important missing evaluation metric.

---

## Future Work

**F1 - Scale to 300–1,000 bus systems.** Run `runopp()` on case300 and case1354 (PEGASE) with tighter perturbation bounds. This is where iteration-count savings materialise (flat NR needs 6–10+ iterations under heavy stress) and is the single most important validation step.

**F2 - Full AC physics residuals.** Replace the DC penalty with true AC power-flow residuals:
$$\mathcal{L}_{\text{phys}}^{\text{AC}} = \|\Delta P(V, \theta)\|^2 + \|\Delta Q(V, \theta)\|^2$$
More expensive per batch but provides a genuine AC signal including reactive power and voltage magnitudes.

**F3 - Temporal GNNs with renewable forecasts.** A T-GCN or STGCN using rolling wind/solar forecast sequences for warm starts that account for generation trajectory - useful for 5-minute-ahead dispatch under ramp events.

**F4 - Variable / learnable λ.** Schedule λ or make it a learnable parameter. The crossover between physics-dominated and label-dominated regimes reveals where the DC approximation helps vs. hurts.

**F5 - Objective gap metric.** Augment MAPE with extra generation cost of warm-start dispatch vs. the true OPF optimum to quantify real economic impact.

Also on the roadmap: N-1 contingency evaluation, transfer learning across grid topologies, and a real-time SCADA integration prototype.

---

## References

1. W. Huang & M. Chen, "DeepOPF+: A feasibility-optimized deep neural network approach for DC optimal power flow problems," *IEEE Systems Journal*, vol. 16, no. 4, pp. 6507–6517, 2022.
2. D. Owerko, F. Gama & A. Ribeiro, "Optimal power flow using graph neural networks," *arXiv:1910.09658*, 2020.
3. T. Falconer & L. Mones, "Leveraging power grid topology in machine learning assisted optimal power flow," *IEEE Transactions on Power Systems*, vol. 38, no. 3, pp. 2234–2246, 2023.
4. T. N. Kipf & M. Welling, "Semi-supervised classification with graph convolutional networks," *ICLR*, 2017.
5. W. Hamilton, Z. Ying & J. Leskovec, "Inductive representation learning on large graphs," *NeurIPS*, 2017.
6. S. Brody, U. Alon & E. Yahav, "How attentive are graph attention networks?" *ICLR*, 2022.
7. R. D. Zimmerman, C. E. Murillo-Sanchez & R. J. Thomas, "MATPOWER: Steady-state operations, planning, and analysis tools for power systems research and education," *IEEE Transactions on Power Systems*, vol. 26, no. 1, pp. 12–19, 2011.
