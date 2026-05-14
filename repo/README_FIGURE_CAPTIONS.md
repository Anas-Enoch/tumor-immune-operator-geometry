# Figure Captions — Operator-Regime HCC Manuscript

Full legends for Figures 1–7 of:

> Locally constrained immune-interaction fronts at tumour–immune interfaces
> identify operator-regime states associated with immunotherapy response in HCC
> — Anas Enoch (2025)

---

## Figure 1 | Operator-regime framework for tumour–immune interface analysis

**A** Tumour–immune tissue architecture. The interface zone (dashed orange)
separates the tumour core (red fill, irregular boundary) from the surrounding
immune region (blue dots). Spatial transcriptomics sections this boundary
at ~55 µm spot resolution.

**B** Identical programme abundance, different interaction structure. Two
hypothetical interfaces with the same tumour (red) and immune (blue) cell
counts but different spatial field organisation — a gradient pattern (left)
versus a non-gradient antisymmetric mosaic (right). Abundance-based analyses
cannot distinguish these configurations.

**C** Antisymmetric wedge field. For each pair of neighbouring spots $u$, $v$
with programme scores $A$ and $B$, the edge field
$\omega(u,v) = A(u)B(v) - A(v)B(u)$ captures the antisymmetric
spatial coupling (opposing arcs, orange = positive direction,
purple = negative direction).

**D** Hodge decomposition. Solving $L_0\alpha = B_1^\top\omega$ separates
$\omega$ into an exact gradient component (aligned green arrows) and a
coexact rotational component (orange arc with circulating arrow). The coexact
fraction quantifies how much of the interface interaction is non-gradient.

**E** Two interface regimes. The responder regime (orange, upper box) shows
a spatially connected coexact front — a band of organized antisymmetric
interaction. The non-responder regime (blue, lower box) shows scattered
isolated puncta with no coherent front structure.

**F** Six-layer analytical pipeline. Hexagonal nodes represent successive
analytical layers, left to right: Hodge geometry → NCG commutator algebra →
KTS regime transitions → Step 23 finite-range predictability → external
CABO/NIVO replication → IMC cross-modality stress test. Each layer answers
one mechanistic question.

---

## Figure 2 | Responder interfaces form connected coexact fronts

**A** Representative responder section (HCC84_post, S3 state). Coexact
density map with Gaussian KDE contour overlays (4 levels) computed from
the top 30% of coexact-density spots. The coherent density ridge is visible
as a connected front within the interface zone. Color scale: low (pale yellow)
to high (dark red).

**B** Representative non-responder section (HCC72_post, S1 state). Grey
background indicates the negative-control presentation. Coexact density is
diffusely distributed as isolated puncta without contour-level organization.

**C** Hotspot connectivity (LCC fraction). Largest connected component of
coexact hotspot nodes (density > Q75) as a fraction of all interface nodes.
Mann-Whitney p = 0.025; n = 6 responder, 9 non-responder sections.
Each point is one section; horizontal bars indicate group medians.

**D** Spatial score (LCC × Gini). Composite metric combining hotspot
connectivity and spatial concentration. Mann-Whitney p = 0.006.

**E** Pre→post therapy coexact fraction trajectories. Individual patient
trajectories (paired pre/post sections). Red = responders, blue =
non-responders; line colour and opacity consistent across all figures.

**F** Whole-section Hodge (negative control, grey background). Coexact
fraction computed on the full tissue section rather than the interface
subgraph. p = 0.93, confirming that the response signal is
interface-localised. Desaturated colours indicate secondary/negative result.

---

## Figure 3 | Immune–immune algebra, not tumour–myeloid strength, discriminates response

**A** Commutator fraction matrix. Median fraction of total noncommutative
energy for each programme pair (TM, IM, EM, IE), rows: responders (R) vs
non-responders (NR). Color scale: 0 (pale yellow) to 0.9 (dark red).
TM dominates both rows; IE/IM/EM are elevated in R.

**B** TM backbone (stacked bars). Fraction of total commutator energy in the
tumour × myeloid (TM, deep purple) versus all other pairs (pale lilac) for
each of 15 sections. Vertical dashed line at 0.85 marks the approximate
minimum TM fraction. All sections are TM-dominated regardless of response.

**C** Immune–immune commutator enrichment. Swarm plots of IE (T-cell ×
exhaustion, red), IM (T-cell × myeloid, purple), and EM (exhaustion ×
myeloid, dark purple) fractions. Each pair: responders (coloured) left,
non-responders (grey) right. Statistical brackets: Mann-Whitney two-sided.
p values: IE = 0.0048, EM = 0.0048, IM = 0.012.

**D** Operator entropy. Shannon entropy of the commutator norm distribution
across all six programme pairs. Higher entropy = more diverse programme
interaction structure. Mann-Whitney p = 0.050. Patient 76_pre (complete
responder) shows the highest cohort operator entropy (0.994).

**E** Spatial immune programme mosaic (Voronoi tessellation). Left panel:
responder interface — Voronoi cells are heterogeneously coloured red (T-cell)
and blue (exhaustion), reflecting the spatial mosaic of functional states.
Right panel: non-responder — cells are uniformly grey, reflecting
homogeneous immune programme co-distribution.

**F** HCC5NR comparison (CABO/NIVO external cohort). Grouped bar chart
comparing HCC5NR (grey) versus responder median (red) for five metrics: TM,
IE, IM, EM fractions and operator entropy. HCC5NR falls below responders on
all immune–immune metrics despite high spatial organisation.

---

## Figure 4 | KTS state transitions: all non-responders converge to S1

**A** Operator-regime state definitions. Five discrete states with color
coding (S0 grey, S1 blue, S2 green, S3 orange, S4 purple) and classification
criteria (spatial score, P95 coexact density, Gini coefficient).

**B** Representative spatial maps. Inset panels showing coexact density maps
for an S3 responder interface (HCC84, connected front with contours) and an
S1 non-responder interface (HCC72, fragmented puncta, grey background).

**C** Pre→post transition diagram. All 11 patient trajectories; boxes show
pre-therapy state (left) and post-therapy state (right) with section-level
colour. Arrows coloured by response group (red = responder, blue =
non-responder). Dashed blue rectangle highlights the post-therapy S1
convergence cluster for non-responders.

**D** Post-therapy state distribution. Stacked bar chart for responders
(n=5, x=−0.22) and non-responders (n=6, x=+0.22). Bars are colour-coded
by KTS state. Non-responders: 6/6 in S1. Responders: S0 (2), S3 (1), S4 (1), S1 (1).

**E** S1 convergence matrix. 2×2 heatmap; colour scale from RdYlGn_r (red =
high). Rows: responders (n=5) and non-responders (n=6). Columns: "In S1"
and "Non-S1" post-therapy. Fisher exact p = 0.015.

**F** HCC5NR necessary vs sufficient analysis. Two-column table with green
(what HCC5NR has: spatial organisation, connected hotspots) versus red (what
it lacks: immune–immune commutators, finite-range constraint geometry). Bottom
row states therapy outcome. Demonstrates that spatial organisation is necessary
but not sufficient.

---

## Figure 5 | Responder interfaces are locally constrained, not globally periodic

**A** Local k-NN prediction schematic. Colour-graded spot nodes around a
central node (i, black), with arrows showing neighbourhood prediction.
Caption notes that local R² = 0.35–0.84 for responder sections.

**B** Global eigenmode overshoot (Gibbs-like failure). The true coexact
density (black curve, Gaussian peak) versus 5-mode Laplacian eigenmode
reconstruction (red dashed). Red shading = overshoot zone; blue shading =
undershoot. Horizontal dotted line = mean. Failure arises because a compact
local cluster cannot be faithfully reconstructed from smooth global modes.

**C** Local vs global R² scatter. Each point is one section; responders
(red circles) and non-responders (blue circles). Diagonal dashed line =
equal local/global. Vertical/horizontal dotted lines at zero. Red shading
highlights the locally constrained zone (negative global R², positive local R²).
HCC5NR (CABO/NIVO) appears as a blue diamond near the globally-coherent zone.

**D** KTS state hierarchy in local R². Bar chart (S4, S3, S2, S1) coloured
by state. Error bars = SEM. Values above bars: S4 = 0.464, S3 = 0.344,
S2 = 0.240, S1 = 0.150. Horizontal dashed line at zero.

**E** Geometric schematic: locally constrained front vs exclusion ring. Left:
orange-filled ellipse (responder front) with scattered dot hotspots. Caption:
"local R²↑ global R²↓." Right: blue dashed circle (ring topology, HCC5NR
pattern). Caption: "local R² ≈ global R²." Both panels labelled with
response-group text.

**F** Local R² group comparison. Swarm plot; Mann-Whitney p = 0.033.
All 6 responder sections positive; 2 of 9 NR sections negative
(locally unpredictable fields). Horizontal dashed line at zero.

---

## Figure 6 | External CABO/NIVO replication confirms immune–immune algebra

**A** Discovery → CABO/NIVO concordance table. Color-coded rows: orange
(directional replication), green (replicated), teal (new finding).
Symbols: → (directional), ✓ (replicated), ✓✓ (full replication), ✦ (new).

**B** CABO/NIVO Hodge metrics. Swarm plots for coexact fraction, spatial
score, and LCC. Responders (red) left, non-responders (blue) right.
Differences are directional but below significance at n = 4R/3NR.

**C** NCG replication (p = 0.029). Swarm plots for IE fraction (purple) and
operator entropy (red). Minimum achievable p for n = 4 vs 3 (every responder
value exceeds every non-responder value for both metrics).

**D** External KTS state distribution. Stacked bars for responders (n=4) and
non-responders (n=3). Responders: 3 in S3, 1 in S1. Non-responders: 2 in S1,
1 in S4 (HCC5NR).

**E** HCC5NR in local vs global R² space. Local vs global R² scatter for
CABO/NIVO samples; HCC5NR marked as diamond (larger point, black edge).
HCC5NR sits in the globally coherent zone (both R² positive and nearly equal).
Responders cluster in the locally constrained zone (negative global R²).

**F** Necessary conditions for response regime. Four stacked boxes
representing the required conditions: spatial organisation (orange —
necessary but not sufficient), immune–immune algebra (red — required),
locally constrained front geometry (red — required), and S3/S4 KTS state
(blue — associated). HCC5NR satisfies only the first.

---

## Figure 7 | IMC cross-modality stress test: operator regime is emergent at mesoscopic scale

**A** IMC single-cell resolution. Schematic of HCC ROI at true cellular
resolution: red circles = tumour cells (left), blue circles = immune cells
(right), separated by a central boundary. Caption: "No coherent coexact fronts
at cell scale."

**B** Pseudospot coarse-graining pipeline. 4×4 grid of pseudospot bins
(~80 px). Bins coloured by programme content: red (tumour-enriched), blue
(immune-enriched), gold (interface: tumour bin adjacent to immune bin, ⊕
symbol). Caption: "⊕ = interface bin (tumour adj. immune)."

**C** LCC emergence across scales (phase-transition style). Line plot of LCC
fraction vs pseudospot size (40–240 px) for responders (red circles) and
non-responders (blue squares). Yellow shaded band = emergence window (72–108 px).
Responder LCC crosses above non-responder LCC at ~80 px, then both converge
as the interface collapses above 120 px.

**D** Response-signal peak at emergence window. Bar chart of |LCC_R − LCC_NR|
by scale. Yellow bars (65–110 px) highlight the emergence window with peak
effect size at ~80 px. Grey bars outside the window show lower or zero effect.

**E** Interface collapse at macro scale. Number of valid ROIs (≥20 interface
bins) vs pseudospot size. Three shaded regions: sub-mesoscopic (pale orange),
emergence window (yellow), over-coarse-graining (pale blue). ROI count peaks
at 80 px (n=62) and falls to ~3 at 240 px.

**F** Three-scale emergence hierarchy. Schematic with three tiers connected
by downward arrows: (1) ~10 µm / Single cells (yellow-gold box) — "Discrete
puncta / No coherent fronts"; (2) 55–70 µm / Mesoscopic (cream box) —
"Coherent operator fronts / Response signal"; (3) >200 µm / Macroscopic
(pale blue box) — "Interface disappears / Over-homogenised."
