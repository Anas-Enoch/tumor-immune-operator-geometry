#!/usr/bin/env python3
"""
build_kts_transitions.py — Operator-Regime KTS Transition Analysis
====================================================================
Maps each patient's tumor–immune interface to a discrete operator-regime
state pre- and post-therapy, then asks whether responders and non-responders
follow different transition edges.

State definitions (Kripke-style discrete operator regimes)
──────────────────────────────────────────────────────────
S0  ABSENT          Low/absent tumor signal or n_interface = 0.
                    Likely complete response or sample too small to classify.
                    Source: status in {low_tumor, skip} or n_interface = 0.

S1  FRAGMENTED      Interface exists but is spatially scattered and
                    low-organization. Low spatial score, low coexact fraction.
                    The "null operator regime" — near-gradient, diffuse.

S2  CONNECTED       Spatially coherent coexact hotspot clusters.
                    High spatial score (LCC×Gini), moderate–high coexact
                    fraction, low absolute intensity (P95 < 50).
                    The canonical pre-therapy responder baseline state.

S3  AMPLIFIED       High-intensity connected interface.
                    High P95 (≥ 50) AND spatial score ≥ 0.08.
                    Extreme local coexact density with spatial coherence.

S4  RESIDUAL        Very concentrated but low-energy interface.
                    High Gini (≥ 0.60) AND low P95 (< 20).
                    Coexact mass concentrated in a few residual nodes —
                    the signature of near-complete response.

S?  UNCLASSIFIABLE  Small interface subgraph from hotspot analysis
                    (n_interface > 0 but spatially unreliable).
                    Kept in the table but excluded from transition counts.

Transition table
────────────────
Each patient's (pre_state, post_state) pair defines one directed edge.
Edge colour = response group.
The key question: do R and NR use different edges?

Classification thresholds
─────────────────────────
SCORE_THRESHOLD   = 0.10   (LCC × Gini separating S1 from S2)
P95_THRESHOLD     = 50.0   (P95 separating S3 from S2)
P95_SCORE_FLOOR   = 0.08   (minimum spatial score for S3 classification)
GINI_RESIDUAL     = 0.60   (Gini threshold for S4)
P95_RESIDUAL_MAX  = 20.0   (P95 upper bound for S4)

Outputs
───────
results_hcc_kts_states.csv         per-sample state assignments
results_hcc_kts_transitions.csv    per-patient transition edges
fig_hcc_kts_graph.png              transition graph
fig_hcc_kts_transition_matrix.png  transition matrix heatmap
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# ── Patient metadata ───────────────────────────────────────────────────────
PATIENT_MAP = {
    'cytassist_70': (85557, 'Responder',    'List A'),
    'cytassist_71': (85558, 'Responder',    'List A'),
    'cytassist_72': (85559, 'Non_Responder','List A'),
    'cytassist_73': (85560, 'Non_Responder','List A'),
    'cytassist_74': (85561, 'Non_Responder','List A'),
    'cytassist_76': (85563, 'Responder',    'List B'),
    'cytassist_79': (85565, 'Non_Responder','List A'),
    'cytassist_83': (85566, 'Responder',    'List B'),
    'cytassist_84': (85567, 'Responder',    'List B'),
    'cytassist_85': (85568, 'Non_Responder','List B'),
    'cytassist_86': (85569, 'Non_Responder','List B'),
}

SCORE_THRESHOLD  = 0.10
P95_THRESHOLD    = 50.0
P95_SCORE_FLOOR  = 0.08
GINI_RESIDUAL    = 0.60
P95_RESIDUAL_MAX = 20.0

STATE_LABELS = {
    'S0': 'S0\nAbsent',
    'S1': 'S1\nFragmented',
    'S2': 'S2\nConnected',
    'S3': 'S3\nAmplified',
    'S4': 'S4\nResidual',
    'S?': 'S?\nUnclass.',
}

STATE_COLOURS = {
    'S0': '#aaaaaa',
    'S1': '#4477aa',
    'S2': '#66bb6a',
    'S3': '#ff9800',
    'S4': '#ab47bc',
    'S?': '#eeeeee',
}

RESP_COLOURS = {'Responder': '#d62728', 'Non_Responder': '#1f77b4'}


# ══════════════════════════════════════════════════════════════════════════════
# STATE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_state(row: pd.Series) -> str:
    """
    Classify a single sample row from results_hcc_coexact_hotspots.csv
    into one of {S0, S1, S2, S3, S4, S?}.

    Priority order:
      1. S0 — absent / low-tumor signal
      2. S4 — collapse residual (high Gini, low P95)
      3. S3 — amplified high-intensity (high P95 + spatial score)
      4. S2 — connected-concentrated (high spatial score, moderate intensity)
      5. S1 — fragmented (default ok)
      6. S? — unclassifiable small interface
    """
    status = str(row.get('status', 'ok'))
    n_iface = int(row.get('n_interface', 0) or 0)

    # S0: complete response or absent
    if status in ('low_tumor', 'low_tumor_signal', 'skip') or n_iface == 0:
        return 'S0'

    # S?: small but non-zero interface
    if status == 'small_interface':
        return 'S?'

    if status != 'ok':
        return 'S?'

    gini    = float(row.get('hotspot_gini',               np.nan) or np.nan)
    p95     = float(row.get('hotspot_p95',                np.nan) or np.nan)
    score   = float(row.get('spatial_score',              np.nan) or np.nan)
    lcc     = float(row.get('largest_component_fraction', np.nan) or np.nan)

    if any(np.isnan(v) for v in [gini, p95]):
        return 'S?'

    # Spatial score: compute if not present
    if np.isnan(score):
        score = gini * lcc if not np.isnan(lcc) else np.nan

    # S4: concentrated residual (high Gini, low absolute energy)
    if gini >= GINI_RESIDUAL and p95 <= P95_RESIDUAL_MAX:
        return 'S4'

    # S3: amplified high-intensity
    sc = score if not np.isnan(score) else 0.
    if p95 >= P95_THRESHOLD and sc >= P95_SCORE_FLOOR:
        return 'S3'

    # S2: connected-concentrated (high spatial score, no extreme P95)
    if not np.isnan(score) and score >= SCORE_THRESHOLD:
        return 'S2'

    # S1: fragmented (default)
    return 'S1'


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_transition_graph(transitions: pd.DataFrame, outpath: Path) -> None:
    """
    Directed graph: nodes = states, edges = pre→post transitions.
    Edge colour = response group; edge width = number of patients.
    """
    try:
        import networkx as nx
    except ImportError:
        print("  networkx not available — skipping graph figure")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title('Operator-Regime KTS Transition Graph\n'
                 'Nodes = interface states; Arrows = pre→post therapy per patient',
                 fontsize=11)

    G = nx.DiGraph()
    ordered_states = ['S0', 'S1', 'S2', 'S3', 'S4', 'S?']
    G.add_nodes_from(ordered_states)

    # Build edge data
    edge_data = defaultdict(lambda: {'R': 0, 'NR': 0, 'patients': []})
    for _, row in transitions.iterrows():
        s_pre  = str(row['state_pre'])
        s_post = str(row['state_post'])
        resp   = 'R' if row['Response'] == 'Responder' else 'NR'
        edge_data[(s_pre, s_post)][resp] += 1
        edge_data[(s_pre, s_post)]['patients'].append(
            row['cytassist_id'].replace('cytassist_','') + f'({resp})')
        G.add_edge(s_pre, s_post)

    # Node positions: semi-circular layout
    pos = {
        'S0': (0.5, 1.0),
        'S1': (0.0, 0.5),
        'S2': (1.0, 0.5),
        'S3': (1.5, 0.0),
        'S4': (0.0, 0.0),
        'S?': (0.5, -0.2),
    }

    # Draw nodes
    for state, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.10,
                                color=STATE_COLOURS[state], zorder=3,
                                ec='white', lw=1.5))
        ax.text(x, y, STATE_LABELS[state],
                ha='center', va='center', fontsize=7.5, zorder=4,
                fontweight='bold', color='white' if state != 'S?' else '#888')

    # Draw edges
    seen_edges = set()
    for (s_pre, s_post), data in edge_data.items():
        if s_pre not in pos or s_post not in pos:
            continue
        x0, y0 = pos[s_pre]
        x1, y1 = pos[s_post]
        n_R  = data['R']
        n_NR = data['NR']
        pats = ', '.join(data['patients'])

        # Mixed: draw two arrows side-by-side
        for n, col, dx_off in [
            (n_R,  '#d62728', -0.02),
            (n_NR, '#1f77b4', +0.02),
        ]:
            if n == 0: continue
            lw = 1.5 + n * 1.0
            dx = x1 - x0; dy = y1 - y0
            norm = max((dx**2+dy**2)**0.5, 1e-8)
            perp = np.array([-dy, dx]) / norm * dx_off
            mid  = np.array([(x0+x1)/2, (y0+y1)/2]) + perp

            if s_pre == s_post:
                # Self-loop
                ax.annotate('', xy=(x0+0.15, y0+0.05),
                            xytext=(x0+0.05, y0+0.15),
                            arrowprops=dict(arrowstyle='->', color=col,
                                           lw=lw,
                                           connectionstyle='arc3,rad=0.5'))
            else:
                ax.annotate('',
                    xy=(x1 + perp[0], y1 + perp[1]),
                    xytext=(x0 + perp[0], y0 + perp[1]),
                    arrowprops=dict(
                        arrowstyle=f'->, head_width={0.12+n*0.04}',
                        color=col, lw=lw,
                        connectionstyle='arc3,rad=0.05'))
            ax.text(mid[0]+perp[0], mid[1]+perp[1], str(n),
                    fontsize=9, color=col, fontweight='bold',
                    ha='center', va='center')

    ax.set_xlim(-0.3, 1.8); ax.set_ylim(-0.45, 1.2)
    ax.axis('off')

    r_patch  = mpatches.Patch(color='#d62728', label='Responder')
    nr_patch = mpatches.Patch(color='#1f77b4', label='Non-Responder')
    ax.legend(handles=[r_patch, nr_patch], fontsize=9,
              loc='lower right', bbox_to_anchor=(1.0, 0.0))

    # State legend
    for i, (s, label) in enumerate(STATE_LABELS.items()):
        ax.add_patch(plt.Circle((-0.25, 0.95 - i*0.18), 0.05,
                                color=STATE_COLOURS[s], zorder=3))
        ax.text(-0.15, 0.95 - i*0.18, label.replace('\n', ' '),
                fontsize=7, va='center')

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close()


def plot_transition_matrix(transitions: pd.DataFrame, outpath: Path) -> None:
    """
    Two side-by-side transition matrices: Responders and Non-Responders.
    Cells show patient count; coloured by response group.
    """
    states = ['S0', 'S1', 'S2', 'S3', 'S4', 'S?']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('KTS Transition Matrix: pre-state → post-state\n'
                 '(rows = pre-therapy state; cols = post-therapy state)',
                 fontsize=11)

    for ax, resp_val, cmap_base in [
        (axes[0], 'Responder',    '#d62728'),
        (axes[1], 'Non_Responder','#1f77b4'),
    ]:
        mat = pd.DataFrame(0, index=states, columns=states)
        sub = transitions[transitions['Response'] == resp_val]
        for _, row in sub.iterrows():
            s_pre, s_post = str(row['state_pre']), str(row['state_post'])
            if s_pre in mat.index and s_post in mat.columns:
                mat.loc[s_pre, s_post] += 1

        # Only show non-zero rows/cols
        active = sorted(set(sub['state_pre'].tolist() + sub['state_post'].tolist()))
        mat_sub = mat.loc[active, active]

        im = ax.imshow(mat_sub.values, cmap='Blues',
                       vmin=0, vmax=max(mat_sub.values.max(), 1))
        ax.set_xticks(range(len(active)))
        ax.set_yticks(range(len(active)))
        ax.set_xticklabels([STATE_LABELS[s].replace('\n',' ')
                            for s in active], fontsize=8)
        ax.set_yticklabels([STATE_LABELS[s].replace('\n',' ')
                            for s in active], fontsize=8)
        ax.set_xlabel('Post-therapy state', fontsize=9)
        ax.set_ylabel('Pre-therapy state', fontsize=9)
        ax.set_title(f"{'Responders' if resp_val=='Responder' else 'Non-Responders'}",
                     fontsize=11, color=cmap_base)

        for i, r in enumerate(active):
            for j, c in enumerate(active):
                v = int(mat_sub.loc[r, c])
                if v > 0:
                    ax.text(j, i, str(v), ha='center', va='center',
                            fontsize=14, fontweight='bold',
                            color='white' if v > mat_sub.values.max()/2 else '#333')

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='KTS operator-regime transition analysis')
    ap.add_argument('--hotspot-csv',  default='results_hcc_coexact_hotspots.csv')
    ap.add_argument('--hodge-csv',    default='results_hcc_hodge_interface_summary.csv',
                    help='Optional: adds coexact_fraction to state context')
    ap.add_argument('--out-states',   default='results_hcc_kts_states.csv')
    ap.add_argument('--out-trans',    default='results_hcc_kts_transitions.csv')
    ap.add_argument('--fig-graph',    default='fig_hcc_kts_graph.png')
    ap.add_argument('--fig-matrix',   default='fig_hcc_kts_transition_matrix.png')
    args = ap.parse_args()

    hot = pd.read_csv(args.hotspot_csv)
    hot['cytassist_id'] = hot['sample_id'].str.extract(r'(cytassist_\d+)')
    hot['timepoint']    = hot['sample_id'].str.extract(r'(pre|post)$')

    # Add spatial score
    hot['spatial_score'] = hot['hotspot_gini'] * hot['largest_component_fraction']

    # Optionally merge coexact_fraction
    try:
        hodge = pd.read_csv(args.hodge_csv)[
            ['sample_id','coexact_fraction','coexact_energy']
        ]
        hot = hot.merge(hodge, on='sample_id', how='left')
    except Exception:
        pass

    # Classify each sample
    hot['state'] = hot.apply(classify_state, axis=1)

    # Add patient metadata
    hot['patient_id'] = hot['cytassist_id'].map(
        lambda x: PATIENT_MAP.get(x, (None,None,None))[0])
    hot['Response']   = hot['cytassist_id'].map(
        lambda x: PATIENT_MAP.get(x, (None,None,None))[1])

    hot.to_csv(args.out_states, index=False)
    print(f"Saved state assignments → {args.out_states}\n")

    # ── State assignments table ────────────────────────────────────────────
    print("="*65)
    print("STATE ASSIGNMENTS  (all 22 samples)")
    print("="*65)
    print(f"  {'sample':<24} {'R/NR':>4} {'n_iface':>8} {'score':>7} "
          f"{'p95':>8} {'gini':>6}  → state")
    print("  " + "─"*70)
    for resp_val in ['Responder','Non_Responder']:
        tag = 'R' if resp_val == 'Responder' else 'NR'
        sub = hot[hot['Response']==resp_val].sort_values('cytassist_id')
        for _, r in sub.iterrows():
            sc   = f"{r.get('spatial_score', np.nan):.3f}" if not pd.isna(
                r.get('spatial_score', np.nan)) else ' nan'
            p95  = f"{r.get('hotspot_p95', np.nan):.1f}" if not pd.isna(
                r.get('hotspot_p95', np.nan)) else '  nan'
            gini = f"{r.get('hotspot_gini', np.nan):.3f}" if not pd.isna(
                r.get('hotspot_gini', np.nan)) else ' nan'
            n_i  = int(r.get('n_interface', 0) or 0)
            print(f"  {r['sample_id']:<24} {tag:>4} {n_i:>8} {sc:>7} "
                  f"{p95:>8} {gini:>6}  → {r['state']}")
        print()

    # ── Build transition table ─────────────────────────────────────────────
    pre_df  = hot[hot['timepoint']=='pre'].set_index('cytassist_id')
    post_df = hot[hot['timepoint']=='post'].set_index('cytassist_id')

    trans_rows = []
    for cid in sorted(PATIENT_MAP.keys()):
        pid, resp, trt = PATIENT_MAP[cid]
        s_pre  = pre_df.loc[cid, 'state']  if cid in pre_df.index  else 'S?'
        s_post = post_df.loc[cid, 'state'] if cid in post_df.index else 'S?'
        n_pre  = int(pre_df.loc[cid, 'n_interface'])  if cid in pre_df.index  else 0
        n_post = int(post_df.loc[cid, 'n_interface']) if cid in post_df.index else 0
        trans_rows.append(dict(
            cytassist_id=cid, patient_id=pid, Response=resp, treatment=trt,
            state_pre=s_pre, state_post=s_post,
            n_interface_pre=n_pre, n_interface_post=n_post,
            transition=f'{s_pre}→{s_post}',
        ))

    trans_df = pd.DataFrame(trans_rows)
    trans_df.to_csv(args.out_trans, index=False)
    print(f"Saved transitions → {args.out_trans}\n")

    # ── Transition table output ────────────────────────────────────────────
    print("="*65)
    print("TRANSITION TABLE  (pre → post per patient)")
    print("="*65)
    print(f"  {'patient':<16} {'R/NR':>4}  pre → post   transition")
    print("  " + "─"*55)
    for resp_val in ['Responder', 'Non_Responder']:
        tag = 'R' if resp_val == 'Responder' else 'NR'
        sub = trans_df[trans_df['Response'] == resp_val]
        for _, r in sub.iterrows():
            print(f"  {r['cytassist_id']:<16} {tag:>4}  "
                  f"{r['state_pre']:>3} → {r['state_post']:<3}   "
                  f"{r['transition']}")
        print()

    # ── Transition counts by response ─────────────────────────────────────
    print("="*65)
    print("TRANSITION EDGES — by response group")
    print("="*65)
    edge_counts = trans_df.groupby(['transition','Response']).size().unstack(fill_value=0)
    print(edge_counts.to_string())

    print("\n  Responder edges:     ",
          trans_df[trans_df['Response']=='Responder']['transition'].tolist())
    print("  Non-Responder edges: ",
          trans_df[trans_df['Response']=='Non_Responder']['transition'].tolist())

    # Key test: do all NR end in S1?
    nr_post = trans_df[trans_df['Response']=='Non_Responder']['state_post']
    r_post  = trans_df[trans_df['Response']=='Responder']['state_post']
    print(f"\n  NR post-therapy states: {nr_post.tolist()}")
    print(f"  R  post-therapy states: {r_post.tolist()}")
    n_nr_s1 = (nr_post == 'S1').sum()
    n_r_s1  = (r_post  == 'S1').sum()
    print(f"\n  NR ending in S1: {n_nr_s1}/{len(nr_post)}")
    print(f"  R  ending in S1: {n_r_s1}/{len(r_post)}")

    # ── Figures ───────────────────────────────────────────────────────────
    plot_transition_graph(trans_df, Path(args.fig_graph))
    plot_transition_matrix(trans_df, Path(args.fig_matrix))
    print(f"\nFigures: {args.fig_graph}  {args.fig_matrix}")


if __name__ == '__main__':
    main()
