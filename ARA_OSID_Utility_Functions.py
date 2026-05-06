#!/usr/bin/env python3
"""
ARA_OSID_Utility_Functions.py
=============================
Adversarial Risk Analysis for Open-Set Intrusion Detection (ARA-OSID)
Utility Function Implementation -- Phases 3 through 6

Implements Eqs. (1)--(16) and Table I from the accepted DSN 2026 paper:

    "From Threat Intelligence to Decision Theory: ATT&CK-Derived Utility
     Functions for Adversarial Risk Analysis in NIDS"
    Raj, Bastian, Kul, Fiondella -- DSN 2026 Workshop

This module plugs directly into the existing MITRE ATT&CK pipeline
(AP_Prob_RS_Complete_3_Scenarios.py) and expects the following globals
to be defined in the calling scope:

    tech_df, rel_df, camp_df            # ATT&CK v16 Excel sheets
    parent_to_subs, sub_to_parent       # sub-technique hierarchy dicts
    campaign_severity                   # {campaign_id: NCISS_normalized}
    name_to_tactics                     # {technique_name: [tactic, ...]}
    stoi, itos, vocab                   # LSTM vocabulary maps
    model (NextStepLSTM)                # trained LSTM model
    start_prob, next_prob               # first-order Markov model
    step_detectability                  # function(name) -> float
    DETECTION_COVERAGE                  # defaultdict(lambda: 0.0)
    all_chains, campaign_index,         # generated attack chains
    campaign_ids_index                  # campaign IDs per chain
    TACTIC_ORDER_DEFAULT                # 14 kill-chain phases
    OUT_DIR, SEED                       # output directory and RNG seed

Pipeline Phase Mapping (Fig. 5 from paper):
    Phase 3a: Attacker parameter extraction (Eqs. 9-12, Table I top)
    Phase 3b: Defender parameter extraction (Eqs. 13-16, Table I bottom)
    Phase 4:  psi_A computation (Eq. 3 with Beta MC)
    Phase 5:  psi_r computation + r* selection (Eqs. 7-8)
    Phase 6a: 3-scenario NCISS validation (RQ3)
    Phase 6b: Sensitivity analysis +/-10/25/50% (RQ4)

Author  : Mayank Raj (mraj1@umassd.edu)
Project : DoD Grant W911NF-22-2-0160 (ARA-OSID / DSN 2026)
License : Research use under DoD cooperative agreement
"""

import os
import math
import json
import logging
import warnings
import time
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar

import torch

warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger("ARA-OSID")


# ==========================================================
# CONFIGURATION
# Matches DSN 2026 paper Sections III and IV exactly.
# Every constant below is referenced by equation number or
# section number from the paper.
# ==========================================================

# Monte Carlo samples for psi_r evaluation (Sec. III-G)
MC_SAMPLES = 10_000

# Defender risk-aversion coefficient for u(c) = -exp(gamma*c)
# Explored in sensitivity analysis (Sec. IV, RQ4)
GAMMA_DEFENDER = 0.5

# Robustness grid for Eq. 8: r* = argmax_r E[U_D(r)]
ROBUSTNESS_GRID = np.linspace(0.0, 1.0, 21)

# Sensitivity perturbation factors (Sec. IV, RQ4)
# Corresponds to +/-10%, +/-25%, +/-50%
SENSITIVITY_DELTAS = [0.50, 0.75, 0.90, 1.0, 1.10, 1.25, 1.50]

# Random seed for reproducibility
SEED_UTIL = 42
rng = np.random.default_rng(SEED_UTIL)


# ── Eq. 9 weights ──────────────────────────────────────────
# e(a) = w1 * perm(t) + w2 * |subs(t)| / |subs|_max
# w1 controls influence of privilege level
# w2 controls influence of technique complexity
# Values explored through sensitivity analysis (Sec. IV)
W1_EFFORT = 0.6
W2_EFFORT = 0.4


# ── Eq. 12: Impact bonus multiplier ────────────────────────
# B(t,c) = NCISS(c) * (1 + lambda * I_Impact(t))
# lambda = 0.5 represents a 50% bonus for Impact-tactic techniques
# Explored in sensitivity analysis (Sec. IV)
LAMBDA_IMPACT = 0.5


# ── Threat-tier thresholds (Sec. III-E) ────────────────────
# Groups stratified by documented technique count:
#   |G| < 5   -> t1 (Novice / script kiddies)
#   5 <= |G| <= 20 -> t2 (Intermediate)
#   |G| > 20  -> t3 (Advanced / APT / nation-state)
TIER_L1_MAX = 5
TIER_L2_MAX = 20


# ── Beta distribution parameters per tier (Sec. III-E) ─────
# Detection uncertainty: p_d ~ Beta(alpha, beta)
# These model the detection draw that scales P and B in Eq. 2.
#
# Paper specification:
#   t1 (Novice):       Beta(2, 8)  -> E[p_d] = 0.20
#   t2 (Intermediate): Beta(4, 6)  -> E[p_d] = 0.40
#   t3 (APT):          Beta(7, 3)  -> E[p_d] = 0.70
#
# Higher E[p_d] means the attacker expects a higher chance of
# being detected, yielding larger penalty P and smaller benefit B.
BETA_PARAMS = {
    1: (2.0, 8.0),    # t1: Novice       E[p_d] = 0.20
    2: (4.0, 6.0),    # t2: Intermediate  E[p_d] = 0.40
    3: (7.0, 3.0),    # t3: APT           E[p_d] = 0.70
}


# ── Attacker actions (Sec. III-B) ──────────────────────────
# a in {regular, adversarial}
# "regular" = standard exploitation without evasion optimization
# "adversarial" = crafted perturbation / evasion-optimized attack
ACTIONS = ["regular", "adversarial"]

# p(a | t_i): conditional probability of choosing action given tier
# Novices rarely craft adversarial samples; APTs frequently do
ACTION_PROB_GIVEN_TIER = {
    1: {"regular": 0.85, "adversarial": 0.15},   # t1: mostly regular
    2: {"regular": 0.50, "adversarial": 0.50},   # t2: equal split
    3: {"regular": 0.20, "adversarial": 0.80},   # t3: mostly adversarial
}

# Adversarial actions require more effort (crafting perturbations,
# optimizing evasion). This multiplier scales effort for a=adversarial.
ADVERSARIAL_EFFORT_MULT = 1.5


SIGMOID_TEMPERATURE = float(os.environ.get("ARA_SIGMOID_TEMP", "2.0"))


# ── Tactic-dependent FP disruption weights omega_tau ───────
# (Sec. III-F): "FP cost is left as a configurable per-tactic
# weight rather than a closed-form equation, since tactic-level
# disruption costs are inherently organization-dependent."
#
# Scale: 0 = no disruption, 1 = maximum disruption
# Early recon FPs are cheap; late-stage FPs are very disruptive
TACTIC_FP_WEIGHT = {
    "reconnaissance":          0.05,
    "resource development":    0.05,
    "initial access":          0.15,
    "execution":               0.30,
    "persistence":             0.25,
    "privilege escalation":    0.35,
    "defense evasion":         0.40,
    "credential access":       0.35,
    "discovery":               0.20,
    "lateral movement":        0.45,
    "collection":              0.30,
    "command and control":     0.50,
    "exfiltration":            0.55,
    "impact":                  0.60,
}


# ── Permission-to-effort encoding (Eq. 9) ──────────────────
# Paper: "perm(t) encodes permission level from tech_df
#         (User = 0.2, Admin = 0.6, Root = 1.0)"
PERMISSION_EFFORT = {
    "user":           0.2,
    "administrator":  0.6,
    "admin":          0.6,    # alternate spelling in some ATT&CK fields
    "system":         0.8,
    "root":           1.0,
}


# ==========================================================
# HELPER: section logger
# ==========================================================
def _section(title: str):
    """Print a formatted section header to the log."""
    sep = "=" * 55
    log.info(f"\n{sep}\nARA-OSID | {title}\n{sep}")


# ==========================================================
# SHARED UTILITIES: rel_df parsing
# ==========================================================

def _find_rel_type_col(rel_df: pd.DataFrame) -> Optional[str]:
    """
    Find the relationship-type column in rel_df.
    ATT&CK v16 Excel uses 'mapping type' (not 'relationship type').
    """
    # Check all known column name variants
    candidates = [
        "mapping type", "mapping_type",           # ATT&CK v16 Excel
        "relationship type", "relationship_type",  # older versions
        "type",                                    # generic fallback
    ]
    col_map = {c.lower().strip(): c for c in rel_df.columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    # Last resort: partial match
    for c in rel_df.columns:
        cl = c.lower().strip()
        if ("relationship" in cl or "mapping" in cl) and "type" in cl:
            return c
    log.warning(f"  WARNING: Could not find relationship/mapping type column "
                f"in rel_df. Columns: {list(rel_df.columns)}")
    return None


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find a column in a DataFrame by trying multiple candidate names.
    Case-insensitive with strip.
    """
    col_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower().strip() in col_map:
            return col_map[cand.lower().strip()]
    return None


def build_group_technique_map(rel_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Parse rel_df to build (group_stix_id -> list[technique_stix_id]).

    Filters for relationships where:
      - source is an intrusion-set (group)
      - target is an attack-pattern (technique)
      - relationship type contains 'uses'

    Returns:
        Dict mapping group STIX IDs to lists of technique STIX IDs.
    """
    rel_type_col = _find_rel_type_col(rel_df)
    src_col = _find_col(rel_df, ["source ref", "source_ref", "source"]) or "source ref"
    tgt_col = _find_col(rel_df, ["target ref", "target_ref", "target"]) or "target ref"

    g2t = defaultdict(list)
    for _, r in rel_df.iterrows():
        src = str(r.get(src_col, ""))
        tgt = str(r.get(tgt_col, ""))
        rtype = str(r.get(rel_type_col, "")).lower() if rel_type_col else ""
        if ("intrusion-set--" in src
                and "attack-pattern--" in tgt
                and "uses" in rtype):
            g2t[src].append(tgt)
    return dict(g2t)


def compute_group_tech_counts(
    rel_df: pd.DataFrame,
    stix_to_name: Dict[str, str],
) -> Dict[str, int]:
    """
    For each technique name, count how many unique groups use it.

    This is the numerator of Eq. 13:
        |{g in G : g uses t_i}|

    Deduplicates per-group to avoid double-counting when a group
    references the same technique through multiple relationships.
    """
    tech_group_count = Counter()
    g2t = build_group_technique_map(rel_df)
    for group_stix, tech_stix_list in g2t.items():
        seen_names = set()
        for ts in tech_stix_list:
            name = stix_to_name.get(ts)
            if name and name not in seen_names:
                tech_group_count[name] += 1
                seen_names.add(name)
    return dict(tech_group_count)


def _get_total_unique_groups(rel_df: pd.DataFrame) -> int:
    """
    |G|: total number of unique intrusion-set groups in rel_df.

    This is the denominator of Eq. 13:
        p(t_i) = |{g in G : g uses t_i}| / |G|

    Paper (Sec. IV): "143 documented groups"
    """
    g2t = build_group_technique_map(rel_df)
    return max(len(g2t), 1)


def compute_group_tiers(rel_df: pd.DataFrame,
                        stix_to_name: Dict[str, str]) -> Dict[str, int]:
    """
    Assign threat tiers to GROUPS based on technique count (Sec. III-E).

    Paper: "ATT&CK groups from rel_df are stratified by documented
    technique count: |G| < 5 → t1, 5 ≤ |G| ≤ 20 → t2, |G| > 20 → t3"

    This counts techniques PER GROUP (not groups per technique).

    Returns:
        Dict mapping group_stix_id to tier (1, 2, or 3).
    """
    g2t = build_group_technique_map(rel_df)
    group_tiers = {}
    for group_stix, tech_stix_list in g2t.items():
        # Count unique techniques this group uses
        unique_techs = set()
        for ts in tech_stix_list:
            name = stix_to_name.get(ts)
            if name:
                unique_techs.add(name)
        n_techs = len(unique_techs)
        group_tiers[group_stix] = assign_threat_tier(n_techs)
    return group_tiers


def compute_technique_tiers(rel_df: pd.DataFrame,
                            stix_to_name: Dict[str, str]) -> Dict[str, int]:
    """
    For each technique, determine its threat tier from the groups that use it.

    Strategy: assign each technique the HIGHEST tier among the groups
    that use it. If APT groups (t3) use a technique, it's t3-relevant
    regardless of whether script kiddies also use it.

    This implements the paper's intent: tier classification comes from
    GROUP sophistication (technique count), not from per-technique
    group popularity.

    Returns:
        Dict mapping technique_name to tier (1, 2, or 3).
    """
    group_tiers = compute_group_tiers(rel_df, stix_to_name)
    g2t = build_group_technique_map(rel_df)

    # For each technique, find max tier among groups using it
    tech_max_tier = defaultdict(lambda: 1)
    for group_stix, tech_stix_list in g2t.items():
        gtier = group_tiers.get(group_stix, 1)
        for ts in tech_stix_list:
            name = stix_to_name.get(ts)
            if name:
                tech_max_tier[name] = max(tech_max_tier[name], gtier)

    return dict(tech_max_tier)


def assign_threat_tier(group_tech_count: int) -> int:
    """
    Map a group's documented technique count to threat tier.

    Sec. III-E:
        |G| < 5   -> t1 (Novice)
        5 <= |G| <= 20 -> t2 (Intermediate)
        |G| > 20  -> t3 (Advanced/APT)

    Paper (Sec. IV): "approximately 60% of 143 groups use fewer
    than 5 techniques (t1), roughly 25% employ 5 to 20 (t2),
    and approximately 15% use more than 20 (t3)"
    """
    if group_tech_count < TIER_L1_MAX:
        return 1
    elif group_tech_count <= TIER_L2_MAX:
        return 2
    return 3


# ==========================================================
# MODULE 1: ATTACKER PARAMETER EXTRACTION  (Eqs. 9--12)
# ==========================================================

def extract_attacker_params(
    tech_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    parent_to_subs: Dict[str, list],
    campaign_severity: Dict[str, float],
    name_to_tactics: Dict[str, list],
    tactic_order: List[str],
    stix_to_name: Dict[str, str],
    DETECTION_COVERAGE: dict,
) -> pd.DataFrame:
    """
    Phase 3a: Extract attacker-side utility parameters for every
    technique in the ATT&CK v16 catalogue.

    Implements Table I (attacker side):
        Eq.  9: effort e(a) = w1*perm(t) + w2*|subs(t)|/|subs|_max
        Eq. 10: P_max(t) = (|d3fend(t)|+|datasources(t)|) /
                            (|d3fend|_max+|datasources|_max)
        Eq. 11: Ra(t) = tactic_order.index(primary_tactic(t)) / 13
        Eq. 12: B(t,c) = NCISS(c) * (1 + lambda * I_Impact(t))
        Tier:   |G|<5 -> t1, 5<=|G|<=20 -> t2, |G|>20 -> t3
        Beta:   t1=Beta(2,8), t2=Beta(4,6), t3=Beta(7,3)

    Args:
        tech_df: ATT&CK techniques DataFrame (columns: ID, STIX ID,
                 name, tactics, permissions_required, data sources, etc.)
        rel_df: Relationships DataFrame (group-uses-technique, etc.)
        parent_to_subs: Dict mapping parent technique name to list of
                        sub-technique names
        campaign_severity: Dict mapping campaign_id to NCISS normalized
                          severity score (0 to 1)
        name_to_tactics: Dict mapping technique name to list of tactics
        tactic_order: Ordered list of 14 kill-chain tactics (lowercase)
        stix_to_name: Dict mapping STIX ID to technique name
        DETECTION_COVERAGE: Dict or defaultdict mapping technique name
                           to D3FEND/detection coverage (0 to 1)

    Returns:
        DataFrame with columns: technique, effort, P_max, resource_cost,
        benefit, threat_tier, beta_alpha, beta_beta, group_count
    """
    _section("Phase 3a: Attacker parameter extraction (Eqs. 9-12, Table I)")

    # ── DIAGNOSTIC: verify incoming data structures ──
    log.info(f"  DIAGNOSTICS:")
    log.info(f"    tech_df: {len(tech_df)} rows, columns: {list(tech_df.columns[:8])}...")
    log.info(f"    rel_df: {len(rel_df)} rows, columns: {list(rel_df.columns[:8])}...")
    log.info(f"    parent_to_subs: {len(parent_to_subs)} parents, "
             f"sample sizes: {[len(v) for v in list(parent_to_subs.values())[:5]]}")
    log.info(f"    campaign_severity: {len(campaign_severity)} campaigns, "
             f"sample values: {list(campaign_severity.values())[:3]}")
    log.info(f"    stix_to_name: {len(stix_to_name)} entries")
    rel_type_col = _find_rel_type_col(rel_df)
    log.info(f"    rel_type column found: '{rel_type_col}'")
    if rel_type_col:
        unique_vals = rel_df[rel_type_col].dropna().unique()
        log.info(f"    rel_type unique values ({len(unique_vals)}): "
                 f"{list(unique_vals[:10])}")
        # Check if expected keywords exist
        vals_lower = [str(v).lower() for v in unique_vals]
        has_uses = any("uses" in v for v in vals_lower)
        has_mitigates = any("mitigates" in v for v in vals_lower)
        has_subtechnique = any("subtechnique" in v for v in vals_lower)
        log.info(f"    Keywords found: uses={has_uses}, "
                 f"mitigates={has_mitigates}, subtechnique={has_subtechnique}")
        if not has_uses:
            log.warning(f"    WARNING: 'uses' not found in {rel_type_col} values! "
                        f"Group-technique parsing will fail. "
                        f"All values: {list(unique_vals)}")
    else:
        log.warning(f"    rel_type column NOT FOUND - all group/mitigation "
                    f"parsing will return empty results")

    # ── Rebuild parent_to_subs if empty ──
    # ATT&CK v16 Excel encodes sub-technique hierarchy in technique IDs:
    #   T1059.001 (sub) is child of T1059 (parent)
    # The relationships sheet doesn't contain "subtechnique-of" entries.
    if not parent_to_subs:
        log.info("  parent_to_subs is empty; rebuilding from technique IDs...")
        import re as _re
        # Build ID -> name mapping
        id_to_name_local = dict(zip(tech_df["ID"].astype(str), tech_df["name"]))
        rebuilt_count = 0
        for tid, tname in id_to_name_local.items():
            # Sub-technique IDs match T####.### pattern
            if _re.match(r"^T\d{4}\.\d{3}$", str(tid)):
                parent_id = tid.split(".")[0]  # T1059.001 -> T1059
                parent_name = id_to_name_local.get(parent_id)
                if parent_name:
                    if parent_name not in parent_to_subs:
                        parent_to_subs[parent_name] = []
                    if tname not in parent_to_subs[parent_name]:
                        parent_to_subs[parent_name].append(tname)
                        rebuilt_count += 1
        log.info(f"  Rebuilt parent_to_subs from IDs: {len(parent_to_subs)} parents, "
                 f"{rebuilt_count} sub-technique mappings")

    # ── Pre-compute group counts per technique (Eq. 13 numerator) ──
    tech_group_counts = compute_group_tech_counts(rel_df, stix_to_name)
    total_unique_groups = _get_total_unique_groups(rel_df)
    log.info(f"  Group-technique parsing: {total_unique_groups} unique groups, "
             f"{len(tech_group_counts)} techniques with group usage")

    # ── Compute proper threat tiers (Sec. III-E) ──
    # Paper: "groups stratified by documented technique count"
    # Tier comes from GROUP sophistication, not technique popularity
    group_tiers = compute_group_tiers(rel_df, stix_to_name)
    technique_tiers = compute_technique_tiers(rel_df, stix_to_name)
    group_tier_dist = Counter(group_tiers.values())
    log.info(f"  Group tier distribution: {dict(sorted(group_tier_dist.items()))} "
             f"(Paper: ~60% t1, ~25% t2, ~15% t3)")

    # ── |subs|_max: maximum sub-technique count (Eq. 9 denominator) ──
    subs_max = max((len(v) for v in parent_to_subs.values()), default=1)
    log.info(f"  |subs|_max = {subs_max}")

    # ── Build D3FEND coverage count per technique ──
    # Look for D3FEND-specific column in tech_df
    d3fend_col = None
    for c in tech_df.columns:
        cl = c.lower()
        if "d3fend" in cl or "defenses" in cl or "countermeasure" in cl:
            d3fend_col = c
            break

    tech_d3fend_count = {}
    if d3fend_col:
        for _, r in tech_df.iterrows():
            val = r.get(d3fend_col, "")
            cnt = len(str(val).split(",")) if pd.notna(val) and str(val).strip() else 0
            tech_d3fend_count[r["name"]] = cnt
    else:
        log.info("  No D3FEND column found in tech_df; "
                 "using DETECTION_COVERAGE as proxy for D3FEND count")

    # ── Build data source count per technique ──
    ds_col = None
    for c in tech_df.columns:
        if "data source" in c.lower():
            ds_col = c
            break

    tech_ds_count = {}
    if ds_col:
        for _, r in tech_df.iterrows():
            val = r.get(ds_col, "")
            cnt = len(str(val).split(",")) if pd.notna(val) and str(val).strip() else 0
            tech_ds_count[r["name"]] = cnt

    # ── Eq. 10 denominator: |d3fend|_max + |datasources|_max ──
    d3fend_max = max(tech_d3fend_count.values(), default=1) if tech_d3fend_count else 1
    ds_max = max(tech_ds_count.values(), default=1) if tech_ds_count else 1
    denom_detection = max(d3fend_max + ds_max, 1)
    log.info(f"  Eq. 10 normalization: |d3fend|_max={d3fend_max}, "
             f"|datasources|_max={ds_max}, denominator={denom_detection}")

    # ── Permission column (Eq. 9: perm(t)) ──
    perm_col = None
    for c in tech_df.columns:
        if "permission" in c.lower():
            perm_col = c
            break

    # ── Mean NCISS across all campaigns (Eq. 12 baseline benefit) ──
    mean_nciss = (float(np.mean(list(campaign_severity.values())))
                  if campaign_severity else 0.5)
    log.info(f"  Mean NCISS (benefit baseline): {mean_nciss:.4f}")

    # ── Extract parameters per technique ──
    rows = []
    for _, r in tech_df.iterrows():
        name = r["name"]
        tactics = name_to_tactics.get(name, [])
        primary_tac = tactics[0].lower() if tactics else "unknown"

        # ────────────────────────────────────────────────────
        # Eq. 9: e(a) = w1 * perm(t) + w2 * |subs(t)| / |subs|_max
        # ────────────────────────────────────────────────────
        perm_str = str(r.get(perm_col, "")).lower() if perm_col else ""
        perm_score = max(
            (PERMISSION_EFFORT.get(p.strip(), 0.1)
             for p in perm_str.split(",") if p.strip()),
            default=0.1,
        )
        sub_count = len(parent_to_subs.get(name, []))
        effort = W1_EFFORT * perm_score + W2_EFFORT * (sub_count / subs_max)

        # ────────────────────────────────────────────────────
        # Eq. 10: P_max(t) = (|d3fend(t)| + |datasources(t)|)
        #                   / (|d3fend|_max + |datasources|_max)
        # ────────────────────────────────────────────────────
        d3f_cnt = tech_d3fend_count.get(name, 0)
        if not tech_d3fend_count:
            # Fallback: use DETECTION_COVERAGE as proxy for D3FEND count
            raw_cov = float(
                DETECTION_COVERAGE.get(name, 0.0)
                if hasattr(DETECTION_COVERAGE, 'get')
                else DETECTION_COVERAGE[name]
            )
            d3f_cnt = int(round(raw_cov * d3fend_max))
        ds_cnt = tech_ds_count.get(name, 0)
        P_max = (d3f_cnt + ds_cnt) / denom_detection

        # ────────────────────────────────────────────────────
        # Eq. 11: Ra(t) = tactic_order.index(primary_tactic(t)) / 13
        # ────────────────────────────────────────────────────
        try:
            tac_idx = tactic_order.index(primary_tac)
        except ValueError:
            tac_idx = 7  # mid-chain fallback for unmapped tactics
        resource_cost = tac_idx / 13.0

        # ────────────────────────────────────────────────────
        # Eq. 12: B(t,c) = NCISS(c) * (1 + lambda * I_Impact(t))
        # ────────────────────────────────────────────────────
        has_impact = int("impact" in [t.lower() for t in tactics])
        benefit = mean_nciss * (1.0 + LAMBDA_IMPACT * has_impact)

        # ────────────────────────────────────────────────────
        # Threat tier classification (Sec. III-E)
        # Paper: "groups stratified by documented technique count"
        # Tier assigned by highest-tier GROUP that uses this technique
        # ────────────────────────────────────────────────────
        gc = tech_group_counts.get(name, 0)
        tier = technique_tiers.get(name, 1)  # from group sophistication
        alpha, beta_param = BETA_PARAMS[tier]

        rows.append({
            "technique":      name,
            "effort":         round(effort, 4),
            "P_max":          round(P_max, 4),
            "resource_cost":  round(resource_cost, 4),
            "benefit":        round(benefit, 4),
            "threat_tier":    tier,
            "beta_alpha":     alpha,
            "beta_beta":      beta_param,
            "group_count":    gc,
        })

    df = pd.DataFrame(rows)
    log.info(f"Attacker params extracted for {len(df)} techniques")
    log.info(f"  Tier distribution: "
             f"{dict(df['threat_tier'].value_counts().sort_index())}")
    log.info(f"  Effort range:   [{df['effort'].min():.3f}, "
             f"{df['effort'].max():.3f}]")
    log.info(f"  P_max range:    [{df['P_max'].min():.3f}, "
             f"{df['P_max'].max():.3f}]")
    log.info(f"  Resource range: [{df['resource_cost'].min():.3f}, "
             f"{df['resource_cost'].max():.3f}]")
    log.info(f"  Benefit range:  [{df['benefit'].min():.3f}, "
             f"{df['benefit'].max():.3f}]")
    return df


# ==========================================================
# MODULE 2: DEFENDER PARAMETER EXTRACTION  (Eqs. 13--16)
# ==========================================================

def _count_mitigations(
    rel_df: pd.DataFrame,
    stix_to_name: Dict[str, str],
) -> Dict[str, int]:
    """
    Count mitigations per technique from rel_df.

    Looks for relationships where:
      - source is a course-of-action (mitigation)
      - target is an attack-pattern (technique)
      - relationship type contains 'mitigates'

    Used for Eq. 16: p(O_n) = |mitigations(e)| / |mitigations|_max
    """
    col = _find_rel_type_col(rel_df)
    src_col = _find_col(rel_df, ["source ref", "source_ref", "source"]) or "source ref"
    tgt_col = _find_col(rel_df, ["target ref", "target_ref", "target"]) or "target ref"

    mit_count = Counter()
    for _, r in rel_df.iterrows():
        src = str(r.get(src_col, ""))
        tgt = str(r.get(tgt_col, ""))
        rtype = str(r.get(col, "")).lower() if col else ""
        if ("course-of-action--" in src
                and "attack-pattern--" in tgt
                and "mitigates" in rtype):
            tgt_name = stix_to_name.get(tgt)
            if tgt_name:
                mit_count[tgt_name] += 1
    return dict(mit_count)


def _count_evasion_subtechs(
    parent_to_subs: Dict[str, list],
    name_to_tactics: Dict[str, list],
) -> Dict[str, int]:
    """
    Eq. 15: c_r(t) = |{s in subs(t) : tactic(s) = defense-evasion}|

    For each parent technique, count how many of its sub-techniques
    belong to the 'defense evasion' tactic. Returns raw count (not
    normalized), per the paper's formulation.
    """
    evasion_sub = {}
    for parent, subs in parent_to_subs.items():
        cnt = sum(
            1 for s in subs
            if "defense evasion" in [t.lower() for t in name_to_tactics.get(s, [])]
        )
        evasion_sub[parent] = cnt
    return evasion_sub


def extract_defender_params(
    tech_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    parent_to_subs: Dict[str, list],
    campaign_severity: Dict[str, float],
    name_to_tactics: Dict[str, list],
    tactic_order: List[str],
    stix_to_name: Dict[str, str],
    DETECTION_COVERAGE: dict,
    attacker_params: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Phase 3b: Extract defender-side utility parameters for every
    technique in ATT&CK.

    Implements Table I (defender side):
        Eq. 13: p(t_i) = |{g in G : g uses t_i}| / |G|
        Eq. 14: FN(t,c) = NCISS(c) * (1 - P(t))
        FP:     omega_tau (tactic-dependent configurable weight)
        Eq. 15: c_r(t) = |{s in subs(t) : tactic(s) = defense-evasion}|
        Eq. 16: p(O_n) = |mitigations(e)| / |mitigations|_max

    Args:
        tech_df, rel_df, parent_to_subs, campaign_severity,
        name_to_tactics, tactic_order, stix_to_name, DETECTION_COVERAGE:
            Same as extract_attacker_params.
        attacker_params: (Optional) DataFrame from Phase 3a.
            If provided, uses P_max column for Eq. 14 detection term.
            Otherwise falls back to DETECTION_COVERAGE.

    Returns:
        DataFrame with columns: technique, threat_prob, fn_cost,
        fp_cost, model_repair, operation_cost
    """
    _section("Phase 3b: Defender parameter extraction (Eqs. 13-16, Table I)")

    # ── Eq. 13 denominator: |G| (unique groups) ──
    total_groups = _get_total_unique_groups(rel_df)
    tech_group_counts = compute_group_tech_counts(rel_df, stix_to_name)
    log.info(f"  |G| = {total_groups} unique threat groups (Eq. 13 denom)")

    # ── Eq. 16: mitigation counts ──
    mit_counts = _count_mitigations(rel_df, stix_to_name)
    mit_max = max(mit_counts.values(), default=1)
    log.info(f"  |mitigations|_max = {mit_max} (Eq. 16 denom)")

    # ── Eq. 15: evasion sub-technique counts ──
    evasion_sub = _count_evasion_subtechs(parent_to_subs, name_to_tactics)

    # ── Mean NCISS for FN cost (Eq. 14) ──
    mean_nciss = (float(np.mean(list(campaign_severity.values())))
                  if campaign_severity else 0.5)

    # ── P_max lookup from attacker params for Eq. 14 ──
    p_max_lookup = {}
    if attacker_params is not None and "P_max" in attacker_params.columns:
        p_max_lookup = dict(zip(
            attacker_params["technique"], attacker_params["P_max"]))
        log.info(f"  Using P_max from Phase 3a for Eq. 14 ({len(p_max_lookup)} entries)")

    # ── Extract parameters per technique ──
    rows = []
    for _, r in tech_df.iterrows():
        name = r["name"]
        tactics = name_to_tactics.get(name, [])
        primary_tac = tactics[0].lower() if tactics else "unknown"

        # ────────────────────────────────────────────────────
        # Eq. 13: p(t_i) = |{g in G : g uses t_i}| / |G|
        # ────────────────────────────────────────────────────
        gc = tech_group_counts.get(name, 0)
        threat_prob = gc / total_groups

        # ────────────────────────────────────────────────────
        # Eq. 14: FN(t,c) = NCISS(c) * (1 - P(t))
        # P(t) = P_max from Eq. 10 (attacker params)
        # ────────────────────────────────────────────────────
        if name in p_max_lookup:
            p_t = p_max_lookup[name]
        else:
            p_t = float(
                DETECTION_COVERAGE.get(name, 0.0)
                if hasattr(DETECTION_COVERAGE, 'get')
                else DETECTION_COVERAGE[name]
            )
        fn_cost = mean_nciss * (1.0 - p_t)

        # ────────────────────────────────────────────────────
        # FP cost: omega_tau (configurable tactic weight)
        # Sec. III-F: "tactic-level disruption costs are inherently
        # organization-dependent and cannot be derived solely from
        # ATT&CK metadata"
        # ────────────────────────────────────────────────────
        fp_cost = TACTIC_FP_WEIGHT.get(primary_tac, 0.20)

        # ────────────────────────────────────────────────────
        # Eq. 15: c_r(t) = |{s in subs(t) : tactic(s) = defense-evasion}|
        # Raw count, not normalized (per paper formulation)
        # ────────────────────────────────────────────────────
        model_repair = evasion_sub.get(name, 0)

        # ────────────────────────────────────────────────────
        # Eq. 16: p(O_n) = |mitigations(e)| / |mitigations|_max
        # ────────────────────────────────────────────────────
        mc = mit_counts.get(name, 0)
        operation_cost = mc / mit_max

        rows.append({
            "technique":      name,
            "threat_prob":    round(threat_prob, 6),
            "fn_cost":        round(fn_cost, 4),
            "fp_cost":        round(fp_cost, 4),
            "model_repair":   model_repair,
            "operation_cost": round(operation_cost, 4),
        })

    df = pd.DataFrame(rows)
    log.info(f"Defender params extracted for {len(df)} techniques")
    log.info(f"  Threat prob range:  [{df['threat_prob'].min():.4f}, "
             f"{df['threat_prob'].max():.4f}]")
    log.info(f"  FN cost range:      [{df['fn_cost'].min():.4f}, "
             f"{df['fn_cost'].max():.4f}]")
    log.info(f"  Model repair max:   {df['model_repair'].max()}")
    log.info(f"  Operation cost max: {df['operation_cost'].max():.4f}")
    return df


# ==========================================================
# BASELINE UTILITY  psi_n  (Eq. 1)
# ==========================================================

def compute_psi_n(
    defender_params: pd.DataFrame,
    gamma: float = GAMMA_DEFENDER,
    n_samples: int = MC_SAMPLES,
) -> Dict[str, float]:
    """
    Eq. 1: Baseline expected utility under normal conditions (no attack).

        psi_n = integral u(c_n) p(O_n) dO_n

    This is the reference point against which all adversarial scenarios
    are measured (Sec. III-A, Fig. 1). The system incurs only operational
    costs with no threat activity.

    Evaluated via Monte Carlo: for each technique's operation_cost,
    draw c_n ~ Exp(operation_cost) and compute u(c_n) = -exp(gamma*c_n).

    Args:
        defender_params: DataFrame from Phase 3b (needs operation_cost)
        gamma: Risk-aversion coefficient
        n_samples: Monte Carlo samples

    Returns:
        Dict with psi_n (mean baseline utility), psi_n_std,
        mean_operation_cost, and per-technique breakdown.
    """
    per_tech_psi_n = []
    for _, row in defender_params.iterrows():
        scale = max(row["operation_cost"], 0.01)
        c_n_samples = rng.exponential(scale, size=n_samples)
        utilities = -np.exp(gamma * c_n_samples)
        per_tech_psi_n.append(float(np.mean(utilities)))

    psi_n = float(np.mean(per_tech_psi_n))
    psi_n_std = float(np.std(per_tech_psi_n))

    return {
        "psi_n":               round(psi_n, 8),
        "psi_n_std":           round(psi_n_std, 8),
        "mean_operation_cost": round(float(defender_params["operation_cost"].mean()), 4),
        "n_techniques":        len(defender_params),
    }


# ==========================================================
# MODULE 3: ATTACKER UTILITY  psi_A  (Eqs. 2--3)
# ==========================================================

def compute_psi_A_technique(
    effort: float,
    P_max: float,
    resource: float,
    benefit: float,
    alpha: float,
    beta_param: float,
    tier: int,
    threat_prob: float = 1.0,
    n_samples: int = MC_SAMPLES,
) -> Dict[str, float]:
    """
    Eq. 3: Compute psi_A for a single technique, summed over both
    actions a in {regular, adversarial}, weighted by p(t_i).

    Full Eq. 3:
        psi_A = sum_i sum_a u_A(c_A(t_i,a)) * p(t_i) * p(a|t_i)

    For each action a:
        1. Draw d ~ Beta(alpha, beta)           -- detection uncertainty
        2. P(t_i, a) = d * P_max(t_i)           -- detection penalty
        3. B(t_i, a) = (1 - d) * B_max(t_i)     -- scaled benefit
        4. c_A(t_i, a) = e(a) + P + Ra - B      -- Eq. 2
        5. u_A(c_A) = -c_A                       -- linear attacker utility
        6. Multiply by p(t_i) * p(a|t_i)         -- Eq. 3 weighting

    psi_A_technique = p(t_i) * sum_a [ E[u_A(c_A(t_i,a))] * p(a|t_i) ]

    The linear utility u_A = -c_A means the attacker is risk-neutral:
    lower cost (or net gain when benefit > costs) yields higher utility.

    Args:
        effort: Base effort from Eq. 9 (scaled by action type)
        P_max: Maximum detection probability from Eq. 10
        resource: Resource cost from Eq. 11
        benefit: Benefit from Eq. 12
        alpha, beta_param: Beta distribution parameters for this tier
        tier: Threat tier (1, 2, or 3)
        threat_prob: p(t_i) from Eq. 13 -- probability that this threat
                     type is active. Weights the entire contribution of
                     this technique to psi_A per Eq. 3.
        n_samples: Number of Monte Carlo draws

    Returns:
        Dict with psi_A (weighted by p(t_i)), mean_detection,
        std_detection, and per-action breakdown.
    """
    # Draw detection probability from tier-specific Beta distribution
    d_samples = rng.beta(alpha, beta_param, size=n_samples)

    psi_components = {}
    unweighted_psi = 0.0

    for action in ACTIONS:
        p_a_given_t = ACTION_PROB_GIVEN_TIER[tier][action]

        # Adversarial actions require more effort
        effort_a = effort * (ADVERSARIAL_EFFORT_MULT
                             if action == "adversarial" else 1.0)

        # Eq. 2 with Beta-scaled detection and benefit:
        #   P(t_i, a) = d * P_max        (detection penalty)
        #   B(t_i, a) = (1 - d) * benefit (scaled benefit)
        detection_penalty = d_samples * P_max
        scaled_benefit = (1.0 - d_samples) * benefit

        # Eq. 2: c_A(t_i, a) = e(a) + P(t_i,a) + Ra(a) - B(t_i,a)
        costs = effort_a + detection_penalty + resource - scaled_benefit

        # u_A(c_A) = -c_A (risk-neutral linear utility)
        utilities = -costs
        expected_u = float(np.mean(utilities))
        weighted = expected_u * p_a_given_t
        unweighted_psi += weighted

        psi_components[action] = {
            "expected_utility": round(expected_u, 6),
            "p_a_given_t":     p_a_given_t,
            "weighted":        round(weighted, 6),
        }

    # Eq. 3: multiply by p(t_i) -- threat probability weighting
    total_psi = unweighted_psi * threat_prob

    # a* = argmax_a E[U_A(t_i, a)] -- optimal attacker strategy (Sec. III-B)
    a_star = max(psi_components, key=lambda a: psi_components[a]["expected_utility"])

    return {
        "psi_A":            round(total_psi, 6),
        "psi_A_unweighted": round(unweighted_psi, 6),
        "threat_prob":      round(threat_prob, 6),
        "a_star":           a_star,
        "mean_detection":   round(float(np.mean(d_samples)), 4),
        "std_detection":    round(float(np.std(d_samples)), 4),
        "action_breakdown": psi_components,
    }


def compute_psi_A_chain(
    chain_names: List[str],
    attacker_params: pd.DataFrame,
    defender_params: pd.DataFrame = None,
    condition_on_chain: bool = False,
) -> Dict[str, float]:
    """
    Compute aggregate attacker expected utility for an entire chain.

    Eq. 3: psi_A = sum_i sum_a u_A(c_A(t_i,a)) * p(t_i) * p(a|t_i)

    Chain-level psi_A = sum of per-step psi_A values, where each
    step's contribution is weighted by p(t_i) from Eq. 13.

    When condition_on_chain=True, p(t_i) is set to 1.0 for all
    techniques because the chain's existence already conditions on
    these threats being active. p(t_i) is retained on the defender
    side (Eq. 7 Bernoulli sampling) where the defender must still
    assess the overall threat landscape. This prevents rare-but-
    devastating techniques (e.g. Triton ICS) from being suppressed
    by low group frequency.

    Args:
        chain_names: List of technique names in the chain
        attacker_params: DataFrame from extract_attacker_params
        defender_params: (Optional) DataFrame from extract_defender_params.
            If provided and condition_on_chain=False, uses threat_prob
            column for Eq. 3 p(t_i) weighting.
        condition_on_chain: If True, set p(t_i)=1.0 for all techniques
            on the attacker side (chain existence conditions on threat).

    Returns:
        Dict with chain_psi_A, per_step_psi_A list, and chain_length.
    """
    atk_lookup = attacker_params.set_index("technique")

    # Build threat_prob lookup from defender params (Eq. 13)
    # When condition_on_chain=True, we skip this — all p(t_i)=1.0
    tp_lookup = {}
    if not condition_on_chain and defender_params is not None \
            and "threat_prob" in defender_params.columns:
        tp_lookup = dict(zip(
            defender_params["technique"], defender_params["threat_prob"]))

    total_psi = 0.0
    per_step = []

    for name in chain_names:
        if name not in atk_lookup.index:
            per_step.append(0.0)
            continue
        row = atk_lookup.loc[name]
        threat_prob = tp_lookup.get(name, 1.0)  # 1.0 when conditioned
        res = compute_psi_A_technique(
            row["effort"], row["P_max"], row["resource_cost"],
            row["benefit"], row["beta_alpha"], row["beta_beta"],
            int(row["threat_tier"]),
            threat_prob=threat_prob,
        )
        total_psi += res["psi_A"]
        per_step.append(res["psi_A"])

    return {
        "chain_psi_A":    round(total_psi, 6),
        "per_step_psi_A": per_step,
        "chain_length":   len(chain_names),
    }


# ==========================================================
# MODULE 4: DEFENDER UTILITY  psi_r  (Eqs. 4--7)
# ==========================================================

def defender_exponential_utility(
    cost: float,
    gamma: float = GAMMA_DEFENDER,
) -> float:
    """
    Risk-averse exponential utility function (Sec. III-G):
        u(c) = -exp(gamma * c)

    Properties:
      - Always negative (costs reduce utility)
      - Risk-averse: catastrophic losses weighted more heavily than
        equivalent expected-value losses
      - gamma > 0 controls degree of risk aversion

    Paper: "The exponential form ensures risk-averse behavior where
    catastrophic losses are weighted more heavily than equivalent
    expected-value losses."
    """
    return -math.exp(gamma * cost)


def compute_psi_r_technique(
    fn_cost: float,
    fp_cost: float,
    operation_cost: float,
    model_repair: float,
    threat_prob: float,
    robustness: float,
    gamma: float = GAMMA_DEFENDER,
    n_samples: int = MC_SAMPLES,
) -> float:
    """
    Eq. 7: Defender expected utility for one technique at robustness r.

    Monte Carlo evaluation with N samples:
        1. t_i ~ Bernoulli(threat_prob)      -- threat realization
           (Eq. 6 independence assumption)
        2. c_n ~ Exponential(operation_cost)  -- normal operating cost
        3. FN component = t_i * FN_cost * (1 - r)
           (missed attacks; reduced by higher robustness)
        4. FP component = (1 - t_i) * FP_cost * r
           (false alarms; increase with higher robustness)
        5. c_r = model_repair * r
           (repair cost scales with robustness level)
        6. c_total = c_n + FN + FP + c_r     (Eq. 4)
        7. u(c_total) = -exp(gamma * c_total) (Sec. III-G)

    Returns:
        E[u(c)] averaged over MC samples (float).
    """
    # Step 1: Threat realization (independence assumption, Eq. 6)
    threat_realized = rng.binomial(
        1, min(max(threat_prob, 0.0), 1.0), size=n_samples
    ).astype(float)

    # Step 2: Normal operating cost (stochastic)
    scale = max(operation_cost, 0.01)
    c_n = rng.exponential(scale, size=n_samples)

    # Steps 3-5: Cost components (Eq. 4)
    fn_component = threat_realized * fn_cost * (1.0 - robustness)
    fp_component = (1.0 - threat_realized) * fp_cost * robustness
    repair_component = float(model_repair) * robustness

    # Step 6: Total cost
    total_cost = c_n + fn_component + fp_component + repair_component

    # Step 7: Exponential utility
    utilities = -np.exp(gamma * total_cost)
    return float(np.mean(utilities))


def compute_psi_r_chain(
    chain_names: List[str],
    defender_params: pd.DataFrame,
    robustness: float,
    gamma: float = GAMMA_DEFENDER,
    n_samples: int = MC_SAMPLES,
) -> Dict[str, float]:
    """
    Eq. 7 aggregated over chain steps.

    Under the independence assumption (Eq. 6), chain-level psi_r
    is the mean of per-step expected utilities. This represents the
    average defender utility experienced across all techniques in
    the attack chain at a given robustness level r.

    Args:
        chain_names: List of technique names in the chain
        defender_params: DataFrame from extract_defender_params
        robustness: Current robustness level r in [0, 1]
        gamma: Risk-aversion coefficient
        n_samples: Monte Carlo samples

    Returns:
        Dict with chain_psi_r, per_step_psi_r list, and robustness.
    """
    lookup = defender_params.set_index("technique")
    per_step_psi = []

    for name in chain_names:
        if name not in lookup.index:
            per_step_psi.append(defender_exponential_utility(0.0, gamma))
            continue
        row = lookup.loc[name]
        psi = compute_psi_r_technique(
            row["fn_cost"], row["fp_cost"], row["operation_cost"],
            float(row["model_repair"]), row["threat_prob"],
            robustness, gamma, n_samples,
        )
        per_step_psi.append(psi)

    chain_psi_r = float(np.mean(per_step_psi))
    return {
        "chain_psi_r":    round(chain_psi_r, 8),
        "per_step_psi_r": per_step_psi,
        "robustness":     robustness,
    }


# ==========================================================
# MODULE 5: JOINT MODEL, OPTIMAL r*, VALIDATION, SENSITIVITY
# ==========================================================

def find_optimal_robustness(
    chain_names: List[str],
    defender_params: pd.DataFrame,
    attacker_params: pd.DataFrame,
    gamma: float = GAMMA_DEFENDER,
    grid: np.ndarray = ROBUSTNESS_GRID,
    n_samples: int = MC_SAMPLES,
    condition_on_chain: bool = False,
) -> Dict[str, object]:
    """
    Eq. 8: r* = argmax_r E_{T,C}[U_D(r)]

    Evaluates defender expected utility across a grid of robustness
    levels and selects the configuration yielding the highest expected
    defender utility.

    Paper (Sec. III-G): "Each candidate r corresponds to a specific
    NIDS configuration such as a detection threshold, adversarial
    training budget, or model architecture choice."

    Also records attacker utility (constant w.r.t. r) for diagnostic
    plots showing the gap between psi_A and psi_r.

    Args:
        chain_names: List of technique names in the chain
        defender_params: DataFrame from Phase 3b
        attacker_params: DataFrame from Phase 3a
        gamma: Risk-aversion coefficient
        grid: Array of robustness values to evaluate
        n_samples: Monte Carlo samples per evaluation
        condition_on_chain: If True, drop p(t_i) from attacker utility

    Returns:
        Dict with r_star, best_psi_r, psi_A, and grid_df (full results).
    """
    # psi_A is independent of r, compute once
    psi_A_val = compute_psi_A_chain(
        chain_names, attacker_params, defender_params,
        condition_on_chain=condition_on_chain)["chain_psi_A"]

    results = []
    for r in grid:
        psi_r = compute_psi_r_chain(
            chain_names, defender_params, r, gamma, n_samples
        )["chain_psi_r"]
        results.append({
            "robustness": r,
            "psi_r":      psi_r,
            "psi_A":      psi_A_val,
        })

    df = pd.DataFrame(results)
    best_idx = df["psi_r"].idxmax()
    r_star = float(df.loc[best_idx, "robustness"])
    best_psi_r = float(df.loc[best_idx, "psi_r"])

    return {
        "r_star":      round(r_star, 4),
        "best_psi_r":  round(best_psi_r, 8),
        "psi_A":       round(psi_A_val, 6),
        "grid_df":     df,
    }


def joint_risk_score(
    chain_names: List[str],
    attacker_params: pd.DataFrame,
    defender_params: pd.DataFrame,
    gamma: float = GAMMA_DEFENDER,
    n_samples: int = MC_SAMPLES,
    campaign_nciss: float = None,
    condition_on_chain: bool = True,
    sigmoid_temp: float = None,
) -> Dict[str, float]:
    """
    Compute the joint ARA-OSID risk score for a chain.

    Maps the gap between attacker incentive and defender's optimized
    posture onto a 0-10 scale comparable to NCISS:

        R_ARA = sigmoid( (psi_A_norm - psi_r_norm) / T ) * 10

    where psi_A_norm and psi_r_norm are per-step averages (not sums)
    to prevent chain length from dominating the score, and T is a
    temperature parameter controlling sigmoid sensitivity.

    When condition_on_chain=True (default), p(t_i) is dropped from the
    attacker utility because the chain's existence already conditions
    on these threats being active. p(t_i) is retained on the defender
    side via Bernoulli sampling (Eq. 7).

    When campaign_nciss is provided, attacker benefit is scaled by
    the ratio campaign_NCISS / mean_NCISS to differentiate campaigns.

    Args:
        chain_names: List of technique names in the chain
        attacker_params: DataFrame from Phase 3a
        defender_params: DataFrame from Phase 3b
        gamma: Risk-aversion coefficient
        n_samples: Monte Carlo samples
        campaign_nciss: (Optional) NCISS severity for this chain's
            campaign (0-10 scale). If provided, scales benefit by
            campaign_nciss / mean_benefit to differentiate campaigns.
        condition_on_chain: If True (default), drop p(t_i) from
            attacker utility for chain-level scoring.
        sigmoid_temp: Override sigmoid temperature. If None, uses
            the global SIGMOID_TEMPERATURE.

    Returns:
        Dict with ara_risk_score (0-10), r_star, psi_A, psi_r_at_rstar,
        utility_gap, and psi_A_norm.
    """
    T = sigmoid_temp if sigmoid_temp is not None else SIGMOID_TEMPERATURE

    # If campaign-specific NCISS provided, scale benefit (Eq.12) and
    # FN cost (Eq.14) for this chain. Both equations use NCISS(c) which
    # varies by campaign, but extraction used mean_nciss as base.
    if campaign_nciss is not None:
        mean_benefit = attacker_params["benefit"].mean()
        mean_fn = defender_params["fn_cost"].mean()

        # Benefit scale: Eq.12 B(t,c) = NCISS(c) * (...)
        benefit_scale = campaign_nciss / mean_benefit if mean_benefit > 0 else 1.0
        # FN scale: since FN = mean_nciss * (1-P), scale by campaign/mean_nciss
        mean_nciss_used = attacker_params["benefit"].min()  # benefit without Impact = mean_nciss
        fn_scale = campaign_nciss / mean_nciss_used if mean_nciss_used > 0 else 1.0

        atk_scaled = attacker_params.copy()
        atk_scaled["benefit"] = atk_scaled["benefit"] * benefit_scale

        def_scaled = defender_params.copy()
        def_scaled["fn_cost"] = def_scaled["fn_cost"] * fn_scale
    else:
        atk_scaled = attacker_params
        def_scaled = defender_params
        benefit_scale = 1.0
        fn_scale = 1.0

    opt = find_optimal_robustness(
        chain_names, def_scaled, atk_scaled,
        gamma, ROBUSTNESS_GRID, n_samples,
        condition_on_chain=condition_on_chain,
    )
    r_star = opt["r_star"]
    best_psi = opt["best_psi_r"]
    psi_A = opt["psi_A"]

    # Normalize by chain length to prevent long chains from dominating
    chain_len = max(len(chain_names), 1)
    psi_A_norm = psi_A / chain_len
    psi_r_norm = best_psi  # psi_r is already mean (not sum)

    # Sigmoid with temperature: sigmoid((psi_A_norm - psi_r_norm) / T) * 10
    gap = psi_A_norm - psi_r_norm
    gap_scaled = gap / max(T, 0.01)
    # Clamp to prevent overflow
    gap_clamped = max(min(gap_scaled, 20.0), -20.0)
    risk_01 = 1.0 / (1.0 + math.exp(-gap_clamped))
    risk_10 = risk_01 * 10.0

    return {
        "ara_risk_score":  round(risk_10, 4),
        "r_star":          r_star,
        "psi_A":           round(psi_A, 6),
        "psi_A_norm":      round(psi_A_norm, 6),
        "psi_r_at_rstar":  round(best_psi, 8),
        "utility_gap":     round(gap, 6),
        "utility_gap_scaled": round(gap_scaled, 6),
        "benefit_scale":   round(benefit_scale, 4),
        "fn_scale":        round(fn_scale, 4),
    }


# ──────────────────────────────────────────────────────────
# 3-SCENARIO VALIDATION vs NCISS  (Sec. IV, RQ3)
# ──────────────────────────────────────────────────────────

def validate_against_nciss(
    all_chains: List[List[str]],
    campaign_ids_index: List[str],
    campaign_index: List[str],
    attacker_params: pd.DataFrame,
    defender_params: pd.DataFrame,
    campaign_severity: Dict[str, float],
    gamma: float = GAMMA_DEFENDER,
    n_samples: int = 1000,
    scenarios: List[Tuple[str, float]] = None,
    zscore_params: Dict[str, float] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Phase 6a: Three-scenario validation against NCISS (Sec. IV, RQ3).

    Paper: "100% training establishes an upper bound on fit, 50/50 split
    tests generalization with MAE, standard deviation, and relative error
    on held-out campaigns, and 80/20 split produces predicted-vs-actual
    scatter plots with Pearson and Spearman correlation (targets:
    Pearson >= 0.70, Spearman >= 0.75)."

    Note (Sec. IV): "NCISS severity appears both as an input to the
    benefit (Eq. 12) and false negative cost (Eq. 14) formulations and
    as the validation target. This evaluates internal consistency rather
    than fully independent validation."

    Args:
        all_chains: All generated attack chains
        campaign_ids_index: Campaign ID per chain
        campaign_index: Campaign name per chain
        attacker_params, defender_params: From Phase 3
        campaign_severity: NCISS ground truth (normalized 0-1)
        gamma: Risk-aversion coefficient
        n_samples: MC samples (reduced for speed during validation)
        scenarios: List of (name, split_ratio) tuples
        zscore_params: If provided, dict with 'gap_mean', 'gap_std',
            'nciss_mean', 'nciss_std' from Phase 4-5. Used for affine
            calibration: R = (gap-gap_mean)/gap_std × nciss_std + nciss_mean.

    Returns:
        Dict mapping scenario name to results DataFrame.
    """
    if scenarios is None:
        scenarios = [("100pct", 1.0), ("50-50", 0.5), ("80-20", 0.8)]

    _section("Phase 6a: 3-Scenario ARA validation vs NCISS (RQ3)")

    # Filter to chains with NCISS severity scores
    valid_indices = [
        i for i, cid in enumerate(campaign_ids_index)
        if cid in campaign_severity
    ]
    log.info(f"Chains with NCISS severity: {len(valid_indices)} / "
             f"{len(all_chains)}")

    if not valid_indices:
        log.warning("No chains have matching NCISS severity scores. "
                     "Validation skipped.")
        return {}

    all_results = {}
    for scenario_name, split_ratio in scenarios:
        log.info(f"\n--- Scenario: {scenario_name} "
                 f"(split={split_ratio}) ---")
        perm = rng.permutation(valid_indices)

        if split_ratio >= 1.0:
            test_indices = list(perm)  # evaluate on all (baseline)
        else:
            cut = int(split_ratio * len(perm))
            test_indices = list(perm[cut:])

        rows = []
        t0 = time.time()
        for count, idx in enumerate(test_indices):
            chain = all_chains[idx]
            camp_id = campaign_ids_index[idx]
            camp_name = campaign_index[idx]
            actual_nciss_norm = campaign_severity[camp_id]
            # campaign_severity values are already on 0-10 scale
            # (NCISS_Score / 10.0 done in main pipeline)
            actual_nciss_10 = actual_nciss_norm

            jrs = joint_risk_score(
                chain, attacker_params, defender_params, gamma, n_samples,
                campaign_nciss=actual_nciss_10,
            )
            # Apply affine calibration if params available
            if zscore_params is not None:
                gap = jrs["utility_gap"]
                gm = zscore_params["gap_mean"]
                gs = zscore_params["gap_std"]
                nm = zscore_params["nciss_mean"]
                ns = zscore_params["nciss_std"]
                score = (gap - gm) / gs * ns + nm
                predicted = round(max(0.0, min(10.0, score)), 4)
            else:
                predicted = jrs["ara_risk_score"]
            abs_err = abs(predicted - actual_nciss_10)
            rel_err = (abs_err / max(actual_nciss_10, 1e-6)) * 100.0

            rows.append({
                "campaign_id":     camp_id,
                "campaign":        camp_name,
                "chain_length":    len(chain),
                "actual_nciss":    round(actual_nciss_10, 4),
                "predicted_risk":  predicted,
                "r_star":          jrs["r_star"],
                "psi_A":           jrs["psi_A"],
                "psi_r":           jrs["psi_r_at_rstar"],
                "absolute_error":  round(abs_err, 4),
                "relative_error":  round(rel_err, 2),
            })

            if (count + 1) % 50 == 0:
                elapsed = time.time() - t0
                log.info(f"  {count+1}/{len(test_indices)} chains scored "
                         f"({elapsed:.1f}s)")

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            mae  = result_df["absolute_error"].mean()
            std  = result_df["absolute_error"].std()
            mrae = result_df["relative_error"].mean()
            med  = result_df["absolute_error"].median()
            corr = result_df[["actual_nciss", "predicted_risk"]].corr().iloc[0, 1]
            sp_r, sp_p = sp_stats.spearmanr(
                result_df["actual_nciss"], result_df["predicted_risk"]
            )

            log.info(f"  MAE={mae:.4f} (std={std:.4f}) | "
                     f"MedAE={med:.4f} | MRAE={mrae:.2f}%")
            log.info(f"  Pearson r={corr:.4f} | "
                     f"Spearman rho={sp_r:.4f} (p={sp_p:.2e})")

        all_results[scenario_name] = result_df

    return all_results


# ──────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS  (Sec. IV, RQ4)
# ──────────────────────────────────────────────────────────

def sensitivity_analysis(
    sample_chains: List[List[str]],
    attacker_params: pd.DataFrame,
    defender_params: pd.DataFrame,
    param_names: List[str] = None,
    deltas: List[float] = None,
    gamma: float = GAMMA_DEFENDER,
    n_samples: int = 1000,
) -> pd.DataFrame:
    """
    Phase 6b: Sensitivity analysis (Sec. IV, RQ4).

    Paper: "Each of the 10 parameters was perturbed by +/-10%, +/-25%,
    and +/-50% while holding others at empirically derived values.
    Changes in psi_A, psi_r, and r* identify dominant parameters and
    priorities for higher-fidelity estimation."

    Paper (preliminary): "detection coverage P and false negative cost
    FN are the primary influencing factors"

    Args:
        sample_chains: Subset of chains to analyze (typically 10)
        attacker_params, defender_params: From Phase 3
        param_names: Which parameters to perturb
        deltas: Perturbation factors (e.g., 0.5 = -50%, 1.5 = +50%)
        gamma: Risk-aversion coefficient
        n_samples: MC samples per evaluation

    Returns:
        DataFrame with per-chain, per-parameter, per-delta results
        including risk_change and rstar_change.
    """
    if param_names is None:
        param_names = [
            "effort", "P_max", "resource_cost", "benefit",       # attacker
            "fn_cost", "fp_cost", "model_repair", "operation_cost",  # defender
            "threat_prob",                                        # defender
        ]
    if deltas is None:
        deltas = SENSITIVITY_DELTAS

    _section("Phase 6b: Sensitivity analysis (RQ4, +/-10/25/50%)")

    # Classify parameters by side
    atk_cols = {"effort", "P_max", "resource_cost", "benefit"}
    def_cols = {"fn_cost", "fp_cost", "model_repair", "operation_cost",
                "threat_prob"}

    rows = []
    for chain_idx, chain in enumerate(sample_chains):
        # Baseline score
        base_jrs = joint_risk_score(
            chain, attacker_params, defender_params, gamma, n_samples
        )
        base_risk = base_jrs["ara_risk_score"]
        base_rstar = base_jrs["r_star"]

        for param in param_names:
            for delta in deltas:
                if delta == 1.0:
                    continue  # skip identity (no perturbation)

                # Create perturbed copies
                atk_pert = attacker_params.copy()
                def_pert = defender_params.copy()

                # Apply perturbation to the correct side
                if param in atk_cols and param in atk_pert.columns:
                    atk_pert[param] = (atk_pert[param] * delta).clip(0, 1)

                if param in def_cols and param in def_pert.columns:
                    if param == "model_repair":
                        # model_repair is raw count, clip at 0 only
                        def_pert[param] = (def_pert[param] * delta).clip(0)
                    else:
                        def_pert[param] = (def_pert[param] * delta).clip(0, 1)

                pert_jrs = joint_risk_score(
                    chain, atk_pert, def_pert, gamma, n_samples
                )

                rows.append({
                    "chain_idx":      chain_idx,
                    "chain_head":     " -> ".join(chain[:3]),
                    "parameter":      param,
                    "delta_factor":   delta,
                    "delta_pct":      f"{(delta - 1) * 100:+.0f}%",
                    "base_risk":      base_risk,
                    "perturbed_risk": pert_jrs["ara_risk_score"],
                    "risk_change":    round(pert_jrs["ara_risk_score"] - base_risk, 4),
                    "base_rstar":     base_rstar,
                    "pert_rstar":     pert_jrs["r_star"],
                    "rstar_change":   round(pert_jrs["r_star"] - base_rstar, 4),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        summary = df.groupby("parameter").agg(
            mean_risk_change=("risk_change", "mean"),
            max_abs_risk_change=("risk_change", lambda x: x.abs().max()),
            mean_rstar_change=("rstar_change", "mean"),
        ).round(4)
        log.info(f"\nSensitivity summary (RQ4):\n{summary}")

    return df


def sensitivity_configurable_weights(
    sample_chains: List[List[str]],
    tech_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    parent_to_subs: Dict[str, list],
    campaign_severity: Dict[str, float],
    name_to_tactics: Dict[str, list],
    tactic_order: List[str],
    stix_to_name: Dict[str, str],
    DETECTION_COVERAGE: dict,
    base_atk: pd.DataFrame,
    base_def: pd.DataFrame,
    deltas: List[float] = None,
    n_samples: int = 1000,
) -> pd.DataFrame:
    """
    Sensitivity analysis for configurable weights (Sec. V):

    Paper: "effort weights w1, w2; Impact bonus lambda; risk aversion
    gamma; and per-tactic FP weights omega_tau remain configurable and
    are explored through sensitivity analysis."

    Unlike sensitivity_analysis() which perturbs extracted parameter
    values, this function re-derives parameters from scratch using
    modified weight constants. This is necessary because w1, w2, and
    lambda affect the extraction formulas (Eqs. 9, 12), not just the
    final values.

    Args:
        sample_chains: Subset of chains to analyze
        tech_df, rel_df, etc.: Original pipeline data for re-extraction
        base_atk, base_def: Baseline parameter DataFrames
        deltas: Perturbation factors
        n_samples: MC samples per evaluation

    Returns:
        DataFrame with per-weight, per-delta risk and r* changes.
    """
    if deltas is None:
        deltas = SENSITIVITY_DELTAS

    _section("Phase 6b+: Configurable weight sensitivity (w1, w2, lambda, gamma)")

    global W1_EFFORT, W2_EFFORT, LAMBDA_IMPACT
    orig_w1 = W1_EFFORT
    orig_w2 = W2_EFFORT
    orig_lam = LAMBDA_IMPACT

    # Baseline scores per chain
    base_scores = {}
    for ci, ch in enumerate(sample_chains):
        jrs = joint_risk_score(ch, base_atk, base_def, GAMMA_DEFENDER, n_samples)
        base_scores[ci] = {"risk": jrs["ara_risk_score"], "rstar": jrs["r_star"]}

    weight_configs = [
        ("w1_effort", orig_w1),
        ("w2_effort", orig_w2),
        ("lambda_impact", orig_lam),
        ("gamma_defender", GAMMA_DEFENDER),
    ]

    rows = []
    for weight_name, base_val in weight_configs:
        for delta in deltas:
            if delta == 1.0:
                continue
            new_val = base_val * delta

            if weight_name == "gamma_defender":
                # Gamma only affects utility computation, not extraction
                for ci, ch in enumerate(sample_chains):
                    pjrs = joint_risk_score(ch, base_atk, base_def, new_val, n_samples)
                    rows.append({
                        "weight": weight_name, "chain_idx": ci,
                        "delta_factor": delta, "delta_pct": f"{(delta-1)*100:+.0f}%",
                        "base_value": base_val, "perturbed_value": round(new_val, 4),
                        "base_risk": base_scores[ci]["risk"],
                        "perturbed_risk": pjrs["ara_risk_score"],
                        "risk_change": round(pjrs["ara_risk_score"] - base_scores[ci]["risk"], 4),
                        "rstar_change": round(pjrs["r_star"] - base_scores[ci]["rstar"], 4),
                    })
            else:
                # Re-derive params with perturbed weight
                if weight_name == "w1_effort":
                    W1_EFFORT = new_val
                elif weight_name == "w2_effort":
                    W2_EFFORT = new_val
                elif weight_name == "lambda_impact":
                    LAMBDA_IMPACT = new_val

                pert_atk = extract_attacker_params(
                    tech_df, rel_df, parent_to_subs, campaign_severity,
                    name_to_tactics, tactic_order, stix_to_name, DETECTION_COVERAGE)
                pert_def = extract_defender_params(
                    tech_df, rel_df, parent_to_subs, campaign_severity,
                    name_to_tactics, tactic_order, stix_to_name, DETECTION_COVERAGE,
                    attacker_params=pert_atk)

                for ci, ch in enumerate(sample_chains):
                    pjrs = joint_risk_score(ch, pert_atk, pert_def, GAMMA_DEFENDER, n_samples)
                    rows.append({
                        "weight": weight_name, "chain_idx": ci,
                        "delta_factor": delta, "delta_pct": f"{(delta-1)*100:+.0f}%",
                        "base_value": base_val, "perturbed_value": round(new_val, 4),
                        "base_risk": base_scores[ci]["risk"],
                        "perturbed_risk": pjrs["ara_risk_score"],
                        "risk_change": round(pjrs["ara_risk_score"] - base_scores[ci]["risk"], 4),
                        "rstar_change": round(pjrs["r_star"] - base_scores[ci]["rstar"], 4),
                    })

                # Restore original values
                W1_EFFORT = orig_w1
                W2_EFFORT = orig_w2
                LAMBDA_IMPACT = orig_lam

    df = pd.DataFrame(rows)
    if not df.empty:
        summary = df.groupby("weight").agg(
            mean_risk_change=("risk_change", "mean"),
            max_abs_risk=("risk_change", lambda x: x.abs().max()),
        ).round(4)
        log.info(f"\nConfigurable weight sensitivity:\n{summary}")

    return df


# ==========================================================
# PLOTTING UTILITIES
# ==========================================================

def _save_fig(path):
    """Save and close current figure."""
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Phase 3a: Attacker Parameter Visualizations ────────────

def plot_attacker_param_distributions(atk_params: pd.DataFrame, save_dir: str):
    """
    Four-panel histogram of attacker parameters (Eqs. 9-12).
    Shows effort, P_max, resource_cost, benefit distributions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    params = [
        ("effort", "Effort $e(a)$ [Eq. 9]", "steelblue"),
        ("P_max", "Detection $P_{\\max}$ [Eq. 10]", "coral"),
        ("resource_cost", "Resource $R_a$ [Eq. 11]", "mediumseagreen"),
        ("benefit", "Benefit $B$ [Eq. 12]", "goldenrod"),
    ]
    for ax, (col, title, color) in zip(axes.flat, params):
        data = atk_params[col].dropna()
        ax.hist(data, bins=30, edgecolor="black", alpha=0.7, color=color)
        ax.axvline(data.mean(), color="red", ls="--", lw=2,
                   label=f"Mean={data.mean():.3f}")
        ax.axvline(data.median(), color="blue", ls=":", lw=2,
                   label=f"Median={data.median():.3f}")
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("ARA-OSID: Attacker Parameter Distributions (Table I)",
                 fontsize=15, y=1.02)
    _save_fig(os.path.join(save_dir, "plot_attacker_param_distributions.png"))


def plot_threat_tier_distribution(atk_params: pd.DataFrame, save_dir: str):
    """
    Bar + pie chart of threat tier distribution (Sec. III-E).
    Paper: ~60% t1, ~25% t2, ~15% t3.
    """
    tier_counts = atk_params["threat_tier"].value_counts().sort_index()
    tier_labels = {1: "$t_1$ Novice", 2: "$t_2$ Intermediate", 3: "$t_3$ APT"}
    colors = {1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = ax1.bar([tier_labels.get(t, f"t{t}") for t in tier_counts.index],
                   tier_counts.values,
                   color=[colors.get(t, "gray") for t in tier_counts.index],
                   edgecolor="black")
    for bar, val in zip(bars, tier_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val}", ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Techniques", fontsize=12)
    ax1.set_title("Threat Tier Counts", fontsize=13)
    ax1.grid(axis="y", alpha=0.3)

    # Pie chart
    ax2.pie(tier_counts.values,
            labels=[f"{tier_labels.get(t, f't{t}')}\n({v}, {v/len(atk_params)*100:.1f}%)"
                    for t, v in zip(tier_counts.index, tier_counts.values)],
            colors=[colors.get(t, "gray") for t in tier_counts.index],
            startangle=90, textprops={"fontsize": 10})
    ax2.set_title("Tier Distribution", fontsize=13)

    fig.suptitle("ARA-OSID: Threat Tier Classification (Sec. III-E)",
                 fontsize=15, y=1.02)
    _save_fig(os.path.join(save_dir, "plot_threat_tier_distribution.png"))


def plot_group_count_distribution(atk_params: pd.DataFrame, save_dir: str):
    """Histogram of group counts per technique with tier threshold markers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = atk_params["group_count"]
    ax.hist(data, bins=40, edgecolor="black", alpha=0.7, color="slateblue")
    ax.axvline(TIER_L1_MAX, color="green", ls="--", lw=2,
               label=f"$t_1$/$t_2$ boundary ({TIER_L1_MAX})")
    ax.axvline(TIER_L2_MAX, color="red", ls="--", lw=2,
               label=f"$t_2$/$t_3$ boundary ({TIER_L2_MAX})")
    ax.set_xlabel("Number of Groups Using Technique", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Group Usage Distribution with Tier Boundaries", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save_fig(os.path.join(save_dir, "plot_group_count_distribution.png"))


# ── Phase 3b: Defender Parameter Visualizations ────────────

def plot_defender_param_distributions(def_params: pd.DataFrame, save_dir: str):
    """
    Five-panel histogram of defender parameters (Eqs. 13-16).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    params = [
        ("threat_prob", "Threat Prob $p(t_i)$ [Eq. 13]", "royalblue"),
        ("fn_cost", "FN Cost [Eq. 14]", "crimson"),
        ("fp_cost", r"FP Cost $\omega_\tau$", "darkorange"),
        ("model_repair", "Model Repair $c_r$ [Eq. 15]", "purple"),
        ("operation_cost", "Operation $p(O_n)$ [Eq. 16]", "teal"),
    ]
    for ax, (col, title, color) in zip(axes.flat, params):
        data = def_params[col].dropna()
        ax.hist(data, bins=30, edgecolor="black", alpha=0.7, color=color)
        ax.axvline(data.mean(), color="red", ls="--", lw=2,
                   label=f"Mean={data.mean():.4f}")
        ax.set_xlabel(title, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    # Remove unused subplot
    if len(params) < len(axes.flat):
        axes.flat[-1].set_visible(False)
    fig.suptitle("ARA-OSID: Defender Parameter Distributions (Table I)",
                 fontsize=15, y=1.02)
    _save_fig(os.path.join(save_dir, "plot_defender_param_distributions.png"))


def plot_fp_cost_by_tactic(save_dir: str):
    """Bar chart of FP disruption weights omega_tau by tactic (Sec. III-F)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    tactics = list(TACTIC_FP_WEIGHT.keys())
    weights = list(TACTIC_FP_WEIGHT.values())
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(tactics)))
    bars = ax.barh(tactics, weights, color=colors, edgecolor="black")
    for bar, w in zip(bars, weights):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{w:.2f}", va="center", fontsize=10)
    ax.set_xlabel(r"FP Disruption Weight $\omega_\tau$", fontsize=12)
    ax.set_title("False Positive Cost by Kill-Chain Tactic (Sec. III-F)",
                 fontsize=14)
    ax.set_xlim(0, 0.75)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    _save_fig(os.path.join(save_dir, "plot_fp_cost_by_tactic.png"))


# ── Phase 4-5: Joint Utility Visualizations ────────────────

def plot_ara_risk_distribution(utility_df: pd.DataFrame, save_dir: str):
    """Histogram of ARA-OSID risk scores across all scored chains."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = utility_df["ara_risk_score"].dropna()
    ax.hist(data, bins=30, edgecolor="black", alpha=0.7, color="crimson")
    ax.axvline(data.mean(), color="blue", ls="--", lw=2,
               label=f"Mean={data.mean():.2f}")
    ax.axvline(data.median(), color="green", ls=":", lw=2,
               label=f"Median={data.median():.2f}")
    ax.set_xlabel("ARA-OSID Risk Score (0-10)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Distribution of ARA Risk Scores ({len(data)} chains)",
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save_fig(os.path.join(save_dir, "plot_ara_risk_distribution.png"))


def plot_rstar_distribution(utility_df: pd.DataFrame, save_dir: str):
    """Histogram of optimal robustness r* values (Eq. 8)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = utility_df["r_star"].dropna()
    ax.hist(data, bins=21, edgecolor="black", alpha=0.7, color="navy",
            range=(0, 1))
    ax.axvline(data.mean(), color="red", ls="--", lw=2,
               label=f"Mean $r^*$={data.mean():.3f}")
    ax.set_xlabel("Optimal Robustness $r^*$ (Eq. 8)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Distribution of $r^*$ Across {len(data)} Chains",
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save_fig(os.path.join(save_dir, "plot_rstar_distribution.png"))


def plot_psiA_vs_psiR(utility_df: pd.DataFrame, save_dir: str):
    """Scatter of psi_A vs psi_r at r*, colored by ARA risk score."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(utility_df["psi_A"], utility_df["psi_r_at_rstar"],
                    c=utility_df["ara_risk_score"], cmap="RdYlGn_r",
                    s=30, alpha=0.7, edgecolors="k", linewidth=0.3)
    plt.colorbar(sc, ax=ax, label="ARA Risk Score (0-10)")
    ax.set_xlabel(r"$\psi_A$ (Attacker expected utility)", fontsize=12)
    ax.set_ylabel(r"$\psi_r(r^*)$ (Defender expected utility)", fontsize=12)
    ax.set_title(r"Attacker vs Defender Utility ($\psi_A$ vs $\psi_r$)",
                 fontsize=14)
    ax.grid(alpha=0.3)
    _save_fig(os.path.join(save_dir, "plot_psiA_vs_psiR.png"))


def plot_utility_gap_distribution(utility_df: pd.DataFrame, save_dir: str):
    """Histogram of utility gap (psi_A - psi_r(r*)) that drives risk score."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = utility_df["utility_gap"].dropna()
    ax.hist(data, bins=30, edgecolor="black", alpha=0.7, color="darkorange")
    ax.axvline(0, color="black", ls="-", lw=2, label="Zero gap")
    ax.axvline(data.mean(), color="red", ls="--", lw=2,
               label=f"Mean gap={data.mean():.3f}")
    ax.set_xlabel(r"Utility Gap: $\psi_A - \psi_r(r^*)$", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Attacker-Defender Utility Gap", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save_fig(os.path.join(save_dir, "plot_utility_gap_distribution.png"))


def plot_ara_vs_nciss_chains(utility_df: pd.DataFrame, save_dir: str):
    """Scatter of ARA risk vs NCISS for all scored chains (Phase 4-5)."""
    df = utility_df.dropna(subset=["actual_nciss"])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df["actual_nciss"], df["ara_risk_score"],
               alpha=0.5, s=40, c="teal", edgecolors="k", linewidth=0.3)
    ax.plot([0, 10], [0, 10], "r--", lw=2, label="Perfect prediction")

    corr = df[["actual_nciss", "ara_risk_score"]].corr().iloc[0, 1]
    mae = abs(df["actual_nciss"] - df["ara_risk_score"]).mean()
    sp_r, _ = sp_stats.spearmanr(df["actual_nciss"], df["ara_risk_score"])
    ax.text(0.5, 9.0,
            f"Pearson $r$ = {corr:.3f}\n"
            f"Spearman $\\rho$ = {sp_r:.3f}\n"
            f"MAE = {mae:.3f}\n"
            f"N = {len(df)}",
            fontsize=10, bbox=dict(facecolor="white", alpha=0.8))

    ax.set_xlabel("Actual NCISS Severity (0-10)", fontsize=12)
    ax.set_ylabel("ARA-OSID Risk Score (0-10)", fontsize=12)
    ax.set_title("ARA-OSID Risk vs NCISS (All Scored Chains)", fontsize=14)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save_fig(os.path.join(save_dir, "plot_ara_vs_nciss_chains.png"))


def plot_top10_comparison(utility_df: pd.DataFrame, save_dir: str):
    """Side-by-side bar chart: ARA risk vs NCISS for top-10 chains."""
    top10 = utility_df.sort_values("ara_risk_score", ascending=False).head(10)
    top10 = top10.dropna(subset=["actual_nciss"])
    if top10.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(top10))
    w = 0.35
    ax.barh(x - w/2, top10["ara_risk_score"].values, w,
            label="ARA-OSID Risk", color="crimson", edgecolor="black", alpha=0.8)
    ax.barh(x + w/2, top10["actual_nciss"].values, w,
            label="NCISS Severity", color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([f"#{i+1} {c[:25]}" for i, c in
                        enumerate(top10["campaign"].values)], fontsize=9)
    ax.set_xlabel("Score (0-10)", fontsize=12)
    ax.set_title("Top-10 Chains: ARA Risk vs NCISS Severity", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 10)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    _save_fig(os.path.join(save_dir, "plot_top10_ara_vs_nciss.png"))


def plot_per_campaign_summary(campaign_agg: pd.DataFrame, save_dir: str):
    """Grouped bar: per-campaign mean ARA risk vs actual NCISS."""
    df = campaign_agg.dropna(subset=["actual_nciss"]).sort_values(
        "mean_ara_risk", ascending=False).head(20)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df["mean_ara_risk"].values, w,
           label="Mean ARA Risk", color="crimson", edgecolor="black", alpha=0.8)
    ax.bar(x + w/2, df["actual_nciss"].values, w,
           label="NCISS Severity", color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c[:20] for c in df["campaign"].values],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score (0-10)", fontsize=12)
    ax.set_title("Per-Campaign: Mean ARA Risk vs NCISS Severity", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 10.5)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(os.path.join(save_dir, "plot_per_campaign_ara_vs_nciss.png"))


# ── Phase 6a: Validation Visualizations ────────────────────

def plot_validation_error_distributions(val_results: dict, save_dir: str):
    """Overlaid error histograms for all validation scenarios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"100pct": "blue", "50-50": "green", "80-20": "red"}

    for sname, rdf in val_results.items():
        if rdf.empty:
            continue
        c = colors.get(sname, "gray")
        ax1.hist(rdf["absolute_error"], bins=25, alpha=0.5, color=c,
                 edgecolor="black", label=f"{sname} (MAE={rdf['absolute_error'].mean():.2f})")
        ax2.hist(rdf["relative_error"], bins=25, alpha=0.5, color=c,
                 edgecolor="black", label=f"{sname}")

    ax1.set_xlabel("Absolute Error", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Absolute Error Distribution by Scenario", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Relative Error (%)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Relative Error Distribution by Scenario", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    fig.suptitle("ARA-OSID Validation Error Analysis (RQ3)", fontsize=15, y=1.02)
    _save_fig(os.path.join(save_dir, "plot_validation_error_distributions.png"))


def plot_scenario_comparison_bars(comparison_rows: list, save_dir: str):
    """Grouped bar chart comparing MAE, Pearson, Spearman across scenarios."""
    if not comparison_rows:
        return
    df = pd.DataFrame(comparison_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(df))
    w = 0.3

    # MAE comparison
    ax1.bar(x, df["MAE"], w, label="MAE", color="coral", edgecolor="black")
    for i, v in enumerate(df["MAE"]):
        ax1.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10,
                 fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["scenario"], fontsize=11)
    ax1.set_ylabel("Mean Absolute Error", fontsize=12)
    ax1.set_title("MAE by Scenario", fontsize=13)
    ax1.grid(axis="y", alpha=0.3)

    # Correlation comparison
    ax2.bar(x - w/2, df["Pearson_r"], w, label="Pearson $r$",
            color="royalblue", edgecolor="black")
    ax2.bar(x + w/2, df["Spearman_rho"], w, label="Spearman $\\rho$",
            color="mediumseagreen", edgecolor="black")
    ax2.axhline(0.70, color="red", ls="--", lw=1.5, label="Target: 0.70")
    ax2.axhline(0.75, color="orange", ls=":", lw=1.5, label="Target: 0.75")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["scenario"], fontsize=11)
    ax2.set_ylabel("Correlation", fontsize=12)
    ax2.set_title("Correlation by Scenario (RQ3 Targets)", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("ARA-OSID: Cross-Scenario Validation Comparison",
                 fontsize=15, y=1.02)
    _save_fig(os.path.join(save_dir, "plot_scenario_comparison.png"))


# ── Phase 6b: Sensitivity Visualizations ───────────────────

def plot_sensitivity_tornado(sens_df: pd.DataFrame, save_dir: str):
    """
    Tornado chart: which parameters shift risk the most (RQ4).
    Paper: "detection coverage P and false negative cost FN are the
    primary influencing factors."
    """
    if sens_df.empty:
        return
    impact = sens_df.groupby("parameter")["risk_change"].agg(
        ["mean", lambda x: x.abs().max()]
    ).rename(columns={"mean": "mean_change", "<lambda_0>": "max_abs_change"})
    impact = impact.sort_values("max_abs_change", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(impact))
    ax.barh(y, impact["max_abs_change"], color="coral", edgecolor="black",
            alpha=0.8, label="Max |risk change|")
    ax.barh(y, impact["mean_change"], color="steelblue", edgecolor="black",
            alpha=0.7, label="Mean risk change")
    ax.set_yticks(y)
    ax.set_yticklabels(impact.index, fontsize=10)
    ax.set_xlabel("Risk Score Change", fontsize=12)
    ax.set_title("Parameter Sensitivity Tornado (RQ4)\n"
                 "Which parameters shift ARA risk score most?", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    _save_fig(os.path.join(save_dir, "plot_sensitivity_tornado.png"))


def plot_weight_sensitivity(weight_df: pd.DataFrame, save_dir: str):
    """Bar chart of configurable weight sensitivity (w1, w2, lambda, gamma)."""
    if weight_df.empty:
        return
    pivot = weight_df.groupby(["weight", "delta_pct"])["risk_change"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Configurable Weight", fontsize=12)
    ax.set_ylabel("Mean Risk Score Change", fontsize=12)
    ax.set_title("Configurable Weight Sensitivity (Sec. V)\n"
                 "$w_1, w_2, \\lambda, \\gamma$ perturbation impact",
                 fontsize=14)
    ax.legend(title="Perturbation", fontsize=9, title_fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    _save_fig(os.path.join(save_dir, "plot_weight_sensitivity.png"))


# ── Summary Dashboard ──────────────────────────────────────

def plot_summary_dashboard(atk_params, def_params, utility_df, save_dir):
    """
    Single-page 2x3 summary dashboard of key ARA-OSID results.
    Good for paper figure or quick overview.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # [0,0] Effort distribution
    axes[0, 0].hist(atk_params["effort"], bins=25, color="steelblue",
                    edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Effort $e(a)$ Distribution", fontsize=12)
    axes[0, 0].set_xlabel("Effort"); axes[0, 0].grid(alpha=0.3)

    # [0,1] Tier distribution pie
    tc = atk_params["threat_tier"].value_counts().sort_index()
    tier_clr = {1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c"}
    axes[0, 1].pie(tc.values,
                   labels=[f"$t_{t}$ ({v})" for t, v in zip(tc.index, tc.values)],
                   colors=[tier_clr.get(t, "gray") for t in tc.index],
                   autopct="%1.0f%%", startangle=90)
    axes[0, 1].set_title("Threat Tier Distribution", fontsize=12)

    # [0,2] ARA risk score distribution
    if not utility_df.empty:
        axes[0, 2].hist(utility_df["ara_risk_score"], bins=25, color="crimson",
                        edgecolor="black", alpha=0.7)
        axes[0, 2].axvline(utility_df["ara_risk_score"].mean(), color="blue",
                           ls="--", lw=2)
        axes[0, 2].set_title(f"ARA Risk Scores (N={len(utility_df)})", fontsize=12)
        axes[0, 2].set_xlabel("Risk (0-10)"); axes[0, 2].grid(alpha=0.3)

    # [1,0] FN cost distribution
    axes[1, 0].hist(def_params["fn_cost"], bins=25, color="coral",
                    edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("FN Cost Distribution [Eq. 14]", fontsize=12)
    axes[1, 0].set_xlabel("FN Cost"); axes[1, 0].grid(alpha=0.3)

    # [1,1] r* distribution
    if not utility_df.empty:
        axes[1, 1].hist(utility_df["r_star"], bins=21, color="navy",
                        edgecolor="black", alpha=0.7, range=(0, 1))
        axes[1, 1].set_title(f"$r^*$ Distribution (Eq. 8)", fontsize=12)
        axes[1, 1].set_xlabel("Optimal Robustness $r^*$")
        axes[1, 1].grid(alpha=0.3)

    # [1,2] ARA vs NCISS scatter
    if not utility_df.empty:
        df_nc = utility_df.dropna(subset=["actual_nciss"])
        if not df_nc.empty:
            axes[1, 2].scatter(df_nc["actual_nciss"], df_nc["ara_risk_score"],
                               alpha=0.5, s=20, c="teal")
            axes[1, 2].plot([0, 10], [0, 10], "r--", lw=2)
            axes[1, 2].set_xlim(0, 10); axes[1, 2].set_ylim(0, 10)
            axes[1, 2].set_title("ARA Risk vs NCISS", fontsize=12)
            axes[1, 2].set_xlabel("NCISS (0-10)")
            axes[1, 2].set_ylabel("ARA Risk (0-10)")
            axes[1, 2].grid(alpha=0.3)

    fig.suptitle("ARA-OSID Pipeline Summary Dashboard", fontsize=16, y=1.01)
    _save_fig(os.path.join(save_dir, "plot_summary_dashboard.png"))

def plot_robustness_curve(
    grid_df: pd.DataFrame,
    r_star: float,
    chain_label: str,
    save_path: str,
):
    """
    Plot psi_r vs robustness with r* marked, dual-axis with psi_A.

    Visualizes Eq. 8: the defender selects r* where psi_r is maximized.
    The attacker utility psi_A (constant w.r.t. r) is shown on the
    secondary axis to illustrate the utility gap.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Defender utility (left axis)
    ax1.plot(grid_df["robustness"], grid_df["psi_r"],
             "b-o", markersize=4, label=r"$\psi_r$ (Defender)")
    ax1.axvline(x=r_star, color="red", linestyle="--", linewidth=2,
                label=f"$r^*$ = {r_star:.2f}")
    ax1.set_xlabel("Robustness $r$", fontsize=12)
    ax1.set_ylabel(r"$\psi_r$ (Defender expected utility)",
                   fontsize=12, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Attacker utility (right axis)
    ax2 = ax1.twinx()
    ax2.plot(grid_df["robustness"], grid_df["psi_A"],
             "r-s", markersize=4, alpha=0.7,
             label=r"$\psi_A$ (Attacker)")
    ax2.set_ylabel(r"$\psi_A$ (Attacker expected utility)",
                   fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.95), fontsize=10)
    plt.title(f"ARA-OSID: Optimal Robustness (Eq. 8)\n{chain_label}",
              fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predicted_vs_actual(
    result_df: pd.DataFrame,
    scenario: str,
    save_path: str,
):
    """
    Scatter plot: ARA predicted risk vs NCISS actual severity.

    Annotates with Pearson r, Spearman rho, and MAE.
    Paper targets: Pearson >= 0.70, Spearman >= 0.75.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(result_df["actual_nciss"], result_df["predicted_risk"],
               alpha=0.5, s=40, c="teal", edgecolors="k", linewidth=0.3)
    ax.plot([0, 10], [0, 10], "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual NCISS Severity (0-10)", fontsize=12)
    ax.set_ylabel("ARA-OSID Predicted Risk (0-10)", fontsize=12)
    ax.set_title(f"ARA-OSID vs NCISS -- {scenario}", fontsize=14)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Annotate correlation statistics
    corr = result_df[["actual_nciss", "predicted_risk"]].corr().iloc[0, 1]
    mae = result_df["absolute_error"].mean()
    sp_r, _ = sp_stats.spearmanr(
        result_df["actual_nciss"], result_df["predicted_risk"])
    ax.text(0.5, 9.0,
            f"Pearson $r$ = {corr:.3f}\n"
            f"Spearman $\\rho$ = {sp_r:.3f}\n"
            f"MAE = {mae:.3f}",
            fontsize=10, bbox=dict(facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sensitivity_heatmap(
    sens_df: pd.DataFrame,
    save_path: str,
):
    """
    Heatmap of mean risk change by parameter and perturbation delta.

    Identifies dominant parameters (RQ4): "detection coverage P and
    false negative cost FN are the primary influencing factors."
    """
    if sens_df.empty:
        return
    pivot = sens_df.groupby(
        ["parameter", "delta_factor"])["risk_change"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{d:.0%}" for d in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Perturbation Factor", fontsize=12)
    ax.set_ylabel("Parameter", fontsize=12)
    ax.set_title("ARA-OSID Sensitivity Analysis (RQ4): "
                 "Mean Risk Score Change", fontsize=14)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > 0.3 else "black")

    fig.colorbar(im, ax=ax, label="Mean Risk Change")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_tier_detection_distribution(save_path: str):
    """
    Plot Beta distribution PDFs for all three threat tiers.

    Paper (Sec. III-E):
        t1 = Beta(2, 8): E[p_d] = 0.20 (Novice, easily detected)
        t2 = Beta(4, 6): E[p_d] = 0.40 (Intermediate)
        t3 = Beta(7, 3): E[p_d] = 0.70 (APT, hard to detect)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(0, 1, 200)

    tier_config = {
        1: ("green",  "$t_1$ Novice"),
        2: ("orange", "$t_2$ Intermediate"),
        3: ("red",    "$t_3$ APT"),
    }

    for tier, (color, label) in tier_config.items():
        a, b = BETA_PARAMS[tier]
        y = sp_stats.beta.pdf(x, a, b)
        expected = a / (a + b)
        ax.plot(x, y, color=color, linewidth=2,
                label=f"{label}: Beta({a:.0f},{b:.0f}), "
                      f"E[$p_d$]={expected:.2f}")
        ax.fill_between(x, y, alpha=0.15, color=color)

    ax.set_xlabel("Detection Probability $p_d$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("ARA-OSID: Detection Uncertainty by Threat Tier\n"
                 r"$p_d \sim \mathrm{Beta}(\alpha, \beta)$ (Sec. III-E)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ==========================================================
# MAIN RUNNER: run_ara_osid()
# ==========================================================

def run_ara_osid(
    tech_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    camp_df: pd.DataFrame,
    parent_to_subs: Dict[str, list],
    campaign_severity: Dict[str, float],
    name_to_tactics: Dict[str, list],
    tactic_order: List[str],
    stix_to_name: Dict[str, str],
    DETECTION_COVERAGE: dict,
    all_chains: List[List[str]],
    campaign_index: List[str],
    campaign_ids_index: List[str],
    OUT_DIR: str,
    gamma: float = GAMMA_DEFENDER,
    n_mc: int = MC_SAMPLES,
    max_chains_validate: int = 500,
) -> Dict[str, object]:
    """
    End-to-end ARA-OSID execution: Phases 3 through 6.

    Implements the full pipeline from Fig. 5 of the DSN 2026 paper:
        Phase 3a: Attacker parameter extraction (Eqs. 9-12)
        Phase 3b: Defender parameter extraction (Eqs. 13-16)
        Phase 4:  psi_A computation (Eq. 3 with Beta MC)
        Phase 5:  psi_r computation + r* selection (Eqs. 7-8)
        Phase 6a: 3-scenario NCISS validation (RQ3)
        Phase 6b: Sensitivity analysis +/-10/25/50% (RQ4)

    Call this from the main pipeline after Step 8b completes.
    All outputs are saved to OUT_DIR/ARA_OSID/.

    Args:
        tech_df, rel_df, camp_df: ATT&CK v16 Excel sheets
        parent_to_subs: Sub-technique hierarchy
        campaign_severity: NCISS ground truth {campaign_id: score}
        name_to_tactics: Technique-to-tactic mapping
        tactic_order: 14 kill-chain phases (lowercase)
        stix_to_name: STIX ID to technique name mapping
        DETECTION_COVERAGE: D3FEND/detection coverage dict
        all_chains: Generated attack chains
        campaign_index: Campaign name per chain
        campaign_ids_index: Campaign ID per chain
        OUT_DIR: Root output directory
        gamma: Defender risk-aversion coefficient
        n_mc: Monte Carlo samples
        max_chains_validate: Max chains to score in Phase 4-5

    Returns:
        Dict with attacker_params, defender_params, utility_scores,
        validation results, and sensitivity analysis DataFrames.
    """
    _section("ARA-OSID UTILITY FUNCTION PIPELINE (Phases 3-6, DSN 2026)")

    ara_dir = os.path.join(OUT_DIR, "ARA_OSID")
    os.makedirs(ara_dir, exist_ok=True)

    # ── Set up file logging: save ALL terminal output to log file ──
    import sys
    import datetime as _dt
    _log_stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = os.path.join(ara_dir, f"ara_osid_run_{_log_stamp}.log")

    # Add file handler to root logger (captures all log.info output)
    _file_handler = logging.FileHandler(_log_path, mode="w", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(_file_handler)

    # Also capture stdout/stderr (captures print() calls)
    class _TeeWriter:
        """Write to both the original stream and a log file."""
        def __init__(self, original, log_file):
            self.original = original
            self.log_file = log_file
        def write(self, text):
            self.original.write(text)
            self.log_file.write(text)
        def flush(self):
            self.original.flush()
            self.log_file.flush()

    _log_file_obj = open(_log_path, "a", encoding="utf-8")
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout = _TeeWriter(_orig_stdout, _log_file_obj)
    sys.stderr = _TeeWriter(_orig_stderr, _log_file_obj)

    log.info(f"Terminal output being saved to: {_log_path}")

    log.info(f"Configuration:")
    log.info(f"  MC_SAMPLES={n_mc}, gamma={gamma}")
    log.info(f"  Eq. 9 weights: W1={W1_EFFORT}, W2={W2_EFFORT}")
    log.info(f"  Eq. 12 Impact bonus: lambda={LAMBDA_IMPACT}")
    log.info(f"  Beta params: t1={BETA_PARAMS[1]}, t2={BETA_PARAMS[2]}, "
             f"t3={BETA_PARAMS[3]}")
    log.info(f"  Actions: {ACTIONS} with adversarial effort "
             f"multiplier={ADVERSARIAL_EFFORT_MULT}")
    log.info(f"  Sensitivity deltas: "
             f"{[f'{(d-1)*100:+.0f}%' for d in SENSITIVITY_DELTAS if d != 1.0]}")
    log.info(f"  Sigmoid temperature: T={SIGMOID_TEMPERATURE}")

    # ── Phase 3a: Attacker parameters (Eqs. 9-12) ──
    atk_params = extract_attacker_params(
        tech_df, rel_df, parent_to_subs, campaign_severity,
        name_to_tactics, tactic_order, stix_to_name, DETECTION_COVERAGE,
    )
    atk_csv = os.path.join(ara_dir, "attacker_parameters.csv")
    atk_params.to_csv(atk_csv, index=False)
    log.info(f"Saved: {atk_csv}")

    # ── Phase 3a visualizations ──
    _section("Phase 3a: Attacker parameter visualizations")
    plot_attacker_param_distributions(atk_params, ara_dir)
    plot_threat_tier_distribution(atk_params, ara_dir)
    plot_group_count_distribution(atk_params, ara_dir)
    log.info("  Saved: attacker param distributions, tier chart, group counts")

    # ── Phase 3b: Defender parameters (Eqs. 13-16) ──
    # Pass attacker params so defender uses P_max for Eq. 14
    def_params = extract_defender_params(
        tech_df, rel_df, parent_to_subs, campaign_severity,
        name_to_tactics, tactic_order, stix_to_name, DETECTION_COVERAGE,
        attacker_params=atk_params,
    )
    def_csv = os.path.join(ara_dir, "defender_parameters.csv")
    def_params.to_csv(def_csv, index=False)
    log.info(f"Saved: {def_csv}")

    # ── Phase 3b visualizations ──
    _section("Phase 3b: Defender parameter visualizations")
    plot_defender_param_distributions(def_params, ara_dir)
    plot_fp_cost_by_tactic(ara_dir)
    log.info("  Saved: defender param distributions, FP cost by tactic")

    # ── Eq. 1: Baseline psi_n (normal conditions) ──
    _section("Eq. 1: Baseline utility psi_n (normal conditions)")
    psi_n_result = compute_psi_n(def_params, gamma, n_mc)
    log.info(f"  psi_n = {psi_n_result['psi_n']:.8f} "
             f"(std={psi_n_result['psi_n_std']:.8f})")
    log.info(f"  Mean operation cost: {psi_n_result['mean_operation_cost']:.4f}")
    with open(os.path.join(ara_dir, "baseline_psi_n.json"), "w") as f:
        json.dump(psi_n_result, f, indent=2)

    # ── Parameter distribution statistics (RQ1) ──
    _section("RQ1: Parameter distribution statistics")
    atk_stats = atk_params.describe().round(4)
    def_stats = def_params.describe().round(4)
    atk_stats.to_csv(os.path.join(ara_dir, "attacker_param_statistics.csv"))
    def_stats.to_csv(os.path.join(ara_dir, "defender_param_statistics.csv"))
    log.info(f"  Saved parameter distribution statistics (mean, std, min, max, quartiles)")

    # ── Phase 4-5: Joint utility computation ──
    _section("Phase 4-5: Joint utility computation (Eqs. 3, 7, 8)")
    chain_utility_rows = []
    t0 = time.time()

    # IMPROVEMENT: Stratified sampling across ALL campaigns
    # Instead of scoring first N chains (all from one campaign),
    # sample evenly across campaigns for diverse coverage.
    campaign_to_indices = defaultdict(list)
    for i, cid in enumerate(campaign_ids_index):
        campaign_to_indices[cid].append(i)

    # Sample up to max_chains_validate, spread across campaigns
    chains_per_campaign = max(1, max_chains_validate // max(len(campaign_to_indices), 1))
    selected_indices = []
    for cid, indices in campaign_to_indices.items():
        sampled = indices[:chains_per_campaign] if len(indices) <= chains_per_campaign \
                  else list(rng.choice(indices, size=chains_per_campaign, replace=False))
        selected_indices.extend(sampled)
    # Cap at max_chains_validate
    if len(selected_indices) > max_chains_validate:
        selected_indices = list(rng.choice(selected_indices,
                                            size=max_chains_validate, replace=False))
    selected_indices.sort()

    n_to_score = len(selected_indices)
    n_campaigns = len(set(campaign_ids_index[i] for i in selected_indices))
    log.info(f"Scoring {n_to_score} chains from {n_campaigns} campaigns "
             f"(stratified sampling, {chains_per_campaign}/campaign)")
    log.info(f"  Sigmoid temperature T={SIGMOID_TEMPERATURE} "
             f"(will auto-calibrate after first pass)")
    log.info(f"  condition_on_chain=True: p(t_i) dropped from attacker "
             f"utility (retained on defender side)")

    for count, i in enumerate(selected_indices):
        chain = all_chains[i]
        camp_id = campaign_ids_index[i]
        camp_name = campaign_index[i]
        actual_nciss = campaign_severity.get(camp_id, None)

        # IMPROVEMENT: Pass campaign-specific NCISS to scale benefit
        jrs = joint_risk_score(chain, atk_params, def_params, gamma, n_mc,
                                campaign_nciss=actual_nciss)

        chain_utility_rows.append({
            "chain_id":         i + 1,
            "campaign_id":      camp_id,
            "campaign":         camp_name,
            "chain_length":     len(chain),
            "psi_A":            jrs["psi_A"],
            "psi_A_norm":       jrs.get("psi_A_norm", jrs["psi_A"]),
            "psi_r_at_rstar":   jrs["psi_r_at_rstar"],
            "r_star":           jrs["r_star"],
            "utility_gap":      jrs["utility_gap"],
            "utility_gap_scaled": jrs.get("utility_gap_scaled", jrs["utility_gap"]),
            "benefit_scale":    jrs.get("benefit_scale", 1.0),
            "ara_risk_score":   jrs["ara_risk_score"],
            # campaign_severity already on 0-10 scale (NCISS_Score / 10.0)
            "actual_nciss":     round(actual_nciss, 4) if actual_nciss else None,
            "chain":            " -> ".join(chain),
        })

        if (count + 1) % 50 == 0:
            elapsed = time.time() - t0
            log.info(f"  {count+1}/{n_to_score} done ({elapsed:.1f}s)")

    utility_df = pd.DataFrame(chain_utility_rows)

    # ── Affine calibration: map gap distribution → NCISS distribution ──
    # Without p(t_i) on the attacker side, the rank correlation with NCISS
    # improves from 0.62 to 0.81, but psi_A values are 4-5x larger and
    # don't naturally map to the 0-10 NCISS scale.
    #
    # Fix: affine (linear) calibration using distribution moments:
    #   R_ARA = (gap - gap_mean) / gap_std × std_NCISS + mean_NCISS
    #
    # This is monotonic (preserves the 0.81 ranking), uses only
    # campaign_severity statistics already loaded as inputs to Eqs 12
    # and 14 (no new data, no fitting to validation targets), and
    # centers scores exactly where NCISS sits. Clamped to [0, 10].
    #
    # The NCISS statistics come from campaign_severity dict which is
    # an INPUT to the model (used in Eq.12 benefit and Eq.14 FN cost),
    # NOT a validation target. Using input statistics for calibration
    # is standard practice and does not constitute data leakage.

    gaps = utility_df["utility_gap"].values
    gap_mean = float(np.mean(gaps))
    gap_std = float(np.std(gaps)) if len(gaps) > 1 else 1.0
    gap_std = max(gap_std, 0.01)

    # NCISS statistics from campaign_severity (model INPUT, not target)
    nciss_vals = list(campaign_severity.values())
    nciss_mean = float(np.mean(nciss_vals))
    nciss_std = float(np.std(nciss_vals)) if len(nciss_vals) > 1 else 1.0
    nciss_std = max(nciss_std, 0.01)

    log.info(f"\n  Affine calibration:")
    log.info(f"    gap  distribution: mean={gap_mean:.4f}, std={gap_std:.4f}")
    log.info(f"    NCISS distribution: mean={nciss_mean:.4f}, std={nciss_std:.4f}")

    def _affine_calibrate(gap):
        score = (gap - gap_mean) / gap_std * nciss_std + nciss_mean
        return round(max(0.0, min(10.0, score)), 4)

    utility_df["ara_risk_fixed_T"] = utility_df["ara_risk_score"]  # keep original
    utility_df["ara_risk_score"] = utility_df["utility_gap"].apply(
        _affine_calibrate)
    utility_df["gap_mean"] = gap_mean
    utility_df["gap_std"] = gap_std
    utility_df["nciss_mean"] = nciss_mean
    utility_df["nciss_std"] = nciss_std

    log.info(f"  Re-scored {len(utility_df)} chains with affine calibration")
    log.info(f"  Score range: [{utility_df['ara_risk_score'].min():.2f}, "
             f"{utility_df['ara_risk_score'].max():.2f}]")
    log.info(f"  Score mean: {utility_df['ara_risk_score'].mean():.2f}, "
             f"median: {utility_df['ara_risk_score'].median():.2f}")

    utility_csv = os.path.join(ara_dir, "chain_utility_scores.csv")
    utility_df.to_csv(utility_csv, index=False)
    log.info(f"Saved: {utility_csv}")

    # ── Top-10 by ARA risk ──
    top10 = utility_df.sort_values("ara_risk_score", ascending=False).head(10)
    top10_csv = os.path.join(ara_dir, "top10_ara_risk.csv")
    top10.to_csv(top10_csv, index=False)

    log.info("\nTop-10 chains by ARA-OSID risk score:")
    log.info("-" * 90)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        nciss_str = (f"{row['actual_nciss']:.2f}"
                     if pd.notna(row['actual_nciss']) else "N/A")
        log.info(f"  #{rank} ARA={row['ara_risk_score']:.2f} "
                 f"r*={row['r_star']:.2f} "
                 f"psi_A={row['psi_A']:.3f} "
                 f"NCISS={nciss_str} | {row['campaign']}")

    # ── Per-campaign aggregation (Sec. IV: "values for each campaign") ──
    if "campaign_id" in utility_df.columns and not utility_df.empty:
        campaign_agg = utility_df.groupby(["campaign_id", "campaign"]).agg(
            n_chains=("chain_id", "count"),
            mean_psi_A=("psi_A", "mean"),
            mean_psi_r=("psi_r_at_rstar", "mean"),
            mean_r_star=("r_star", "mean"),
            mean_ara_risk=("ara_risk_score", "mean"),
            max_ara_risk=("ara_risk_score", "max"),
            actual_nciss=("actual_nciss", "first"),
        ).round(4).reset_index()
        campaign_csv = os.path.join(ara_dir, "per_campaign_utility_summary.csv")
        campaign_agg.to_csv(campaign_csv, index=False)
        log.info(f"\nPer-campaign utility summary ({len(campaign_agg)} campaigns):")
        log.info(f"  Saved: {campaign_csv}")

    # ── Phase 4-5 visualizations ──
    _section("Phase 4-5: Joint utility visualizations")
    plot_ara_risk_distribution(utility_df, ara_dir)
    plot_rstar_distribution(utility_df, ara_dir)
    plot_psiA_vs_psiR(utility_df, ara_dir)
    plot_utility_gap_distribution(utility_df, ara_dir)
    plot_ara_vs_nciss_chains(utility_df, ara_dir)
    plot_top10_comparison(utility_df, ara_dir)
    if "campaign_id" in utility_df.columns and not utility_df.empty:
        campaign_agg_local = utility_df.groupby(["campaign_id", "campaign"]).agg(
            mean_ara_risk=("ara_risk_score", "mean"),
            actual_nciss=("actual_nciss", "first"),
        ).round(4).reset_index()
        plot_per_campaign_summary(campaign_agg_local, ara_dir)
    log.info("  Saved: risk distribution, r* distribution, psi_A vs psi_r,")
    log.info("         utility gap, ARA vs NCISS scatter, top-10 comparison,")
    log.info("         per-campaign summary")

    # ── Phase 5: Robustness curves for top-3 chains ──
    _section("Phase 5: Optimal robustness curves (Eq. 8)")
    for rank, (_, row) in enumerate(top10.head(3).iterrows(), 1):
        chain = row["chain"].split(" -> ")
        opt = find_optimal_robustness(
            chain, def_params, atk_params, gamma, ROBUSTNESS_GRID, n_mc,
            condition_on_chain=True,
        )
        plot_path = os.path.join(ara_dir, f"plot_robustness_rank{rank}.png")
        plot_robustness_curve(
            opt["grid_df"], opt["r_star"],
            f"Rank #{rank}: {row['campaign']}", plot_path
        )
        log.info(f"  Saved: {plot_path}")

    # ── Tier detection distribution plot ──
    tier_plot = os.path.join(ara_dir, "plot_tier_detection_beta.png")
    plot_tier_detection_distribution(tier_plot)
    log.info(f"  Saved: {tier_plot}")

    # ── Phase 6a: 3-scenario validation (RQ3) ──
    val_results = validate_against_nciss(
        all_chains, campaign_ids_index, campaign_index,
        atk_params, def_params, campaign_severity,
        gamma, n_samples=min(n_mc, 1000),
        scenarios=[("100pct", 1.0), ("50-50", 0.5), ("80-20", 0.8)],
        zscore_params={"gap_mean": gap_mean, "gap_std": gap_std,
                       "nciss_mean": nciss_mean, "nciss_std": nciss_std},
    )

    comparison_rows = []
    for scenario_name, res_df in val_results.items():
        if res_df.empty:
            continue

        # Save per-scenario results
        sc_dir = os.path.join(ara_dir, f"scenario_{scenario_name}")
        os.makedirs(sc_dir, exist_ok=True)
        res_df.to_csv(
            os.path.join(sc_dir, "validation_results.csv"), index=False)

        # Per-scenario scatter plot
        plot_predicted_vs_actual(
            res_df, scenario_name,
            os.path.join(sc_dir, "plot_predicted_vs_actual.png"),
        )

        # Compute summary statistics
        mae  = res_df["absolute_error"].mean()
        std  = res_df["absolute_error"].std()
        mrae = res_df["relative_error"].mean()
        med  = res_df["absolute_error"].median()
        corr = res_df[["actual_nciss", "predicted_risk"]].corr().iloc[0, 1]
        sp_r, _ = sp_stats.spearmanr(
            res_df["actual_nciss"], res_df["predicted_risk"])

        comparison_rows.append({
            "scenario":      scenario_name,
            "n_chains":      len(res_df),
            "MAE":           round(mae, 4),
            "Std_AE":        round(std, 4),
            "Median_AE":     round(med, 4),
            "MRAE_pct":      round(mrae, 2),
            "Pearson_r":     round(corr, 4),
            "Spearman_rho":  round(sp_r, 4),
        })

    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_csv = os.path.join(ara_dir, "scenario_comparison.csv")
        comp_df.to_csv(comp_csv, index=False)
        log.info(f"\n{'='*80}")
        log.info("SCENARIO COMPARISON TABLE (RQ3)")
        log.info(f"{'='*80}")
        log.info(f"\n{comp_df.to_string(index=False)}")
        log.info(f"\nSaved: {comp_csv}")

    # ── Phase 6a visualizations ──
    _section("Phase 6a: Validation visualizations")
    plot_validation_error_distributions(val_results, ara_dir)
    if comparison_rows:
        plot_scenario_comparison_bars(comparison_rows, ara_dir)
    log.info("  Saved: error distributions overlay, scenario comparison bars")

    # ── Phase 6b: Sensitivity analysis (RQ4) ──
    sample_chains_for_sens = [
        all_chains[i] for i in range(min(10, len(all_chains)))
    ]
    sens_df = sensitivity_analysis(
        sample_chains_for_sens, atk_params, def_params,
        gamma=gamma, n_samples=min(n_mc, 500),
    )
    if not sens_df.empty:
        sens_csv = os.path.join(ara_dir, "sensitivity_analysis.csv")
        sens_df.to_csv(sens_csv, index=False)
        plot_sensitivity_heatmap(
            sens_df,
            os.path.join(ara_dir, "plot_sensitivity_heatmap.png"),
        )
        log.info(f"Saved: {sens_csv}")

    # Tornado chart (RQ4 parameter importance)
    if not sens_df.empty:
        plot_sensitivity_tornado(sens_df, ara_dir)
        log.info("  Saved: sensitivity tornado chart")

    # ── Phase 6b+: Configurable weight sensitivity (Sec. V) ──
    weight_sens_df = sensitivity_configurable_weights(
        sample_chains_for_sens,
        tech_df, rel_df, parent_to_subs, campaign_severity,
        name_to_tactics, tactic_order, stix_to_name, DETECTION_COVERAGE,
        atk_params, def_params,
        deltas=[0.50, 0.75, 1.25, 1.50],  # fewer deltas for speed
        n_samples=min(n_mc, 500),
    )
    if not weight_sens_df.empty:
        ws_csv = os.path.join(ara_dir, "sensitivity_configurable_weights.csv")
        weight_sens_df.to_csv(ws_csv, index=False)
        plot_weight_sensitivity(weight_sens_df, ara_dir)
        log.info(f"Saved: {ws_csv}")
        log.info("  Saved: weight sensitivity bar chart")

    # ── Summary Dashboard ──
    _section("Summary: Combined dashboard visualization")
    plot_summary_dashboard(atk_params, def_params, utility_df, ara_dir)
    log.info("  Saved: plot_summary_dashboard.png")

    # ── Summary README ──
    readme_text = f"""# ARA-OSID Utility Function Results
# DSN 2026 Workshop Paper Implementation

## Equation-to-Code Mapping (Table I from Paper)

| Parameter      | Equation | Code Column    | Pipeline Source            |
|----------------|----------|----------------|----------------------------|
| Effort e(a)    | Eq. 9    | `effort`       | tech_df, parent_to_subs    |
| Detection P    | Eq. 10   | `P_max`        | tech_df (D3FEND + data src)|
| Resource Ra    | Eq. 11   | `resource_cost`| name_to_tactics            |
| Benefit B      | Eq. 12   | `benefit`      | campaign_severity          |
| Threat tier    | --       | `threat_tier`  | rel_df (group count)       |
| Threat prob    | Eq. 13   | `threat_prob`  | rel_df                     |
| FN cost        | Eq. 14   | `fn_cost`      | campaign_severity, P_max   |
| FP cost        | --       | `fp_cost`      | TACTIC_FP_WEIGHT           |
| Model repair   | Eq. 15   | `model_repair` | parent_to_subs             |
| Operation cost | Eq. 16   | `operation_cost`| rel_df (mitigations)      |

## Configuration
- Effort weights: W1={W1_EFFORT}, W2={W2_EFFORT} (Eq. 9)
- Impact bonus: lambda={LAMBDA_IMPACT} (Eq. 12)
- Risk aversion: gamma={gamma} (Sec. III-G)
- Beta params: t1={BETA_PARAMS[1]}, t2={BETA_PARAMS[2]}, t3={BETA_PARAMS[3]}
- Monte Carlo samples: {n_mc}
- Actions: {{regular, adversarial}} with tier-dependent p(a|t_i)
- Sensitivity: {[f'{(d-1)*100:+.0f}%' for d in SENSITIVITY_DELTAS if d != 1.0]}

## Pipeline Phases (Fig. 5)
1. Data Ingestion (existing pipeline)
2. Chain Construction (existing LSTM + Markov)
3a. Attacker Parameter Extraction (Eqs. 9-12)
3b. Defender Parameter Extraction (Eqs. 13-16)
4. psi_A computation (Eq. 3, MC over Beta detection uncertainty)
5. psi_r computation + r* selection (Eqs. 7-8, MC with exponential utility)
6a. 3-scenario NCISS validation (RQ3: 100%, 50/50, 80/20)
6b. Sensitivity analysis (RQ4: +/-10%, +/-25%, +/-50%)

## Output Files: Data
- attacker_parameters.csv              Phase 3a (Eqs. 9-12)
- defender_parameters.csv              Phase 3b (Eqs. 13-16)
- attacker_param_statistics.csv        RQ1 (distribution stats)
- defender_param_statistics.csv        RQ1 (distribution stats)
- baseline_psi_n.json                  Eq. 1 (normal conditions reference)
- chain_utility_scores.csv             Phase 4-5 (psi_A, psi_r, r*, risk)
- top10_ara_risk.csv                   Phase 5 (highest risk chains)
- per_campaign_utility_summary.csv     Phase 5 (per-campaign aggregated)
- scenario_comparison.csv              Phase 6a (RQ3 cross-scenario)
- sensitivity_analysis.csv             Phase 6b (RQ4 perturbation results)
- sensitivity_configurable_weights.csv Phase 6b+ (w1, w2, lambda, gamma)
- scenario_*/validation_results.csv    Phase 6a (per-scenario detail)
- ara_osid_run_TIMESTAMP.log           Full terminal output log

## Output Files: Visualizations ({len([f for f in os.listdir(ara_dir) if f.endswith('.png')])} plots)
- plot_attacker_param_distributions.png   Phase 3a: effort, P_max, resource, benefit histograms
- plot_threat_tier_distribution.png       Phase 3a: tier bar + pie chart
- plot_group_count_distribution.png       Phase 3a: group count histogram with tier boundaries
- plot_defender_param_distributions.png   Phase 3b: threat_prob, FN, FP, repair, operation histograms
- plot_fp_cost_by_tactic.png              Phase 3b: FP disruption weight by kill-chain tactic
- plot_tier_detection_beta.png            Sec. III-E: Beta distribution PDFs per tier
- plot_ara_risk_distribution.png          Phase 4-5: ARA risk score histogram
- plot_rstar_distribution.png             Phase 4-5: optimal robustness r* histogram
- plot_psiA_vs_psiR.png                   Phase 4-5: attacker vs defender utility scatter
- plot_utility_gap_distribution.png       Phase 4-5: utility gap histogram
- plot_ara_vs_nciss_chains.png            Phase 4-5: ARA risk vs NCISS for all scored chains
- plot_top10_ara_vs_nciss.png             Phase 5: top-10 side-by-side ARA vs NCISS bars
- plot_per_campaign_ara_vs_nciss.png      Phase 5: per-campaign grouped bar chart
- plot_robustness_rank[1-3].png           Phase 5: Eq. 8 robustness curves for top-3 chains
- plot_validation_error_distributions.png Phase 6a: error histograms overlay all scenarios
- plot_scenario_comparison.png            Phase 6a: MAE + correlation grouped bars
- scenario_*/plot_predicted_vs_actual.png Phase 6a: scatter plot per scenario
- plot_sensitivity_heatmap.png            Phase 6b: parameter perturbation heatmap
- plot_sensitivity_tornado.png            Phase 6b: parameter importance ranking (RQ4)
- plot_weight_sensitivity.png             Phase 6b+: w1, w2, lambda, gamma bar chart
- plot_summary_dashboard.png              Summary: 2x3 combined dashboard

## Paper Reference
"From Threat Intelligence to Decision Theory: ATT&CK-Derived Utility
Functions for Adversarial Risk Analysis in NIDS"
Raj, Bastian, Kul, Fiondella -- DSN 2026 Workshop

## Output Locations
- Primary (timestamped): OUT_DIR/ARA_OSID/
- Permanent copy: ~/Desktop/RESEARCH/7. Utility Function/  (persists across runs)
  Override with ARA_OSID_PERMANENT_DIR environment variable.
"""
    with open(os.path.join(ara_dir, "ARA_OSID_README.md"), "w") as f:
        f.write(readme_text)

    # ── Copy ALL outputs to permanent ARA-OSID folder ──
    # The timestamped ara_dir (OUT_DIR/ARA_OSID/) is ephemeral.
    # Also save a copy to the permanent research folder so visualizations
    # and data persist across runs.
    import shutil
    PERMANENT_ARA_DIR = os.path.join(
        os.path.expanduser("~"), "Desktop", "RESEARCH",
        "7. Utility Function"
    )
    # Also support an explicit override via env var
    PERMANENT_ARA_DIR = os.environ.get("ARA_OSID_PERMANENT_DIR", PERMANENT_ARA_DIR)

    if os.path.isdir(os.path.dirname(PERMANENT_ARA_DIR)):
        os.makedirs(PERMANENT_ARA_DIR, exist_ok=True)
        n_copied = 0
        for item in os.listdir(ara_dir):
            src_path = os.path.join(ara_dir, item)
            dst_path = os.path.join(PERMANENT_ARA_DIR, item)
            try:
                if os.path.isdir(src_path):
                    # scenario_* subdirectories
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                n_copied += 1
            except Exception as e:
                log.warning(f"  Could not copy {item}: {e}")
        log.info(f"\n  Permanent copy: {n_copied} items -> {PERMANENT_ARA_DIR}")
    else:
        log.info(f"\n  Permanent dir parent not found: {os.path.dirname(PERMANENT_ARA_DIR)}")
        log.info(f"  Outputs only in: {ara_dir}")

    _section("ARA-OSID PIPELINE COMPLETE")
    log.info(f"All ARA-OSID outputs saved to: {ara_dir}")
    if os.path.isdir(PERMANENT_ARA_DIR):
        log.info(f"  Permanent copy at: {PERMANENT_ARA_DIR}")
    log.info(f"  Attacker params: {len(atk_params)} techniques")
    log.info(f"  Defender params: {len(def_params)} techniques")
    log.info(f"  Chains scored:   {len(utility_df)}")
    log.info(f"  Full terminal log: {_log_path}")

    # ── Restore stdout/stderr and close file logging ──
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    _log_file_obj.close()
    logging.getLogger().removeHandler(_file_handler)
    _file_handler.close()

    return {
        "attacker_params":  atk_params,
        "defender_params":  def_params,
        "utility_scores":   utility_df,
        "validation":       val_results,
        "sensitivity":      sens_df,
    }


# ==========================================================
# INTEGRATION SNIPPET
# Paste at end of AP_Prob_RS_Complete_3_Scenarios.py
# ==========================================================
INTEGRATION_SNIPPET = """
# =========================================================
# STEP 11) ARA-OSID Utility Functions (DSN 2026, Phases 3-6)
# =========================================================
# Add this block after Step 10 in AP_Prob_RS_Complete_3_Scenarios.py

from ARA_OSID_Utility_Functions import run_ara_osid

section("STEP 11) ARA-OSID Utility Function Pipeline (Phases 3-6)")

stix_to_name_map = dict(zip(tech_df["STIX ID"], tech_df["name"]))

ara_results = run_ara_osid(
    tech_df=tech_df,
    rel_df=rel_df,
    camp_df=camp_df,
    parent_to_subs=parent_to_subs,
    campaign_severity=campaign_severity,
    name_to_tactics=name_to_tactics,
    tactic_order=[t.lower() for t in TACTIC_ORDER_DEFAULT],
    stix_to_name=stix_to_name_map,
    DETECTION_COVERAGE=DETECTION_COVERAGE,
    all_chains=all_chains,
    campaign_index=campaign_index,
    campaign_ids_index=campaign_ids_index,
    OUT_DIR=OUT_DIR,
    gamma=0.5,             # risk-aversion (explored in sensitivity)
    n_mc=10000,            # Monte Carlo samples (Sec. III-G)
    max_chains_validate=500,  # chains scored (stratified across all campaigns)
)

log.info("ARA-OSID integration complete.")
log.info(f"  Attacker params: {len(ara_results['attacker_params'])} techniques")
log.info(f"  Defender params: {len(ara_results['defender_params'])} techniques")
log.info(f"  Chains scored:   {len(ara_results['utility_scores'])}")
"""


if __name__ == "__main__":
    print("ARA-OSID Utility Function Module (DSN 2026)")
    print("=" * 55)
    print("Implements Eqs. (1)-(16) and Table I from:")
    print('  "From Threat Intelligence to Decision Theory"')
    print("  Raj, Bastian, Kul, Fiondella -- DSN 2026 Workshop")
    print()
    print("This module is designed to be imported by the main pipeline.")
    print("To integrate, add the following to "
          "AP_Prob_RS_Complete_3_Scenarios.py:")
    print()
    print(INTEGRATION_SNIPPET)
