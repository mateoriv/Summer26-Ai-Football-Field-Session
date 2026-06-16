#!/usr/bin/env python3
"""
PRE-TEST: real-yard "distance to home position" matching vs the old RMS matcher.

Question we're answering before committing to a plan: does measuring each player's
ACTUAL yard distance to its template home position (instead of RMS-normalizing the
whole shape, as template_matcher does) raise base-family / exact accuracy above the
current 10.2% exact / 23.4% family?

We test the new coordinate ideas one lever at a time so we can see which part helps:

  V0  old matcher (RMS scale + PCA + mirror + role-agnostic)   [the baseline]
  V1  real yards (NO RMS), bbox-CENTER projection, role-agnostic
  V2  real yards, FEET projection (bbox bottom = y2), role-agnostic
  V3  real yards, feet, ROLE-ANCHORED (QB->Q, back->T, recv->recv slots)

All read-only on the existing cache. Scored against the official breakdown.xlsx
OFF FORM labels (clip N = row N-1).
"""
import csv, glob, json, math, os, re, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import template_matcher as tm
from perFrameHomographyTransform import (
    gather_window_correspondences, get_homography_matrix,
    field_points_are_degenerate, transform_point,
)

FOLDER = "CSAI_FORMATIONS"
CACHE = os.path.join(ROOT, "cache")
ROLE_ORDER = ["Q", "T", "W", "S", "X", "Y", "Z", "U"]
RECV_ROLES = {"W", "S", "X", "Y", "Z", "U"}  # everything that isn't QB(Q)/back(T)


# ---------- labels ----------
def labels():
    off = pd.read_excel(os.path.join(ROOT, "data", FOLDER, "breakdown.xlsx"))["OFF FORM"].tolist()
    def f(n): return off[n-1].strip() if 1 <= n <= len(off) and isinstance(off[n-1], str) and off[n-1].strip() else None
    return f

def norm_pred(t):
    if not t: return None
    name = t.upper().replace("_", " ").replace("TIGHT", "TITE").strip()
    return "DENVER U OFF" if name == "PRO" else name

fam = lambda s: (s or "").split()[0] if s else None


# ---------- templates with roles ----------
def load_role_templates():
    out = []
    with open(tm.DEFAULT_TEMPLATE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("formation") or "").strip()
            if not name: continue
            roles = []
            for r in ROLE_ORDER:
                xs, ys = (row.get(f"{r}_x") or "").strip(), (row.get(f"{r}_y") or "").strip()
                if xs and ys:
                    roles.append((r, float(xs), float(ys)))
            if len(roles) >= 3:
                out.append({"name": name, "roles": roles})
    return out


# ---------- per-clip geometry ----------
def build_H(corr, snap_frame):
    for fr in range(max(0, snap_frame - 20), snap_frame + 21):
        ip, fp = gather_window_correspondences(corr, fr, 15)
        if len(ip) < 4 or field_points_are_degenerate(fp):
            continue
        return get_homography_matrix(ip, fp)
    return None

def projected_points(cleaned, H, use_feet):
    """Return dict class-> list of (fx, fy) field yards, using feet or center."""
    out = collections.defaultdict(list)
    for t in cleaned:
        b = t.get("bbox") or {}
        cx = b.get("center_x")
        py = b.get("y2") if use_feet else b.get("center_y")  # feet = bottom edge
        if cx is None or py is None:
            continue
        p = transform_point((cx, py), H)
        if p is None:
            continue
        out[t["class"]].append((float(p[0]), float(p[1])))
    return out

def real_yard_frame(proj):
    """Field-axis real-yard frame: depth = along-field rel LOS, lateral = across.
    LOS/center from the O-line; depth oriented so the backfield is negative."""
    ol = proj.get(tm.OL_CLASS_NAME, [])
    skill = [p for c in tm.SKILL_CLASSES for p in proj.get(c, [])]
    if len(skill) < 3:
        return None
    anchor = np.asarray(ol if len(ol) >= 2 else skill, dtype=float)
    los_x = float(np.median(anchor[:, 0]))
    ctr_y = float(np.median(anchor[:, 1]))
    back = np.asarray(proj.get("qb", []) + proj.get("running_back", []), dtype=float)

    def to_frame(pts):
        a = np.asarray(pts, dtype=float)
        depth = a[:, 0] - los_x
        lateral = a[:, 1] - ctr_y
        return depth, lateral
    # orient depth so QB/back are negative
    sign = 1.0
    if back.size:
        bd = back[:, 0] - los_x
        if bd.mean() > 0:
            sign = -1.0
    framed = {}
    for cls in list(tm.SKILL_CLASSES):
        if proj.get(cls):
            d, l = to_frame(proj[cls])
            framed[cls] = np.column_stack([l, sign * d])  # (lateral, depth)
    return framed


# ---------- matchers ----------
def _assign_cost(det, tpl):
    cost = np.linalg.norm(det[:, None, :] - tpl[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())

def match_roleagnostic(framed, role_templates):
    det = np.vstack(list(framed.values()))
    mir = det.copy(); mir[:, 0] = -mir[:, 0]
    best = (None, 1e9)
    for t in role_templates:
        tpl = np.array([[x, y] for _, x, y in t["roles"]], dtype=float)
        d = min(_assign_cost(det, tpl), _assign_cost(mir, tpl))
        if d < best[1]:
            best = (t["name"], d)
    return best[0]

def match_roleanchored(framed, role_templates):
    qb = framed.get("qb", np.empty((0, 2)))
    back = framed.get("running_back", np.empty((0, 2)))
    recv = np.vstack([framed[c] for c in ("wide_receiver", "tight_end") if c in framed]) \
        if any(c in framed for c in ("wide_receiver", "tight_end")) else np.empty((0, 2))
    if len(recv) == 0 and len(qb) == 0 and len(back) == 0:
        return None

    def score(det_qb, det_back, det_recv, t):
        tot, n = 0.0, 0
        rr = [(x, y) for r, x, y in t["roles"] if r in RECV_ROLES]
        q = [(x, y) for r, x, y in t["roles"] if r == "Q"]
        tt = [(x, y) for r, x, y in t["roles"] if r == "T"]
        if len(det_qb) and q:
            tot += min(np.linalg.norm(det_qb - np.array(q[0]), axis=1)); n += 1
        if len(det_back) and tt:
            tot += min(np.linalg.norm(det_back - np.array(tt[0]), axis=1)); n += 1
        if len(det_recv) and rr:
            T = np.array(rr, dtype=float)
            cost = np.linalg.norm(det_recv[:, None, :] - T[None, :, :], axis=2)
            r, c = linear_sum_assignment(cost)
            tot += cost[r, c].sum(); n += len(r)
        return tot / n if n else 1e9

    best = (None, 1e9)
    for t in role_templates:
        for s in (1.0, -1.0):  # mirror
            mq = qb.copy();   mq[:, 0] *= s if len(mq) else 1
            mb = back.copy(); mb[:, 0] *= s if len(mb) else 1
            mr = recv.copy(); mr[:, 0] *= s if len(mr) else 1
            d = score(mq, mb, mr, t)
            if d < best[1]:
                best = (t["name"], d)
    return best[0]


# ---------- run ----------
def main():
    lab = labels()
    role_templates = load_role_templates()
    old_templates = tm.load_templates()
    clips = sorted(glob.glob(os.path.join(CACHE, FOLDER, "positions", "*_position.json")))

    variants = {"V0_old": [], "V1_realyd_center": [], "V2_realyd_feet": [], "V3_roleanchored_feet": []}
    n_scored = 0

    for p in clips:
        clip = os.path.basename(p).replace("_position.json", "")
        m = re.search(r"Clip (\d+)", clip)
        if not m: continue
        actual = lab(int(m.group(1)))
        if not actual: continue

        snap_p = os.path.join(CACHE, FOLDER, "snap_detection", f"{clip}_snap_detection.json")
        corr_p = os.path.join(CACHE, FOLDER, "correspondence", f"{clip}_correspondence.json")
        if not (os.path.exists(snap_p) and os.path.exists(corr_p)):
            continue
        pdata = json.load(open(p))
        snaps = json.load(open(snap_p)).get("snaps") or []
        if not snaps: continue
        snap_frame = snaps[0].get("frame")
        corr = json.load(open(corr_p)).get("frame_correspondences", {})
        H = build_H(corr, snap_frame)
        if H is None: continue
        cleaned = tm._aggregate_tracks(pdata, snap_frame)
        if not cleaned: continue

        # V0: old matcher
        old = tm.recognize_from_cache(clip, FOLDER, CACHE, templates=old_templates)
        v0 = norm_pred(old.get("formation"))

        framed_c = real_yard_frame(projected_points(cleaned, H, use_feet=False))
        framed_f = real_yard_frame(projected_points(cleaned, H, use_feet=True))
        if framed_c is None or framed_f is None:
            continue

        v1 = norm_pred(match_roleagnostic(framed_c, role_templates))
        v2 = norm_pred(match_roleagnostic(framed_f, role_templates))
        v3 = norm_pred(match_roleanchored(framed_f, role_templates))

        n_scored += 1
        for key, pred in (("V0_old", v0), ("V1_realyd_center", v1),
                          ("V2_realyd_feet", v2), ("V3_roleanchored_feet", v3)):
            variants[key].append((actual, pred))

    print(f"\nScored on {n_scored} clips (processed + labeled + geometry-readable)\n")
    print(f"{'VARIANT':24s} {'EXACT':>8s} {'FAMILY':>8s}")
    print("-" * 44)
    for key, rows in variants.items():
        n = len(rows)
        if not n:
            print(f"{key:24s}    n=0"); continue
        ex = sum(p == a for a, p in rows) / n
        ba = sum(fam(p) == fam(a) for a, p in rows) / n
        print(f"{key:24s} {ex:7.1%} {ba:8.1%}")
    print(f"\n(baseline to beat: ~10.2% exact / 23.4% family from the full 244-clip run)")


if __name__ == "__main__":
    main()
