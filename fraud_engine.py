"""
FundTrace IQ — Python Fraud Detection Engine
Mirrors the JS FraudEngine class with all 5 detectors + scoring.
Uses NetworkX for graph analysis (centrality, path-finding, cycle detection).
"""

import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx


PROFILES: Dict[str, Dict[str, Any]] = {
    "student":    {"label": "Student",              "maxMonth": 30_000,      "maxSingle": 15_000},
    "salaried":   {"label": "Salaried Individual",  "maxMonth": 150_000,     "maxSingle": 100_000},
    "business":   {"label": "Business Entity",      "maxMonth": 5_000_000,   "maxSingle": 3_000_000},
    "nri":        {"label": "NRI Remittance",        "maxMonth": 2_000_000,   "maxSingle": 1_500_000},
    "investment": {"label": "Investment / AIF",      "maxMonth": 15_000_000,  "maxSingle": 10_000_000},
    "nbfc":       {"label": "NBFC / Lending",        "maxMonth": 3_000_000,   "maxSingle": 2_000_000},
    "offshore":   {"label": "Offshore Entity",       "maxMonth": 5_000_000,   "maxSingle": 4_000_000},
    "crypto":     {"label": "Crypto / OTC Desk",     "maxMonth": 1_000_000,   "maxSingle": 500_000},
}

DEFAULT_TH: Dict[str, Any] = {
    "circular": {
        "maxHops":        6,
        "timeWindowHrs":  72,
        "minReturnRatio": 0.80,
        "minEntry":       500_000,
    },
    "structuring": {
        "ctrlLimit":      100_000,
        "minTxnCount":    3,
        "timeWindowHrs":  24,
        "minAggregate":   250_000,
    },
    "dormant": {
        "inactDays":  180,
        "minSpike":   500_000,
        "sweepRatio": 0.80,
        "rapidHrs":   24,
    },
    "profile": {
        "devMultiple":  3.0,
        "critMultiple": 10.0,
        "windowDays":   14,
    },
    "mule": {
        "retWarn":    0.10,
        "retCrit":    0.05,
        "minInbound": 100_000,
        "rapidHrs":   6,
        "maxHoldHrs": 24,
        "minChain":   2,
        "maxChain":   8,
    },
    "scoring": {
        "w": {
            "circular":    35,
            "structuring": 20,
            "dormant":     20,
            "profile":     18,
            "mule":        30,
            "hub":         10,
            "velocity":    8,
        },
        "levels": {"low": 0, "medium": 20, "high": 45, "critical": 70},
    },
}


def _fmt(v: float) -> str:
    if v >= 10_000_000:
        return f"{v/10_000_000:.2f}Cr"
    if v >= 100_000:
        return f"{v/100_000:.2f}L"
    if v >= 1000:
        return f"{v/1000:.0f}K"
    return str(round(v))


def _hrs(a: datetime, b: datetime) -> float:
    return abs((b - a).total_seconds()) / 3600


class FraudEngine:
    REF_DATE = datetime(2024, 10, 28, 23, 59, 59)

    def __init__(self, accounts: List[dict], txns: List[dict], th: dict):
        self.raw_accounts = accounts
        self.raw_txns = txns
        self.th = th
        self.run_ms = 0

        # Parse dates
        self.txns: List[dict] = []
        for t in txns:
            tc = dict(t)
            try:
                tc["_d"] = datetime.fromisoformat(t["date"].replace("T", " "))
            except Exception:
                tc["_d"] = self.REF_DATE
            self.txns.append(tc)

        self.patterns: Dict[str, List[dict]] = {
            "circular": [], "structuring": [], "dormant": [],
            "profile": [], "mule": [],
        }
        self.txn_flags: Dict[str, Set[str]] = {}
        self.acc_scores: Dict[str, dict] = {}
        self.alerts: List[dict] = []
        self._mule: Dict[str, dict] = {}

        self._build_idx()
        self._build_nx_graph()

    # ── Index building ────────────────────────────────────────────────────────

    def _build_idx(self):
        self.acc_map = {a["id"]: a for a in self.raw_accounts}
        self.out_edges: Dict[str, List[dict]] = {a["id"]: [] for a in self.raw_accounts}
        self.in_edges:  Dict[str, List[dict]] = {a["id"]: [] for a in self.raw_accounts}
        self.all_edges: Dict[str, List[dict]] = {a["id"]: [] for a in self.raw_accounts}
        for t in self.txns:
            if t["from"] in self.out_edges:
                self.out_edges[t["from"]].append(t)
                self.all_edges[t["from"]].append(t)
            if t["to"] in self.in_edges:
                self.in_edges[t["to"]].append(t)
                self.all_edges[t["to"]].append(t)

    def _build_nx_graph(self):
        """Build a NetworkX DiGraph for centrality + path queries."""
        self.G = nx.DiGraph()
        for a in self.raw_accounts:
            self.G.add_node(a["id"])
        for t in self.txns:
            if self.G.has_edge(t["from"], t["to"]):
                self.G[t["from"]][t["to"]]["weight"] += t["amt"]
                self.G[t["from"]][t["to"]]["count"]  += 1
                self.G[t["from"]][t["to"]]["txns"].append(t["id"])
            else:
                self.G.add_edge(
                    t["from"], t["to"],
                    weight=t["amt"], count=1, txns=[t["id"]]
                )
        # Compute centrality scores (used to augment hub scoring)
        try:
            self.centrality = nx.betweenness_centrality(self.G, normalized=True)
        except Exception:
            self.centrality = {n: 0 for n in self.G.nodes}

    # ── DETECTOR 1: CIRCULAR TRANSACTIONS ────────────────────────────────────

    def detect_circular(self) -> List[dict]:
        results: List[dict] = []
        seen: Set[str] = set()
        th = self.th["circular"]

        def dfs(start_id, cur, path, txn_path, visited):
            for t in self.out_edges.get(cur, []):
                if t["to"] == start_id and len(path) >= 3:
                    chain = txn_path + [t]
                    ret_r = t["amt"] / chain[0]["amt"]
                    span  = _hrs(chain[0]["_d"], chain[-1]["_d"])
                    if ret_r < th["minReturnRatio"]:
                        continue
                    if span > th["timeWindowHrs"]:
                        continue
                    if chain[0]["amt"] < th["minEntry"]:
                        continue
                    ids   = list(path)
                    mini  = ids.index(min(ids))
                    key   = "→".join(ids[mini:] + ids[:mini])
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append({
                        "type":        "circular",
                        "accounts":    list(path) + [start_id],
                        "transactions": chain,
                        "returnRatio": round(ret_r * 1000) / 1000,
                        "tSpanHrs":    round(span * 10) / 10,
                        "entryAmt":    chain[0]["amt"],
                        "exitAmt":     t["amt"],
                        "hops":        len(path) - 1,
                        "explain": (
                            f"Loop closed: {'→'.join(path)}→{start_id}. "
                            f"Return ratio {round(ret_r*100)}% ({round((1-ret_r)*100)}% extracted = ₹{_fmt(chain[0]['amt']-t['amt'])}). "
                            f"Completed in {round(span*10)/10}h across {len(path)-1} hops. "
                            f"Threshold: ratio≥{round(th['minReturnRatio']*100)}%, span≤{th['timeWindowHrs']}h."
                        ),
                    })
                elif t["to"] not in visited and len(path) < th["maxHops"]:
                    visited.add(t["to"])
                    dfs(start_id, t["to"], path + [t["to"]], txn_path + [t], visited)
                    visited.discard(t["to"])

        for start_id in list(self.out_edges.keys()):
            dfs(start_id, start_id, [start_id], [], {start_id})
        return results

    # ── DETECTOR 2: STRUCTURING ───────────────────────────────────────────────

    def detect_structuring(self) -> List[dict]:
        results: List[dict] = []
        th = self.th["structuring"]
        for acc in self.raw_accounts:
            sub_t = sorted(
                [t for t in self.txns if t["from"] == acc["id"] and t["amt"] < th["ctrlLimit"]],
                key=lambda t: t["_d"]
            )
            if len(sub_t) < th["minTxnCount"]:
                continue
            for i in range(len(sub_t)):
                win = [t for t in sub_t if _hrs(sub_t[i]["_d"], t["_d"]) <= th["timeWindowHrs"]]
                if len(win) < th["minTxnCount"]:
                    continue
                total = sum(t["amt"] for t in win)
                if total < th["minAggregate"]:
                    continue
                key = ",".join(sorted(t["id"] for t in win))
                if any(r["key"] == key for r in results):
                    continue
                span = _hrs(win[0]["_d"], win[-1]["_d"]) if len(win) > 1 else 0
                recs = list({t["to"] for t in win})
                results.append({
                    "type":        "structuring",
                    "key":         key,
                    "sourceAcc":   acc["id"],
                    "transactions": win,
                    "txnCount":    len(win),
                    "total":       total,
                    "maxSingle":   max(t["amt"] for t in win),
                    "threshold":   th["ctrlLimit"],
                    "tSpanHrs":    round(span * 10) / 10,
                    "accounts":    [acc["id"]] + recs,
                    "explain": (
                        f"{len(win)} transactions in {round(span)}h, each below CTR limit ₹{_fmt(th['ctrlLimit'])}. "
                        f"Aggregate ₹{_fmt(total)} (largest single: ₹{_fmt(max(t['amt'] for t in win))}). "
                        f"Rate: {len(win)/max(1,span/24):.1f} txns/day. "
                        "Classic smurfing — splits to avoid regulatory reporting."
                    ),
                })
                break
        return results

    # ── DETECTOR 3: DORMANT ACCOUNT MISUSE ───────────────────────────────────

    def detect_dormant(self) -> List[dict]:
        results: List[dict] = []
        th = self.th["dormant"]
        for acc in self.raw_accounts:
            last_act  = datetime.fromisoformat(acc["lastActive"])
            inact_d   = (self.REF_DATE - last_act).days
            if inact_d < th["inactDays"]:
                continue
            incoming = [t for t in self.txns if t["to"] == acc["id"]]
            if not incoming:
                continue
            total_in  = sum(t["amt"] for t in incoming)
            max_in    = max(t["amt"] for t in incoming)
            if max_in < th["minSpike"]:
                continue
            outgoing  = [t for t in self.txns if t["from"] == acc["id"]]
            total_out = sum(t["amt"] for t in outgoing)
            sweep_r   = total_out / total_in if total_in else 0

            min_hold_hrs = None
            for in_t in incoming:
                for out_t in [o for o in outgoing if o["_d"] > in_t["_d"]]:
                    h = _hrs(in_t["_d"], out_t["_d"])
                    if min_hold_hrs is None or h < min_hold_hrs:
                        min_hold_hrs = h

            new_conns = list({t["to"] for t in outgoing})
            all_txns  = sorted(incoming + outgoing, key=lambda t: t["_d"])
            results.append({
                "type":         "dormant",
                "account":      acc["id"],
                "accounts":     [acc["id"]] + new_conns,
                "transactions": all_txns,
                "incoming":     incoming,
                "outgoing":     outgoing,
                "inactDays":    round(inact_d),
                "totalIn":      total_in,
                "totalOut":     total_out,
                "sweepRatio":   round(sweep_r * 100) / 100,
                "minHoldHrs":   round(min_hold_hrs * 10) / 10 if min_hold_hrs is not None else None,
                "newConns":     new_conns,
                "explain": (
                    f"Account dormant {round(inact_d)} days (threshold: {th['inactDays']}d). "
                    f"Received ₹{_fmt(total_in)}, forwarded {round(sweep_r*100)}% (₹{_fmt(total_out)})."
                    + (f" Fastest pass-through: {round(min_hold_hrs*10)/10}h." if min_hold_hrs is not None else "")
                    + f" {len(new_conns)} new outgoing connections after reactivation. "
                    f"Threshold: sweep≥{round(th['sweepRatio']*100)}%, spike≥₹{_fmt(th['minSpike'])}."
                ),
            })
        return results

    # ── DETECTOR 4: PROFILE MISMATCH ─────────────────────────────────────────

    def detect_profile(self) -> List[dict]:
        results: List[dict] = []
        th = self.th["profile"]
        wm = th["windowDays"] / 30
        for acc in self.raw_accounts:
            pdef = PROFILES.get(acc.get("cat", ""))
            if not pdef:
                continue
            txns = self.all_edges.get(acc["id"], [])
            if not txns:
                continue
            total_vol  = sum(t["amt"] for t in txns)
            max_single = max(t["amt"] for t in txns)
            obs_mon    = total_vol / wm
            dev_score  = obs_mon / pdef["maxMonth"]
            sin_dev    = max_single / pdef["maxSingle"]
            if dev_score < th["devMultiple"] and sin_dev < th["devMultiple"]:
                continue
            results.append({
                "type":        "profile",
                "account":     acc["id"],
                "accounts":    [acc["id"]],
                "category":    acc.get("cat"),
                "label":       pdef["label"],
                "expMonth":    pdef["maxMonth"],
                "expSingle":   pdef["maxSingle"],
                "obsMon":      round(obs_mon),
                "devScore":    round(dev_score * 10) / 10,
                "sinDev":      round(sin_dev * 10) / 10,
                "maxSingle":   max_single,
                "totalVol":    total_vol,
                "transactions": txns,
                "explain": (
                    f"{pdef['label']} — observed ₹{_fmt(round(obs_mon))}/month vs expected ceiling "
                    f"₹{_fmt(pdef['maxMonth'])}/month ({round(dev_score*10)/10}× deviation). "
                    f"Largest single: ₹{_fmt(max_single)} vs ₹{_fmt(pdef['maxSingle'])} limit ({round(sin_dev*10)/10}×). "
                    f"Declared annual: ₹{_fmt(acc.get('declAnn', 0))}. Total observed: ₹{_fmt(total_vol)}."
                ),
            })
        return results

    # ── DETECTOR 5: MULE ACCOUNTS ─────────────────────────────────────────────

    def detect_mule(self) -> List[dict]:
        results: List[dict] = []
        seen:    Set[str] = set()
        th = self.th["mule"]

        # Step 1+2: Per-account retention + hold-time
        self._mule = {}
        for acc in self.raw_accounts:
            incoming  = [t for t in self.txns if t["to"]   == acc["id"]]
            outgoing  = [t for t in self.txns if t["from"] == acc["id"]]
            total_in  = sum(t["amt"] for t in incoming)
            total_out = sum(t["amt"] for t in outgoing)
            if total_in < th["minInbound"]:
                self._mule[acc["id"]] = {
                    "retention": 1, "minHoldHrs": None,
                    "totalIn": total_in, "totalOut": total_out,
                    "muleScore": 0, "isMule": False,
                    "isCrit": False, "isRapid": False, "speedBonus": 0,
                }
                continue
            retention = max(0, (total_in - total_out) / total_in) if total_in else 1
            min_hold_hrs = None
            for in_t in incoming:
                for out_t in [o for o in outgoing if o["_d"] > in_t["_d"]]:
                    h = _hrs(in_t["_d"], out_t["_d"])
                    if min_hold_hrs is None or h < min_hold_hrs:
                        min_hold_hrs = h
            is_mule    = retention < th["retWarn"] and total_out > 0
            speed_bonus = 0
            if min_hold_hrs is not None:
                speed_bonus = 1.0 if min_hold_hrs < th["rapidHrs"] else (0.5 if min_hold_hrs < th["maxHoldHrs"] else 0)
            thru_put  = min(1, total_out / total_in) if total_in > 0 else 0
            mule_score = min(100, round((1 - retention) * 55 + speed_bonus * 30 + thru_put * 15)) if is_mule else 0
            self._mule[acc["id"]] = {
                "retention":   round(retention * 1000) / 1000,
                "minHoldHrs":  round(min_hold_hrs * 10) / 10 if min_hold_hrs is not None else None,
                "totalIn":     total_in,
                "totalOut":    total_out,
                "muleScore":   mule_score,
                "isMule":      is_mule,
                "speedBonus":  speed_bonus,
                "isCrit":      retention < th["retCrit"],
                "isRapid":     min_hold_hrs is not None and min_hold_hrs < th["maxHoldHrs"],
            }

        # Step 3: Chain detection
        mule_ids = [a["id"] for a in self.raw_accounts if self._mule.get(a["id"], {}).get("isMule")]

        def dfs_chain(path, txn_path, visited):
            if len(path) > th["maxChain"]:
                return
            if len(path) >= th["minChain"]:
                key = "→".join(path)
                if key not in seen:
                    seen.add(key)
                    retentions  = {id_: self._mule[id_]["retention"] for id_ in path if id_ in self._mule}
                    hold_times  = {id_: self._mule[id_]["minHoldHrs"] for id_ in path if id_ in self._mule}
                    avg_ret     = sum(self._mule.get(id_, {}).get("retention", 1) for id_ in path) / len(path)
                    fastest     = None
                    for id_ in path:
                        h = self._mule.get(id_, {}).get("minHoldHrs")
                        if h is not None and (fastest is None or h < fastest):
                            fastest = h
                    chain_score = round(sum(self._mule.get(id_, {}).get("muleScore", 0) for id_ in path) / len(path))
                    total_flow  = txn_path[0]["amt"] if txn_path else 0
                    # Deduplicate txn_path by id
                    seen_tids: Dict[str, dict] = {}
                    for t in txn_path:
                        seen_tids[t["id"]] = t
                    results.append({
                        "type":          "mule",
                        "accounts":      list(path),
                        "transactions":  list(seen_tids.values()),
                        "retentions":    retentions,
                        "holdTimes":     hold_times,
                        "avgRetention":  round(avg_ret * 1000) / 1000,
                        "fastestHoldHrs": round(fastest * 10) / 10 if fastest is not None else None,
                        "chainLen":      len(path),
                        "chainScore":    chain_score,
                        "totalFlow":     total_flow,
                        "hasCrit":       any(self._mule.get(id_, {}).get("isCrit") for id_ in path),
                        "hasRapid":      any(self._mule.get(id_, {}).get("isRapid") for id_ in path),
                        "explain": (
                            f"{len(path)}-node mule chain: {'→'.join(path)}. "
                            f"Avg retention {round(avg_ret*100)}% — forwards {round((1-avg_ret)*100)}% of funds. "
                            + (f"Fastest pass-through: {round(fastest*10)/10}h. " if fastest is not None else "")
                            + f"Total flow: ₹{_fmt(total_flow)}. "
                            f"Chain mule score: {chain_score}/100. "
                            f"Threshold: retention<{round(th['retWarn']*100)}% = suspicious, <{round(th['retCrit']*100)}% = high-risk."
                        ),
                    })
            cur = path[-1]
            for t in self.out_edges.get(cur, []):
                if t["to"] not in visited and self._mule.get(t["to"], {}).get("isMule"):
                    visited.add(t["to"])
                    dfs_chain(path + [t["to"]], txn_path + [t], visited)
                    visited.discard(t["to"])

        for start_id in mule_ids:
            dfs_chain([start_id], [], {start_id})

        return sorted(results, key=lambda r: -r["chainScore"])

    # ── TXN FLAG MAP ──────────────────────────────────────────────────────────

    def _build_txn_flags(self):
        self.txn_flags = {}
        def tag(tid, typ):
            if tid not in self.txn_flags:
                self.txn_flags[tid] = set()
            self.txn_flags[tid].add(typ)
        for p in self.patterns["circular"]:
            for t in p["transactions"]: tag(t["id"], "circular")
        for p in self.patterns["structuring"]:
            for t in p["transactions"]: tag(t["id"], "structuring")
        for p in self.patterns["dormant"]:
            for t in p["transactions"]: tag(t["id"], "dormant")
        for p in self.patterns["profile"]:
            for t in p["transactions"]: tag(t["id"], "profile")
        for p in self.patterns["mule"]:
            for t in p["transactions"]: tag(t["id"], "mule")

    # ── UNIFIED SCORING ───────────────────────────────────────────────────────

    def score_accounts(self):
        w  = self.th["scoring"]["w"]
        lv = self.th["scoring"]["levels"]
        for acc in self.raw_accounts:
            id_    = acc["id"]
            score  = 0
            flags: Set[str] = set()
            bd:    Dict[str, dict] = {}
            explains: List[dict] = []

            # Circular
            circ_pats = [p for p in self.patterns["circular"] if id_ in p["accounts"]]
            if circ_pats:
                best_ret = max(p["returnRatio"] for p in circ_pats)
                pts = min(w["circular"], w["circular"] * (0.6 + len(circ_pats)*0.3) * (0.5 + best_ret*0.5))
                score += pts
                flags.add("circular")
                bd["circular"] = {"pts": round(pts), "max": w["circular"],
                                   "detail": f"{len(circ_pats)} cycle(s), best return {round(best_ret*100)}%"}
                explains.append({"rule": "Circular Fund Loop", "icon": "🔄",
                                  "sev": "crit" if best_ret >= 0.92 else "high",
                                  "body": circ_pats[0]["explain"]})

            # Structuring
            str_pats = [p for p in self.patterns["structuring"] if id_ in p["accounts"]]
            if str_pats:
                is_src = any(p["sourceAcc"] == id_ for p in str_pats)
                pts = w["structuring"] if is_src else round(w["structuring"] * 0.5)
                score += pts
                flags.add("structuring")
                bd["structuring"] = {"pts": round(pts), "max": w["structuring"],
                                      "detail": "Source account" if is_src else "Recipient"}
                explains.append({"rule": "Structuring / Smurfing", "icon": "📉",
                                  "sev": "high", "body": str_pats[0]["explain"]})

            # Dormant
            dor_pats = [p for p in self.patterns["dormant"] if id_ in p["accounts"]]
            if dor_pats:
                has_sw = any(p["sweepRatio"] >= self.th["dormant"]["sweepRatio"] for p in dor_pats)
                pts = min(w["dormant"] * 1.4, w["dormant"] * (1.2 if has_sw else 0.9))
                score += pts
                flags.add("dormant")
                is_ctr = any(p["account"] == id_ for p in dor_pats)
                bd["dormant"] = {
                    "pts": round(pts), "max": w["dormant"],
                    "detail": f"Dormant {dor_pats[0]['inactDays']}d, sweep {round(dor_pats[0]['sweepRatio']*100)}%" if is_ctr else "Connected",
                }
                own = next((p for p in dor_pats if p["account"] == id_), dor_pats[0])
                explains.append({"rule": "Dormant Account Misuse", "icon": "💤",
                                  "sev": "crit" if has_sw else "high",
                                  "body": own["explain"]})

            # Profile mismatch
            prof_pats = [p for p in self.patterns["profile"] if id_ in p["accounts"]]
            if prof_pats:
                max_dev = max(p["devScore"] for p in prof_pats)
                scale   = min(2, max_dev / self.th["profile"]["devMultiple"])
                pts = min(w["profile"] * 2, w["profile"] * scale)
                score += pts
                flags.add("profile_mismatch")
                bd["profile"] = {"pts": round(pts), "max": w["profile"],
                                  "detail": f"{max_dev}× above {prof_pats[0]['label']} ceiling"}
                explains.append({"rule": "Profile–Behaviour Mismatch", "icon": "⚡",
                                  "sev": "crit" if max_dev >= self.th["profile"]["critMultiple"] else "high",
                                  "body": prof_pats[0]["explain"]})

            # Mule
            ms = self._mule.get(id_)
            if ms and ms["isMule"]:
                chain_pat = next((p for p in self.patterns["mule"] if id_ in p["accounts"]), None)
                speed_pts = ms["speedBonus"] * 12
                pts = min(w["mule"] * 1.5, w["mule"] * (1 - ms["retention"]) + speed_pts)
                score += pts
                flags.add("mule")
                bd["mule"] = {"pts": round(pts), "max": w["mule"],
                               "detail": f"Retention {round(ms['retention']*100)}%, hold {ms['minHoldHrs']}h"}
                body = (
                    f"Retention score {ms['retention']} ({round(ms['retention']*100)}%) — "
                    f"forwarded {round((1-ms['retention'])*100)}% of ₹{_fmt(ms['totalIn'])} received."
                    + (f" Min hold time: {ms['minHoldHrs']}h (threshold: <{self.th['mule']['maxHoldHrs']}h = rapid)." if ms['minHoldHrs'] is not None else "")
                    + (f" Part of {chain_pat['chainLen']}-node mule chain (chain score {chain_pat['chainScore']}/100)." if chain_pat else "")
                    + f" Threshold: retention<{round(self.th['mule']['retWarn']*100)}% suspicious, <{round(self.th['mule']['retCrit']*100)}% high-risk."
                )
                explains.append({"rule": "Mule Account", "icon": "🪣",
                                  "sev": "crit" if ms["isCrit"] else ("high" if ms["isRapid"] else "med"),
                                  "body": body})

            # Network hub (augmented with NetworkX betweenness centrality)
            my_txns = self.all_edges.get(id_, [])
            cparts  = {t["from"] for t in my_txns} | {t["to"] for t in my_txns}
            cparts.discard(id_)
            nx_cen = self.centrality.get(id_, 0)
            if len(cparts) >= 4 or nx_cen > 0.05:
                hub_factor = max(len(cparts) * 1.8, nx_cen * 200)
                pts = min(w["hub"], hub_factor)
                score += pts
                bd["hub"] = {"pts": round(pts), "max": w["hub"],
                              "detail": f"{len(cparts)} counterparties (centrality {nx_cen:.3f})"}

            # Velocity
            out_t = sorted([t for t in my_txns if t["from"] == id_], key=lambda t: t["_d"])
            if len(out_t) >= 4:
                span = _hrs(out_t[0]["_d"], out_t[-1]["_d"]) or 1
                vel  = len(out_t) / span
                if vel > 0.04:
                    pts = min(w["velocity"], vel * 30)
                    score += pts
                    bd["velocity"] = {"pts": round(pts), "max": w["velocity"],
                                       "detail": f"{vel*24:.1f} txns/day"}

            # Final
            score = min(100, round(score))
            risk  = ("critical" if score >= lv["critical"] else
                     "high"     if score >= lv["high"]     else
                     "medium"   if score >= lv["medium"]   else "low")
            last_act = datetime.fromisoformat(acc["lastActive"])
            inact_d  = (self.REF_DATE - last_act).days
            status   = "dormant" if inact_d >= self.th["dormant"]["inactDays"] else "active"
            is_mule  = bool(ms and ms["isMule"])
            self.acc_scores[id_] = {
                "score": score, "risk": risk, "flags": list(flags),
                "breakdown": bd, "explains": explains,
                "status": status, "isMule": is_mule, "muleStats": ms,
            }

    # ── ALERT GENERATION ─────────────────────────────────────────────────────

    def generate_alerts(self):
        alerts: List[dict] = []
        n = 0

        # FIXED FORMAT: Uses a platform-independent way to skip leading zeroes
        def fmt_d(d): return f"{d.day} {d.strftime('%b')}" if d else "—"

        for p in self.patterns["circular"]:
            is_crit = p["returnRatio"] >= 0.92 and p["tSpanHrs"] <= 36
            n += 1
            alerts.append({
                "id": f"AL{n}", "type": "circular",
                "sev": "crit" if is_crit else "high",
                "title": f"Circular Flow — {p['hops']}-Hop Round-Trip",
                "body": (
                    f"₹{_fmt(p['entryAmt'])} completed a {p['hops']}-hop loop, returning "
                    f"{round(p['returnRatio']*100)}% (₹{_fmt(p['exitAmt'])}) to origin in {p['tSpanHrs']}h. "
                    f"₹{_fmt(p['entryAmt']-p['exitAmt'])} extracted as fees."
                ),
                "accounts":  p["accounts"],
                "txns":      [t["id"] for t in p["transactions"]],
                "dateRange": f"{fmt_d(p['transactions'][0]['_d'])} – {fmt_d(p['transactions'][-1]['_d'])}",
                "metrics": {
                    "Return ratio": f"{round(p['returnRatio']*100)}%",
                    "Time span":    f"{p['tSpanHrs']}h",
                    "Hops":         p["hops"],
                    "Extracted":    f"₹{_fmt(p['entryAmt']-p['exitAmt'])}",
                },
                "explain": p["explain"],
            })

        for p in self.patterns["structuring"]:
            n += 1
            alerts.append({
                "id": f"AL{n}", "type": "structuring",
                "sev": "high" if p["txnCount"] >= 5 else "med",
                "title": f"Structuring — {p['txnCount']} Sub-Threshold Txns",
                "body":  p["explain"],
                "accounts":  p["accounts"],
                "txns":      [t["id"] for t in p["transactions"]],
                "dateRange": f"{fmt_d(p['transactions'][0]['_d'])} – {fmt_d(p['transactions'][-1]['_d'])}",
                "metrics": {
                    "Count":     p["txnCount"],
                    "Threshold": f"₹{_fmt(p['threshold'])}",
                    "Aggregate": f"₹{_fmt(p['total'])}",
                    "Span":      f"{p['tSpanHrs']}h",
                },
                "explain": p["explain"],
            })

        for p in self.patterns["dormant"]:
            nm = self.acc_map.get(p["account"], {}).get("name", p["account"])
            is_crit = (p["sweepRatio"] >= self.th["dormant"]["sweepRatio"]
                       and (p["minHoldHrs"] or 999) < self.th["dormant"]["rapidHrs"])
            n += 1
            alerts.append({
                "id": f"AL{n}", "type": "dormant",
                "sev": "crit" if is_crit else "high",
                "title": f"Dormant Activation — {nm}",
                "body":  p["explain"],
                "accounts":  p["accounts"],
                "txns":      [t["id"] for t in p["transactions"]],
                "dateRange": fmt_d(p["incoming"][0]["_d"]) if p["incoming"] else "—",
                "metrics": {
                    "Inactive": f"{p['inactDays']}d",
                    "Total in": f"₹{_fmt(p['totalIn'])}",
                    "Sweep":    f"{round(p['sweepRatio']*100)}%",
                    "Hold":     f"{p['minHoldHrs']}h" if p["minHoldHrs"] else "—",
                },
                "explain": p["explain"],
            })

        for p in self.patterns["profile"]:
            nm = self.acc_map.get(p["account"], {}).get("name", p["account"])
            n += 1
            alerts.append({
                "id": f"AL{n}", "type": "profile",
                "sev": "crit" if p["devScore"] >= self.th["profile"]["critMultiple"] else "high",
                "title": f"Profile Mismatch — {nm}",
                "body":  p["explain"],
                "accounts":  p["accounts"],
                "txns":      [t["id"] for t in p["transactions"]],
                "dateRange": "14–28 Oct 2024",
                "metrics": {
                    "Category":  p["label"],
                    "Deviation": f"{p['devScore']}×",
                    "Obs/mo":    f"₹{_fmt(p['obsMon'])}",
                    "Exp/mo":    f"₹{_fmt(p['expMonth'])}",
                },
                "explain": p["explain"],
            })

        for p in self.patterns["mule"]:
            is_crit = p["hasCrit"] and p["chainLen"] >= 3
            n += 1
            alerts.append({
                "id": f"AL{n}", "type": "mule",
                "sev": "crit" if is_crit else ("high" if p["hasRapid"] else "med"),
                "title": f"Mule Network — {p['chainLen']}-Node Pass-Through Chain",
                "body":  p["explain"],
                "accounts":  p["accounts"],
                "txns":      [t["id"] for t in p["transactions"]],
                "dateRange": fmt_d(p["transactions"][0]["_d"]) if p["transactions"] else "—",
                "metrics": {
                    "Chain len":      p["chainLen"],
                    "Avg retention":  f"{round(p['avgRetention']*100)}%",
                    "Chain score":    f"{p['chainScore']}/100",
                    "Fastest hold":   f"{p['fastestHoldHrs']}h" if p["fastestHoldHrs"] else "—",
                },
                "explain": p["explain"],
            })

        sev_ord = {"crit": 0, "high": 1, "med": 2}
        self.alerts = sorted(alerts, key=lambda a: sev_ord.get(a["sev"], 3))

    # ── MAIN PIPELINE ─────────────────────────────────────────────────────────

    def analyze(self) -> "FraudEngine":
        t0 = time.time()
        self.patterns["circular"]    = self.detect_circular()
        self.patterns["structuring"] = self.detect_structuring()
        self.patterns["dormant"]     = self.detect_dormant()
        self.patterns["profile"]     = self.detect_profile()
        self.patterns["mule"]        = self.detect_mule()
        self._build_txn_flags()
        self.score_accounts()
        self.generate_alerts()
        self.run_ms = round((time.time() - t0) * 1000)
        return self

    # ── PUBLIC HELPERS ────────────────────────────────────────────────────────

    def enriched_accounts(self) -> List[dict]:
        default = {"score": 0, "risk": "low", "flags": [],
                   "breakdown": {}, "explains": [], "status": "active",
                   "isMule": False, "muleStats": None}
        return [{**a, **(self.acc_scores.get(a["id"], default))} for a in self.raw_accounts]

    def counts(self) -> dict:
        c = {k: len(v) for k, v in self.patterns.items()}
        c["total"] = sum(c.values())
        return c

    def edge_flag(self, txn_id: str) -> Optional[str]:
        s = self.txn_flags.get(txn_id)
        if not s:
            return None
        for t in ["circular", "mule", "dormant", "structuring", "profile"]:
            if t in s:
                return t
        return None

    def serializable_txns(self) -> List[dict]:
        """Return txns without datetime objects."""
        out = []
        for t in self.txns:
            tc = {k: v for k, v in t.items() if k != "_d"}
            tc["flag"] = self.edge_flag(t["id"])
            out.append(tc)
        return out

    def serializable_patterns(self) -> dict:
        """Return patterns safe for JSON serialization."""
        def clean(p):
            pc = {}
            for k, v in p.items():
                if k in ("transactions", "incoming", "outgoing"):
                    pc[k] = [{kk: vv for kk, vv in t.items() if kk != "_d"} for t in v]
                else:
                    pc[k] = v
            return pc
        return {k: [clean(p) for p in v] for k, v in self.patterns.items()}

    def generate_str_report(self) -> str:
        """Generate the full Suspicious Transaction Report text."""
        from datetime import datetime as dt
        import random
        now = dt.now()
        c   = self.counts()
        enriched = self.enriched_accounts()
        txns_list = self.serializable_txns()
        total_vol = sum(t["amt"] for t in txns_list)
        flagged_vol = sum(t["amt"] for t in txns_list if t.get("flag"))
        crit_accs = [a for a in enriched if a["risk"] == "critical"]

        def fmt_inr(v):
            return f"{v:,.0f}"

        tl_map = {
            "circular":    "Circular / Round-Tripping",
            "structuring": "Structuring (Smurfing)",
            "dormant":     "Dormant Account Activation",
            "profile":     "Profile–Behaviour Mismatch",
            "mule":        "Mule Account Pass-Through",
        }
        sect_b = ""
        bi = 0
        for ptype, pats in self.patterns.items():
            for p in pats:
                bi += 1
                al = next((a for a in self.alerts if set(a["txns"]) == {t["id"] for t in p["transactions"]}), None)
                sev = "CRITICAL" if al and al["sev"] == "crit" else "HIGH"
                txn_ids = ", ".join(t["id"] for t in p["transactions"])
                metrics = ""
                if al:
                    metrics = ", ".join(f"{k}={v}" for k, v in al["metrics"].items())
                sect_b += (
                    f"\nB.{bi}  {tl_map.get(ptype, ptype)}  [{sev}]\n"
                    f"  Accounts   : {' → '.join(p['accounts'])}\n"
                    f"  Txns       : {txn_ids}\n"
                    f"  Explanation: {p['explain']}\n"
                    f"  Metrics    : {metrics}\n"
                )

        sect_c_lines = []
        for i, a in enumerate(crit_accs[:8]):
            sc = self.acc_scores.get(a["id"], {})
            ms = sc.get("muleStats") or {}
            mule_str = (
                f"YES — retention {round((ms.get('retention', 1))*100)}%, hold {ms.get('minHoldHrs')}h"
                if sc.get("isMule") else "No"
            )
            action = ("Account freeze + ED/FIU referral"
                      if sc.get("score", 0) >= 80 else
                      "Immediate EDD + enhanced monitoring")
            sect_c_lines.append(
                f"{i+1}. {a['name']} ({a['id']}) — {a['type']}, {a['branch']}\n"
                f"   Score  : {sc.get('score', 0)}/100  |  Risk: {sc.get('risk', '').upper()}\n"
                f"   Flags  : {', '.join(sc.get('flags', [])) or 'None'}\n"
                f"   Mule   : {mule_str}\n"
                f"   Action : {action}"
            )

        flagged_accs = [a for a in enriched if a.get("flags")]
        high_risk_cnt = len([a for a in enriched if a["risk"] in ("critical", "high")])

        th = self.th
        report = (
            "SUSPICIOUS TRANSACTION REPORT (STR)\n"
            "Financial Intelligence Unit — India (FIU-IND)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"REFERENCE     : STR-{now.year}-{random.randint(10000,99999)}\n"
            f"DATE FILED    : {now.strftime('%d %b %Y')}\n"
            f"ENTITY        : [Reporting Institution] — AML / Compliance\n"
            f"CASE          : FIU-2024-3847 / Operation Phantom Circuit\n"
            f"ENGINE        : FundTrace IQ v4.0 — Python Backend (NetworkX + Flask)\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "SECTION A: PIPELINE SUMMARY\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Analysis period    : 14 October 2024 – 28 October 2024\n"
            f"Engine run time    : {self.run_ms}ms\n"
            f"Detection pipeline : Ingest → Circular → Structuring → Dormant →\n"
            f"                     Profile → Mule → Score → Classify → Alert\n\n"
            f"Accounts analysed  : {len(enriched)}\n"
            f"Transactions       : {len(txns_list)}\n"
            f"Total value        : ₹{fmt_inr(total_vol)}\n"
            f"Flagged value      : ₹{fmt_inr(flagged_vol)} ({round(flagged_vol/total_vol*100) if total_vol else 0}% of total)\n"
            f"Flagged accounts   : {len(flagged_accs)}\n"
            f"Critical accounts  : {len(crit_accs)}\n\n"
            f"Patterns detected:\n"
            f"  Circular txns       : {c['circular']}\n"
            f"  Structuring         : {c['structuring']}\n"
            f"  Dormant activations : {c['dormant']}\n"
            f"  Profile mismatches  : {c['profile']}\n"
            f"  Mule chains         : {c['mule']}\n"
            f"  ─────────────────────\n"
            f"  TOTAL               : {c['total']} detected patterns\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "SECTION B: DETECTED PATTERNS + ENGINE REASONING\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{sect_b}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"SECTION C: HIGH-RISK ACCOUNTS (top {min(8, len(crit_accs))})\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{chr(10).join(sect_c_lines)}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "SECTION D: RECOMMENDED ACTIONS\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"□ File this STR with FIU-IND via FINnet within 7 days\n"
            f"□ EDD for all {high_risk_cnt} high/critical accounts\n"
            f"□ Freeze accounts in confirmed circular loops\n"
            f"□ CTR retrospective for structuring series\n"
            f"□ Refer identified mule accounts for further investigation\n"
            f"□ Refer offshore accounts to RBI/FEMA\n"
            f"□ Preserve digital records (PMLA Sec 12, 10 years)\n"
            f"□ Legal hold — tipping-off prohibition strictly applies\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "SECTION E: EXPLAINABILITY SUMMARY\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Mule Thresholds Used:\n"
            f"  Suspicious: retention < {round(th['mule']['retWarn']*100)}%\n"
            f"  High-risk:  retention < {round(th['mule']['retCrit']*100)}%\n"
            f"  Rapid:      hold time < {th['mule']['maxHoldHrs']}h\n\n"
            f"Circular Thresholds:\n"
            f"  Min return ratio: {round(th['circular']['minReturnRatio']*100)}%\n"
            f"  Time window:      {th['circular']['timeWindowHrs']}h\n"
            f"  Min entry:        ₹{fmt_inr(th['circular']['minEntry'])}\n\n"
            f"All detections are algorithm-generated via Python + NetworkX.\n"
            f"NetworkX betweenness centrality augments hub detection.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "DECLARATION: Filed in good faith under PMLA 2002.\n"
            "CONFIDENTIAL — TIPPING-OFF PROHIBITED S.8A PMLA\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        return report