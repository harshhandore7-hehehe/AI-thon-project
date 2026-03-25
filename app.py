"""
FundTrace IQ — Flask Backend
Serves REST API endpoints for the fraud detection engine.
All detection runs server-side (Python + NetworkX).
"""

import copy
import json
import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from fraud_engine import FraudEngine, DEFAULT_TH, PROFILES

app = Flask(__name__, static_folder="static")
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
#  IN-MEMORY DATA STORE
# ─────────────────────────────────────────────────────────────────────────────

RAW_ACCOUNTS = [
    {"id":"A01","name":"Ahmad Hassan",         "type":"Personal Savings",  "branch":"Bandra West",   "bal":12400,  "cat":"salaried",   "declAnn":320000,    "lastActive":"2024-08-10"},
    {"id":"A02","name":"Priya Corp Ltd",        "type":"Current Account",   "branch":"Fort",          "bal":284000, "cat":"business",   "declAnn":8000000,   "lastActive":"2024-10-01"},
    {"id":"A03","name":"Nexus Holdings",        "type":"Business Account",  "branch":"BKC",           "bal":3400,   "cat":"business",   "declAnn":10000000,  "lastActive":"2024-09-20"},
    {"id":"A04","name":"Rapid Finance Ltd",     "type":"NBFC Account",      "branch":"Santacruz",     "bal":8900,   "cat":"nbfc",       "declAnn":5000000,   "lastActive":"2024-09-28"},
    {"id":"A05","name":"Pacific Trade Co.",     "type":"Current Account",   "branch":"Nariman Point", "bal":890000, "cat":"business",   "declAnn":20000000,  "lastActive":"2024-10-05"},
    {"id":"A06","name":"Green Ventures LLC",    "type":"Business Account",  "branch":"Andheri East",  "bal":1200,   "cat":"business",   "declAnn":4000000,   "lastActive":"2024-09-15"},
    {"id":"A07","name":"Global Assets Corp",    "type":"Business Account",  "branch":"Fort",          "bal":234000, "cat":"business",   "declAnn":30000000,  "lastActive":"2024-10-03"},
    {"id":"A08","name":"Blue Sky Invest.",      "type":"Investment Acct",   "branch":"Parel",         "bal":420000, "cat":"investment", "declAnn":40000000,  "lastActive":"2024-10-02"},
    {"id":"A09","name":"Vikram Offshore LLC",   "type":"Offshore/FEMA",     "branch":"GIFT City",     "bal":2100,   "cat":"offshore",   "declAnn":50000000,  "lastActive":"2024-09-05"},
    {"id":"A10","name":"Samira Patel",          "type":"Savings Account",   "branch":"Borivali",      "bal":67000,  "cat":"salaried",   "declAnn":850000,    "lastActive":"2024-10-10"},
    {"id":"A11","name":"Chen Wei Enterprises",  "type":"Current Account",   "branch":"Powai",         "bal":156000, "cat":"business",   "declAnn":6000000,   "lastActive":"2024-10-08"},
    {"id":"A12","name":"Rajan Mehta",           "type":"Personal Savings",  "branch":"Dadar",         "bal":38000,  "cat":"salaried",   "declAnn":600000,    "lastActive":"2024-10-01"},
    {"id":"A13","name":"Zenith Securities",     "type":"Investment Acct",   "branch":"Parel",         "bal":520000, "cat":"investment", "declAnn":60000000,  "lastActive":"2024-10-05"},
    {"id":"A14","name":"M. Al-Farsi & Co.",     "type":"NRE Account",       "branch":"Worli",         "bal":45000,  "cat":"nri",        "declAnn":1200000,   "lastActive":"2024-01-05"},
    {"id":"A15","name":"Fatima Al-Rashid",      "type":"NRE Account",       "branch":"Bandra West",   "bal":4100,   "cat":"nri",        "declAnn":1500000,   "lastActive":"2024-01-20"},
    {"id":"A16","name":"Karim Exports LLC",     "type":"NRE Account",       "branch":"Worli",         "bal":3100,   "cat":"nri",        "declAnn":2000000,   "lastActive":"2024-02-28"},
    {"id":"A17","name":"Falcon Capital Ltd",    "type":"Business Account",  "branch":"Nariman Point", "bal":4300,   "cat":"investment", "declAnn":50000000,  "lastActive":"2024-09-22"},
    {"id":"A18","name":"Horizon Finserv",       "type":"NBFC Account",      "branch":"Andheri West",  "bal":8200,   "cat":"nbfc",       "declAnn":3000000,   "lastActive":"2024-09-25"},
    {"id":"A19","name":"Lakshmi Finance",       "type":"NBFC Account",      "branch":"Santacruz",     "bal":95000,  "cat":"nbfc",       "declAnn":4000000,   "lastActive":"2024-10-02"},
    {"id":"A20","name":"Pinnacle Crypto Desk",  "type":"Business Account",  "branch":"BKC",           "bal":1800,   "cat":"crypto",     "declAnn":10000000,  "lastActive":"2024-09-30"},
    {"id":"A21","name":"Caspian Trade Corp",    "type":"Current Account",   "branch":"Fort",          "bal":2900,   "cat":"business",   "declAnn":20000000,  "lastActive":"2024-09-18"},
    {"id":"A22","name":"Silverline Exports",    "type":"Current Account",   "branch":"Colaba",        "bal":5600,   "cat":"business",   "declAnn":15000000,  "lastActive":"2024-09-10"},
    {"id":"A23","name":"Indo-Gulf Holdings",    "type":"Business Account",  "branch":"GIFT City",     "bal":6700,   "cat":"offshore",   "declAnn":50000000,  "lastActive":"2024-09-12"},
    {"id":"A24","name":"Meridian Asset Mgmt",   "type":"Investment Acct",   "branch":"Nariman Point", "bal":840000, "cat":"investment", "declAnn":200000000, "lastActive":"2024-10-04"},
    {"id":"A25","name":"Deepika Nair",          "type":"Savings Account",   "branch":"Goregaon",      "bal":92000,  "cat":"salaried",   "declAnn":950000,    "lastActive":"2024-10-12"},
    {"id":"A26","name":"Suresh Iyer",           "type":"NRE Account",       "branch":"Vile Parle",    "bal":175000, "cat":"nri",        "declAnn":1800000,   "lastActive":"2024-02-14"},
    {"id":"A27","name":"Orbit Infra Pvt Ltd",   "type":"Business Account",  "branch":"Lower Parel",   "bal":310000, "cat":"business",   "declAnn":80000000,  "lastActive":"2024-10-04"},
    {"id":"A28","name":"Zara Textiles",         "type":"Current Account",   "branch":"Dharavi",       "bal":56000,  "cat":"business",   "declAnn":8000000,   "lastActive":"2024-10-11"},
    {"id":"A29","name":"Pradeep Khatri",        "type":"Savings Account",   "branch":"Malad",         "bal":44000,  "cat":"salaried",   "declAnn":700000,    "lastActive":"2024-10-07"},
    {"id":"A30","name":"Gamma Logistics Ltd",   "type":"Current Account",   "branch":"Thane",         "bal":125000, "cat":"business",   "declAnn":20000000,  "lastActive":"2024-10-10"},
]

BASE_TXNS = [
    {"id":"T01","from":"A01","to":"A02","amt":1850000,"date":"2024-10-14T09:00","type":"RTGS"},
    {"id":"T02","from":"A02","to":"A03","amt":1820000,"date":"2024-10-14T14:30","type":"RTGS"},
    {"id":"T03","from":"A03","to":"A04","amt":1790000,"date":"2024-10-15T07:50","type":"RTGS"},
    {"id":"T04","from":"A04","to":"A01","amt":1750000,"date":"2024-10-15T13:20","type":"RTGS"},
    {"id":"T05","from":"A05","to":"A06","amt":4200000,"date":"2024-10-16T09:00","type":"NEFT"},
    {"id":"T06","from":"A06","to":"A07","amt":4150000,"date":"2024-10-16T13:15","type":"RTGS"},
    {"id":"T07","from":"A07","to":"A08","amt":4100000,"date":"2024-10-17T07:50","type":"RTGS"},
    {"id":"T08","from":"A08","to":"A09","amt":4060000,"date":"2024-10-17T11:30","type":"RTGS"},
    {"id":"T09","from":"A10","to":"A11","amt":96000,  "date":"2024-10-18T08:00","type":"NEFT"},
    {"id":"T10","from":"A10","to":"A11","amt":95000,  "date":"2024-10-18T11:30","type":"NEFT"},
    {"id":"T11","from":"A10","to":"A11","amt":94500,  "date":"2024-10-18T16:00","type":"NEFT"},
    {"id":"T12","from":"A10","to":"A11","amt":93000,  "date":"2024-10-19T05:45","type":"NEFT"},
    {"id":"T13","from":"A11","to":"A05","amt":370000, "date":"2024-10-19T10:00","type":"RTGS"},
    {"id":"T14","from":"A12","to":"A13","amt":98000,  "date":"2024-10-20T09:00","type":"NEFT"},
    {"id":"T15","from":"A12","to":"A13","amt":97000,  "date":"2024-10-20T12:30","type":"NEFT"},
    {"id":"T16","from":"A12","to":"A13","amt":96500,  "date":"2024-10-20T16:00","type":"NEFT"},
    {"id":"T17","from":"A12","to":"A13","amt":95000,  "date":"2024-10-21T01:30","type":"NEFT"},
    {"id":"T18","from":"A12","to":"A13","amt":94000,  "date":"2024-10-21T04:30","type":"NEFT"},
    {"id":"T19","from":"A13","to":"A05","amt":475000, "date":"2024-10-21T10:00","type":"RTGS"},
    {"id":"T20","from":"A05","to":"A14","amt":2800000,"date":"2024-10-19T09:00","type":"RTGS"},
    {"id":"T21","from":"A14","to":"A09","amt":2750000,"date":"2024-10-20T07:30","type":"RTGS"},
    {"id":"T22","from":"A21","to":"A15","amt":2200000,"date":"2024-10-20T10:00","type":"RTGS"},
    {"id":"T23","from":"A15","to":"A23","amt":2160000,"date":"2024-10-21T08:00","type":"RTGS"},
    {"id":"T24","from":"A05","to":"A16","amt":1900000,"date":"2024-10-22T10:00","type":"RTGS"},
    {"id":"T25","from":"A16","to":"A24","amt":1860000,"date":"2024-10-23T07:00","type":"RTGS"},
    {"id":"T26","from":"A17","to":"A18","amt":3200000,"date":"2024-10-15T08:00","type":"RTGS"},
    {"id":"T27","from":"A18","to":"A19","amt":3150000,"date":"2024-10-15T14:00","type":"RTGS"},
    {"id":"T28","from":"A19","to":"A20","amt":3100000,"date":"2024-10-16T09:00","type":"RTGS"},
    {"id":"T29","from":"A20","to":"A17","amt":3060000,"date":"2024-10-16T15:30","type":"RTGS"},
    {"id":"T30","from":"A05","to":"A21","amt":3500000,"date":"2024-10-17T08:00","type":"RTGS"},
    {"id":"T31","from":"A21","to":"A22","amt":3480000,"date":"2024-10-17T11:30","type":"RTGS"},
    {"id":"T32","from":"A22","to":"A23","amt":3460000,"date":"2024-10-17T15:00","type":"RTGS"},
    {"id":"T33","from":"A23","to":"A24","amt":3440000,"date":"2024-10-17T19:00","type":"RTGS"},
    {"id":"T34","from":"A24","to":"A09","amt":3420000,"date":"2024-10-18T07:30","type":"RTGS"},
    {"id":"T35","from":"A27","to":"A21","amt":800000, "date":"2024-10-22T09:00","type":"RTGS"},
    {"id":"T36","from":"A21","to":"A22","amt":792000, "date":"2024-10-22T14:00","type":"RTGS"},
    {"id":"T37","from":"A22","to":"A09","amt":785000, "date":"2024-10-22T20:00","type":"RTGS"},
    {"id":"T38","from":"A09","to":"A01","amt":920000, "date":"2024-10-23T10:00","type":"RTGS"},
    {"id":"T39","from":"A09","to":"A10","amt":780000, "date":"2024-10-23T11:00","type":"RTGS"},
    {"id":"T40","from":"A09","to":"A12","amt":650000, "date":"2024-10-23T14:00","type":"RTGS"},
    {"id":"T41","from":"A25","to":"A29","amt":42000,  "date":"2024-10-19T11:00","type":"NEFT"},
    {"id":"T42","from":"A29","to":"A25","amt":35000,  "date":"2024-10-20T10:00","type":"NEFT"},
    {"id":"T43","from":"A28","to":"A30","amt":310000, "date":"2024-10-21T13:00","type":"RTGS"},
    {"id":"T44","from":"A30","to":"A28","amt":290000, "date":"2024-10-22T09:00","type":"RTGS"},
    {"id":"T45","from":"A25","to":"A28","amt":55000,  "date":"2024-10-23T10:00","type":"NEFT"},
    {"id":"T46","from":"A27","to":"A30","amt":420000, "date":"2024-10-24T09:00","type":"RTGS"},
    {"id":"T47","from":"A11","to":"A27","amt":180000, "date":"2024-10-24T14:00","type":"NEFT"},
    {"id":"T48","from":"A30","to":"A11","amt":160000, "date":"2024-10-25T10:00","type":"NEFT"},
    {"id":"T49","from":"A29","to":"A30","amt":28000,  "date":"2024-10-26T11:00","type":"IMPS"},
    {"id":"T50","from":"A28","to":"A29","amt":22000,  "date":"2024-10-27T09:00","type":"IMPS"},
]

# Mutable pool (gets new injected transactions)
txn_pool   = [dict(t) for t in BASE_TXNS]
txn_ctr    = len(txn_pool)
current_th = copy.deepcopy(DEFAULT_TH)

# Cached engine result
_engine_cache: dict = {}


def run_engine() -> dict:
    """Run the fraud detection engine and cache the result."""
    global _engine_cache
    eng = FraudEngine(RAW_ACCOUNTS, txn_pool, current_th).analyze()
    enriched  = eng.enriched_accounts()
    txns_safe = eng.serializable_txns()
    patterns  = eng.serializable_patterns()
    alerts    = eng.alerts  # already JSON-safe

    # Build txn flag map for frontend
    txn_flag_map = {tid: list(flags) for tid, flags in eng.txn_flags.items()}

    _engine_cache = {
        "accounts":    enriched,
        "transactions": txns_safe,
        "alerts":      alerts,
        "patterns":    patterns,
        "counts":      eng.counts(),
        "runMs":       eng.run_ms,
        "txnFlagMap":  txn_flag_map,
        "muleStats":   {k: v for k, v in eng._mule.items()},
    }
    return _engine_cache


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "engine": "FundTrace IQ v4.0 Python Backend"})


@app.route("/api/analyze", methods=["GET", "POST"])
def analyze():
    """Run full analysis and return complete result."""
    result = run_engine()
    return jsonify(result)


@app.route("/api/accounts", methods=["GET"])
def get_accounts():
    if not _engine_cache:
        run_engine()
    return jsonify(_engine_cache["accounts"])


@app.route("/api/transactions", methods=["GET"])
def get_transactions():
    if not _engine_cache:
        run_engine()
    return jsonify(_engine_cache["transactions"])


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    if not _engine_cache:
        run_engine()
    return jsonify(_engine_cache["alerts"])


@app.route("/api/patterns", methods=["GET"])
def get_patterns():
    if not _engine_cache:
        run_engine()
    return jsonify(_engine_cache["patterns"])


@app.route("/api/inject", methods=["POST"])
def inject_txn():
    """Inject a new transaction and re-run the engine."""
    global txn_ctr
    data = request.json
    required = ["from", "to", "amt", "date", "type"]
    for f in required:
        if f not in data:
            return jsonify({"error": f"Missing field: {f}"}), 400
    if data["from"] == data["to"]:
        return jsonify({"error": "from and to must differ"}), 400
    txn_ctr += 1
    new_txn = {
        "id":   f"T{txn_ctr}",
        "from": data["from"],
        "to":   data["to"],
        "amt":  float(data["amt"]),
        "date": data["date"],
        "type": data["type"],
    }
    txn_pool.append(new_txn)
    result = run_engine()
    return jsonify({"message": f"Injected T{txn_ctr}", "txnId": f"T{txn_ctr}", "analysis": result})


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(current_th)


@app.route("/api/config", methods=["POST"])
def update_config():
    """Update thresholds and re-run the engine."""
    global current_th
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400
    # Deep merge
    def deep_merge(base, override):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                deep_merge(base[k], v)
            else:
                base[k] = v
    deep_merge(current_th, data)
    result = run_engine()
    return jsonify({"message": "Config updated", "config": current_th, "analysis": result})


@app.route("/api/str", methods=["GET"])
def get_str():
    """Generate the STR report text."""
    eng = FraudEngine(RAW_ACCOUNTS, txn_pool, current_th).analyze()
    report = eng.generate_str_report()
    return jsonify({"report": report})


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset transactions to baseline."""
    global txn_pool, txn_ctr, current_th
    txn_pool   = [dict(t) for t in BASE_TXNS]
    txn_ctr    = len(txn_pool)
    current_th = copy.deepcopy(DEFAULT_TH)
    result = run_engine()
    return jsonify({"message": "Reset to baseline", "analysis": result})


@app.route("/api/graph", methods=["GET"])
def get_graph():
    """Return graph structure for NetworkX-computed metrics."""
    if not _engine_cache:
        run_engine()
    # Build node/edge list for frontend graph
    nodes = []
    for a in _engine_cache["accounts"]:
        nodes.append({
            "id":    a["id"],
            "name":  a["name"],
            "score": a.get("score", 0),
            "risk":  a.get("risk", "low"),
            "flags": a.get("flags", []),
            "status": a.get("status", "active"),
            "isMule": a.get("isMule", False),
        })
    edges = []
    for t in _engine_cache["transactions"]:
        edges.append({
            "id":   t["id"],
            "from": t["from"],
            "to":   t["to"],
            "amt":  t["amt"],
            "flag": t.get("flag"),
        })
    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/api/profiles", methods=["GET"])
def get_profiles():
    return jsonify(PROFILES)


# ─────────────────────────────────────────────────────────────────────────────
#  STATIC FRONTEND
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Warm-up analysis on start
    print("[FundTrace IQ] Running initial analysis…")
    run_engine()
    c = _engine_cache["counts"]
    print(f"[FundTrace IQ] Ready — {c['total']} patterns detected in {_engine_cache['runMs']}ms")
    app.run(debug=True, port=5000)
