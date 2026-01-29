import os
import glob
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from math import erf, sqrt

# -----------------------------
# Helpers: time and paths
# -----------------------------

def quarter_end_from_year_quarter(year: int, quarter: str) -> pd.Timestamp:
    q = quarter.upper().strip()
    if q not in {"Q1", "Q2", "Q3", "Q4"}:
        raise ValueError("Quarter must be one of: Q1, Q2, Q3, Q4")
    q_num = int(q[1])
    return pd.Period(f"{year}Q{q_num}", freq="Q").to_timestamp("Q")


def add_quarters(qe: pd.Timestamp, q: int) -> pd.Timestamp:
    if pd.isna(qe):
        return pd.NaT
    p = pd.Period(qe, freq="Q")
    return (p + int(q)).to_timestamp("Q")


def quarter_range(start_qe: pd.Timestamp, end_qe: pd.Timestamp) -> List[pd.Timestamp]:
    if pd.isna(start_qe) or pd.isna(end_qe):
        return []
    p0 = pd.Period(start_qe, freq="Q")
    p1 = pd.Period(end_qe, freq="Q")
    if p1 < p0:
        return []
    return [p.to_timestamp("Q") for p in pd.period_range(p0, p1, freq="Q")]


def _to_period_q(x) -> pd.Period:
    if isinstance(x, pd.Period):
        return x.asfreq("Q")
    try:
        return pd.Period(x, freq="Q")
    except Exception:
        return pd.Period(pd.Timestamp(x), freq="Q")


def quarter_diff(qe: pd.Timestamp, start_qe: pd.Timestamp) -> int:
    p = _to_period_q(qe)
    s = _to_period_q(start_qe)
    return int(p.ordinal - s.ordinal)


# -----------------------------
# Config (aligns with Structural-cashflows)
# -----------------------------

AGE_BINS_Q = [-1, 7, 15, 23, 31, 39, 59, 79, 10_000]
AGE_LABELS = ["0-2y", "2-4y", "4-6y", "6-8y", "8-10y", "10-15y", "15-20y", "20y+"]

NAV_EPS = 100.0
NAV_STOP_EPS = 1.0
CAP_EPS = 1.0

RUNOFF_Q = 12
REP_RAMP_P = 2.0
REP_RAMP_SIZE = 1.0
REP_RAMP_FLOOR = 0.05

USE_RUNOFF_CALIBRATION = True
RUNOFF_MULT_MIN = 0.5
RUNOFF_MULT_MAX = 3.0

IP_YEARS_DEFAULT = 5
IP_Q_DEFAULT = int(IP_YEARS_DEFAULT * 4)
IP_CUM_PCTL = 0.80
IP_Q_MIN = 4
IP_Q_MAX = 40
DRAW_AGE_MIN_MULT = 0.2
DRAW_AGE_DECAY_POWER = 1.0

ENFORCE_IP_LIMITS = False
USE_DRAW_AGE_SHAPE = False
USE_FORCED_TERMINAL_REPAY = False
AGE_SOURCE = "fund_age"  # "fund_age" or "first_close"
USE_NAV_PROJECTIONS = True  # use NAV Logic omega/nav_start files if available
RUN_NAV_LOGIC_INLINE = True  # run NAV Logic in-memory for backtest window (no file outputs)
NAV_LOGIC_ALPHA_LEVEL = 0.10
NAV_LOGIC_MIN_CLUSTERS = 8
NAV_LOGIC_MSCI_MODE = "unconditional"  # "conditional" or "unconditional"
RUN_CONDITIONAL = False
RUN_UNCONDITIONAL = True

# Calibration / reporting bucketing
CALIBRATION_BUCKET_MODE = "strategy_grade"  # "strategy_grade_age" or "strategy_grade"
REPORT_BUCKET_MODE = "strategy_grade"       # "strategy_grade_age", "strategy_grade", or "strategy"

GRADE_P_MULT = {"A": 1.15, "B": 1.00, "C": 0.85, "D": 0.70}
GRADE_SIZE_MULT = {"A": 1.10, "B": 1.00, "C": 0.90, "D": 0.80}

GRADE_DRAW_P_MULT = {"A": 0.95, "B": 1.00, "C": 1.05, "D": 1.10}
GRADE_DRAW_SIZE_MULT = {"A": 0.95, "B": 1.00, "C": 1.05, "D": 1.10}

MSCI_REP_P_BETA = 0.6
MSCI_REP_SIZE_BETA = 0.4
MSCI_Z_CLIP = 2.0
MSCI_REP_POS_ONLY = True

SIGMA_FLOOR = 0.35
SIGMA_CAP = 2.0

MIN_LN_OBS = 30
MIN_LN_FUNDS = 5
KS_ALPHA = 0.05
SHRINK_N = 100
SHRINK_FUNDS = 10

USE_HAZARD_MODELS = True
LOGIT_L2 = 1.0
LOGIT_MAX_ITER = 50
LOGIT_TOL = 1e-6

SOFT_RHO_PCTL = 0.95
SOFT_EXPIRY_FALLBACK = 20

GRADE_STATES = ["A", "B", "C", "D"]

OMEGA_CLIP = 0.8
GRADE_OMEGA_BIAS = {"A": 0.005, "B": 0.0, "C": -0.005, "D": -0.01}

# -----------------------------
# Calibration controls
# -----------------------------
USE_DRAWDOWN_CALIBRATION = False
# "mean", "median", or "auto" (pick lower SSE per group)
DRAW_CALIB_TARGET = "auto"
DRAW_CALIB_MIN_FUNDS = 10
DRAW_CALIB_MIN_AGES = 8


# -----------------------------
# Helpers: math / distributions
# -----------------------------

def make_age_bucket_q(age_q: float):
    return pd.cut(pd.Series([age_q]), bins=AGE_BINS_Q, labels=AGE_LABELS).iloc[0]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def one_factor_uniforms(n: int, rng: np.random.Generator, rho_mkt: float) -> np.ndarray:
    rho_mkt = float(np.clip(rho_mkt, 0.0, 0.999))
    Z = rng.standard_normal()
    eps = rng.standard_normal(n)
    X = np.sqrt(rho_mkt) * Z + np.sqrt(1.0 - rho_mkt) * eps
    return np.array([norm_cdf(x) for x in X], dtype=float)


def inv_norm(u: float) -> float:
    try:
        from scipy.special import erfinv
        return sqrt(2.0) * float(erfinv(2.0 * u - 1.0))
    except Exception:
        u = float(np.clip(u, 1e-6, 1.0 - 1e-6))
        return float(np.sign(u - 0.5) * np.sqrt(2.0) * np.sqrt(abs(np.log(1.0 - 2.0 * abs(u - 0.5)))))


def lognormal_from_u(mu: float, sigma: float, u: float) -> float:
    z = inv_norm(u)
    return float(np.exp(mu + sigma * z))


# -----------------------------
# Recallable ledger
# -----------------------------

@dataclass
class RecallableBucket:
    created_q: int
    expiry_q: int
    amount_remaining: float


@dataclass
class RecallableLedger:
    rho: float
    expiry_quarters: int
    commitment: float
    buckets: List[RecallableBucket] = field(default_factory=list)

    def _rc_cap(self) -> float:
        return max(float(self.rho), 0.0) * max(float(self.commitment), 0.0)

    def drop_expired(self, q: int) -> None:
        if int(self.expiry_quarters) <= 0:
            self.buckets = []
            return
        self.buckets = [b for b in self.buckets if b.expiry_q >= q and b.amount_remaining > 0]

    def available(self, q: int) -> float:
        self.drop_expired(q)
        return float(sum(b.amount_remaining for b in self.buckets))

    def add_recallable(self, q: int, rc_amount: float, enforce_cap: bool = True) -> float:
        self.drop_expired(q)
        x = max(float(rc_amount or 0.0), 0.0)
        if x <= 0.0 or int(self.expiry_quarters) <= 0:
            return 0.0

        add_amt = x
        if enforce_cap:
            cap = self._rc_cap()
            cur = self.available(q)
            room = max(cap - cur, 0.0)
            add_amt = min(add_amt, room)

        if add_amt <= 0.0:
            return 0.0

        self.buckets.append(RecallableBucket(
            created_q=q,
            expiry_q=q + int(self.expiry_quarters),
            amount_remaining=float(add_amt)
        ))
        return float(add_amt)

    def consume_for_drawdown(self, q: int, draw_amount: float) -> Dict[str, float]:
        self.drop_expired(q)
        need = max(float(draw_amount or 0.0), 0.0)
        if need <= 0.0:
            return {"use_rc": 0.0, "use_commitment": 0.0}

        self.buckets.sort(key=lambda b: b.created_q)
        use_rc = 0.0
        for b in self.buckets:
            if need <= 0:
                break
            take = min(b.amount_remaining, need)
            b.amount_remaining -= take
            need -= take
            use_rc += take

        self.buckets = [b for b in self.buckets if b.amount_remaining > 0]
        use_commitment = max(float(draw_amount) - use_rc, 0.0)
        return {"use_rc": float(use_rc), "use_commitment": float(use_commitment)}


# -----------------------------
# Hazard model helpers
# -----------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def build_feature_matrix(df: pd.DataFrame, include_nav: bool) -> pd.DataFrame:
    d = df.copy()
    d["Adj Strategy"] = d["Adj Strategy"].fillna("Unknown")
    d["Grade"] = d["Grade"].fillna("D").astype(str).str.strip()

    X = pd.DataFrame(index=d.index)
    X["intercept"] = 1.0
    X["age_q"] = d["age_q"].astype(float)
    X["age_q2"] = (d["age_q"].astype(float) ** 2)
    if include_nav:
        X["log_nav_prev"] = d["log_nav_prev"].astype(float)

    strat_d = pd.get_dummies(d["Adj Strategy"], prefix="strat", drop_first=True)
    grade_d = pd.get_dummies(d["Grade"], prefix="grade", drop_first=True)

    X = pd.concat([X, strat_d, grade_d], axis=1)
    return X


def standardize_X(X: pd.DataFrame, cont_cols: list) -> Tuple[pd.DataFrame, dict, dict]:
    means = {}
    stds = {}
    X = X.copy()
    for c in cont_cols:
        if c in X.columns:
            mu = float(X[c].mean())
            sd = float(X[c].std(ddof=1)) if len(X) > 1 else 1.0
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0
            means[c] = mu
            stds[c] = sd
            X[c] = (X[c] - mu) / sd
    return X, means, stds


def apply_standardize(X: pd.DataFrame, means: dict, stds: dict) -> pd.DataFrame:
    X = X.copy()
    for c, mu in means.items():
        if c in X.columns:
            sd = stds.get(c, 1.0)
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0
            X[c] = (X[c] - mu) / sd
    return X


def fit_logit(X: np.ndarray, y: np.ndarray, l2: float = LOGIT_L2, max_iter: int = LOGIT_MAX_ITER, tol: float = LOGIT_TOL) -> np.ndarray:
    n, p = X.shape
    beta = np.zeros(p, dtype=float)
    I = np.eye(p)
    for _ in range(max_iter):
        z = X @ beta
        p_hat = _sigmoid(z)
        W = p_hat * (1.0 - p_hat)
        W = np.clip(W, 1e-6, None)
        z_adj = z + (y - p_hat) / W
        XTW = X.T * W
        A = XTW @ X + l2 * I
        b = XTW @ z_adj
        try:
            beta_new = np.linalg.solve(A, b)
        except Exception:
            beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def build_feature_row(strategy: str, grade: str, age_q: int, log_nav_prev: float, include_nav: bool,
                      cols: list, means: dict, stds: dict) -> np.ndarray:
    row = {
        "Adj Strategy": strategy,
        "Grade": grade,
        "age_q": float(age_q),
        "age_q2": float(age_q) ** 2,
        "log_nav_prev": float(log_nav_prev) if include_nav else 0.0,
    }
    df = pd.DataFrame([row])
    X = build_feature_matrix(df, include_nav=include_nav)
    X = X.reindex(columns=cols, fill_value=0.0)
    X = apply_standardize(X, means, stds)
    return X.to_numpy(dtype=float)


def ks_test_normal(log_x: np.ndarray, alpha: float = KS_ALPHA) -> Tuple[float, bool]:
    n = len(log_x)
    if n < 2:
        return float("nan"), False
    mu = float(np.mean(log_x))
    sig = float(np.std(log_x, ddof=1)) if n > 1 else 0.0
    if not np.isfinite(sig) or sig <= 0:
        return float("nan"), False

    x = np.sort(log_x)
    z = (x - mu) / (sig * np.sqrt(2.0))
    try:
        F = 0.5 * (1.0 + np.erf(z))
    except Exception:
        F = 0.5 * (1.0 + np.vectorize(erf)(z))
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - F)
    d_minus = np.max(F - (i - 1) / n)
    D = float(max(d_plus, d_minus))

    dcrit = float(np.sqrt(-0.5 * np.log(alpha / 2.0) / n))
    return D, bool(D <= dcrit)


def fit_lognormal_stats(x: pd.Series, fund_ids: pd.Series) -> Dict[str, float]:
    g = x.dropna()
    g = g[g > 0]
    n_obs = int(len(g))
    if n_obs == 0:
        return {
            "mu": 0.0, "sig": SIGMA_FLOOR,
            "n": 0, "n_funds": 0,
            "ks_D": float("nan"), "ks_pass": False,
        }

    n_funds = int(fund_ids.loc[g.index].nunique()) if fund_ids is not None else 0

    lx = np.log(g.to_numpy(dtype=float))
    mu = float(np.mean(lx))
    sig = float(np.std(lx, ddof=1)) if n_obs > 1 else SIGMA_FLOOR
    sig = float(np.clip(max(sig, SIGMA_FLOOR), SIGMA_FLOOR, SIGMA_CAP))

    ks_D, ks_pass = (float("nan"), False)
    if n_obs >= MIN_LN_OBS and n_funds >= MIN_LN_FUNDS:
        ks_D, ks_pass = ks_test_normal(lx, alpha=KS_ALPHA)

    return {
        "mu": mu, "sig": sig,
        "n": n_obs, "n_funds": n_funds,
        "ks_D": ks_D, "ks_pass": ks_pass,
    }


def _weight(n_obs: float, n_funds: float, ks_pass: bool) -> float:
    if not ks_pass:
        return 0.0
    if n_obs is None or n_obs <= 0 or n_funds is None or n_funds <= 0:
        return 0.0
    w = (n_obs / (n_obs + SHRINK_N)) * (n_funds / (n_funds + SHRINK_FUNDS))
    return float(np.clip(w, 0.0, 1.0))


def _weight_h(n_obs: float, n_funds: float) -> float:
    if n_obs is None or n_obs <= 0 or n_funds is None or n_funds <= 0:
        return 0.0
    w = (n_obs / (n_obs + SHRINK_N)) * (n_funds / (n_funds + SHRINK_FUNDS))
    return float(np.clip(w, 0.0, 1.0))


def _blend_p(p_c: float, n_c: float, nf_c: float, p_p: float) -> float:
    w = _weight_h(n_c, nf_c)
    return float(np.clip(w * float(p_c) + (1.0 - w) * float(p_p), 0.0, 1.0))


def _combine_p(p_sg, n_sg, nf_sg, p_sa, n_sa, nf_sa, p_s):
    w_sg = _weight_h(n_sg, nf_sg)
    w_sa = _weight_h(n_sa, nf_sa)
    tot = w_sg + w_sa
    if tot > 0:
        p_mid = (w_sg * float(p_sg) + w_sa * float(p_sa)) / tot
        return float(np.clip(tot * p_mid + (1.0 - tot) * float(p_s), 0.0, 1.0))
    return float(np.clip(float(p_s), 0.0, 1.0))


def _combine_mu_sig(mu_sg, sig_sg, n_sg, nf_sg, ks_sg,
                    mu_sa, sig_sa, n_sa, nf_sa, ks_sa,
                    mu_s, sig_s):
    w_sg = _weight(n_sg, nf_sg, ks_sg)
    w_sa = _weight(n_sa, nf_sa, ks_sa)
    tot = w_sg + w_sa
    if tot > 0:
        mu_mid = (w_sg * float(mu_sg) + w_sa * float(mu_sa)) / tot
        sig_mid = (w_sg * float(sig_sg) + w_sa * float(sig_sa)) / tot
        mu = tot * mu_mid + (1.0 - tot) * float(mu_s)
        sig = tot * sig_mid + (1.0 - tot) * float(sig_s)
    else:
        mu = float(mu_s)
        sig = float(sig_s)
    sig = float(np.clip(max(sig, SIGMA_FLOOR), SIGMA_FLOOR, SIGMA_CAP))
    return mu, sig


def _blend(mu_c, sig_c, n_c, nf_c, ks_c, mu_p, sig_p, n_p, nf_p, ks_p):
    w = _weight(n_c, nf_c, ks_c)
    mu = w * float(mu_c) + (1.0 - w) * float(mu_p)
    sig = w * float(sig_c) + (1.0 - w) * float(sig_p)
    sig = float(np.clip(max(sig, SIGMA_FLOOR), SIGMA_FLOOR, SIGMA_CAP))
    return mu, sig


# -----------------------------
# MSCI model (simplified from msci_projection)
# -----------------------------

def load_msci_quarterly(msci_xlsx_path: str) -> pd.DataFrame:
    msci = pd.read_excel(msci_xlsx_path)
    if "Date" not in msci.columns or "SCXP Index" not in msci.columns:
        raise ValueError("MSCI file must contain columns: 'Date' and 'SCXP Index'")
    msci = msci[["Date", "SCXP Index"]].copy()
    msci["Date"] = pd.to_datetime(msci["Date"], errors="coerce")
    msci["SCXP Index"] = pd.to_numeric(msci["SCXP Index"], errors="coerce")
    msci = msci.dropna(subset=["Date", "SCXP Index"]).sort_values("Date")
    msci["quarter_end"] = msci["Date"].dt.to_period("Q").dt.to_timestamp("Q")
    q = (msci.groupby("quarter_end", as_index=False)["SCXP Index"]
         .last()
         .rename(columns={"SCXP Index": "index_level"})
         .sort_values("quarter_end")
         .reset_index(drop=True))
    q["msci_ret_q"] = q["index_level"].pct_change()
    q = q.dropna(subset=["msci_ret_q"]).reset_index(drop=True)
    return q


def label_regimes_by_quantiles(q_returns: pd.Series, low_q=0.33, high_q=0.67) -> pd.Series:
    q_low = q_returns.quantile(low_q)
    q_high = q_returns.quantile(high_q)
    regime = pd.Series(index=q_returns.index, dtype="object")
    regime[q_returns <= q_low] = "bear"
    regime[q_returns >= q_high] = "bull"
    regime[(q_returns > q_low) & (q_returns < q_high)] = "flat"
    return regime


def estimate_transition_matrix(regimes: pd.Series, states=("bear", "flat", "bull"), laplace=1.0) -> pd.DataFrame:
    states = list(states)
    counts = pd.DataFrame(0.0, index=states, columns=states)
    r = regimes.dropna().tolist()
    for a, b in zip(r[:-1], r[1:]):
        if a in states and b in states:
            counts.loc[a, b] += 1.0
    counts = counts + laplace
    P = counts.div(counts.sum(axis=1), axis=0)
    return P


def estimate_regime_params(df_q: pd.DataFrame, states=("bear", "flat", "bull")) -> pd.DataFrame:
    overall_sigma = float(df_q["msci_ret_q"].std(ddof=1))
    overall_sigma = max(overall_sigma, 1e-6)
    out = []
    for s in states:
        sub = df_q.loc[df_q["regime"] == s, "msci_ret_q"].dropna()
        mu = float(sub.mean()) if len(sub) else 0.0
        sigma = float(sub.std(ddof=1)) if len(sub) > 1 else overall_sigma
        sigma = max(sigma, 1e-6)
        out.append((s, mu, sigma))
    return pd.DataFrame(out, columns=["regime", "mu_q", "sigma_q"]).set_index("regime")


def apply_persistence_tilt(P: pd.DataFrame, scenario: str, k: float = 1.2) -> pd.DataFrame:
    scenario = scenario.lower().strip()
    if scenario not in {"bullish", "neutral", "bearish"}:
        raise ValueError("scenario must be one of: bullish, neutral, bearish")
    if scenario == "neutral":
        return P.copy()
    target = "bull" if scenario == "bullish" else "bear"
    P2 = P.copy()
    for s in P2.index:
        P2.loc[s, target] *= k
    P2.loc[target, target] *= k
    P2 = P2.div(P2.sum(axis=1), axis=0)
    return P2


def simulate_markov_regimes(P: pd.DataFrame, start_state: str, n_steps: int, rng: np.random.Generator) -> list:
    states = list(P.index)
    if start_state not in states:
        start_state = "flat" if "flat" in states else states[0]
    path = [start_state]
    for _ in range(n_steps):
        cur = path[-1]
        probs = P.loc[cur].values.astype(float)
        nxt = rng.choice(states, p=probs)
        path.append(nxt)
    return path[1:]


def simulate_msci_path(df_q_hist: pd.DataFrame, start_qe: pd.Timestamp, n_quarters: int,
                       scenario: str, tilt_strength: float, rng: np.random.Generator) -> pd.DataFrame:
    df = df_q_hist.copy()
    df["regime"] = label_regimes_by_quantiles(df["msci_ret_q"], low_q=0.33, high_q=0.67)
    P = estimate_transition_matrix(df["regime"], laplace=1.0)
    params = estimate_regime_params(df)
    P_tilted = apply_persistence_tilt(P, scenario=scenario, k=tilt_strength)
    df_reg = df.loc[df["quarter_end"] <= start_qe].dropna(subset=["regime"])
    start_regime = df_reg["regime"].iloc[-1] if not df_reg.empty else "flat"
    future_qe = quarter_range(add_quarters(start_qe, 1), add_quarters(start_qe, n_quarters))
    regime_path = simulate_markov_regimes(P_tilted, start_regime, n_quarters, rng)
    rows = []
    for qe, s in zip(future_qe, regime_path):
        mu = float(params.loc[s, "mu_q"])
        sig = float(params.loc[s, "sigma_q"])
        r = mu + sig * rng.standard_normal()
        rows.append({"quarter_end": qe, "msci_ret_q": r, "regime": s})
    return pd.DataFrame(rows)


# -----------------------------
# Grade transitions (yearly)
# -----------------------------

def build_yearly_transition_from_data(df: pd.DataFrame, strategy: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    df = df.copy()
    if strategy is not None:
        df = df[df["Adj Strategy"] == strategy].copy()
    transitions = []
    for _, g in df.groupby("FundID"):
        g = g.sort_values("quarter_end")
        grades = g["Grade"].fillna("D").astype(str).tolist()
        if len(grades) < 5:
            continue
        yearly = grades[::4]
        transitions.extend(list(zip(yearly[:-1], yearly[1:])))
    if not transitions:
        states = GRADE_STATES
        counts = pd.DataFrame(1.0, index=states, columns=states)
        probs = counts.div(counts.sum(axis=1), axis=0)
        return counts, probs, 0

    counts = pd.crosstab(
        [a for a, _ in transitions],
        [b for _, b in transitions]
    ).reindex(index=GRADE_STATES, columns=GRADE_STATES, fill_value=0.0)
    counts = counts + 1.0
    probs = counts.div(counts.sum(axis=1), axis=0)
    return counts, probs, len(transitions)


def sample_next_grade(curr_grade: str, P_df: pd.DataFrame, rng: np.random.Generator) -> str:
    if curr_grade not in GRADE_STATES:
        curr_grade = "D"
    row = P_df.loc[curr_grade].values.astype(float)
    return str(rng.choice(GRADE_STATES, p=row))


# -----------------------------
# Omega model (simplified NAV Logic)
# -----------------------------

def fit_ols_beta(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, float]:
    # x: n x 2 (r_t, r_{t-1})
    X = np.column_stack([np.ones(len(y)), x])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        beta = np.zeros(3, dtype=float)
    return float(beta[0]), float(beta[1]), float(beta[2])


def build_omega_models(cal: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], Tuple[float, float, str]],
                                                   Dict[Tuple[str, str, str], Tuple[float, str]],
                                                   Dict[Tuple[str, str], Tuple[float, str]]]:
    # Betas by (strategy, grade), fallback to strategy, then global
    betas_sg = {}
    betas_s = {}
    betas_g = (0.0, 0.0, 0.0)

    # Global
    y = cal["omega"].to_numpy(dtype=float)
    x = cal[["msci_ret_q", "msci_ret_q_lag1"]].to_numpy(dtype=float)
    a_g, b0_g, b1_g = fit_ols_beta(y, x)
    betas_g = (a_g, b0_g, b1_g)

    # Strategy-level
    for s, grp in cal.groupby("Adj Strategy"):
        y = grp["omega"].to_numpy(dtype=float)
        x = grp[["msci_ret_q", "msci_ret_q_lag1"]].to_numpy(dtype=float)
        a, b0, b1 = fit_ols_beta(y, x)
        betas_s[s] = (a, b0, b1)

    # Strategy+grade
    for (s, g), grp in cal.groupby(["Adj Strategy", "Grade"]):
        if len(grp) < 20:
            continue
        y = grp["omega"].to_numpy(dtype=float)
        x = grp[["msci_ret_q", "msci_ret_q_lag1"]].to_numpy(dtype=float)
        a, b0, b1 = fit_ols_beta(y, x)
        betas_sg[(s, g)] = (a, b0, b1)

    # Alpha by (strategy, grade, age_bucket)
    alpha_sga = {}
    alpha_sg = {}
    alpha_s = {}

    cal2 = cal.copy()
    # Use best available betas to compute omega_adj
    def get_betas(strategy: str, grade: str) -> Tuple[float, float, float]:
        if (strategy, grade) in betas_sg:
            return betas_sg[(strategy, grade)]
        if strategy in betas_s:
            return betas_s[strategy]
        return betas_g

    b0_list = []
    b1_list = []
    for _, r in cal2.iterrows():
        a, b0, b1 = get_betas(r["Adj Strategy"], r["Grade"])
        b0_list.append(b0)
        b1_list.append(b1)
    cal2["b0_used"] = b0_list
    cal2["b1_used"] = b1_list
    cal2["omega_adj"] = cal2["omega"] - cal2["b0_used"] * cal2["msci_ret_q"] - cal2["b1_used"] * cal2["msci_ret_q_lag1"]

    for (s, g, a), grp in cal2.groupby(["Adj Strategy", "Grade", "AgeBucket"]):
        if len(grp) < 10:
            continue
        alpha_sga[(s, g, a)] = (float(grp["omega_adj"].mean()), "sga")

    for (s, g), grp in cal2.groupby(["Adj Strategy", "Grade"]):
        if len(grp) < 10:
            continue
        alpha_sg[(s, g)] = (float(grp["omega_adj"].mean()), "sg")

    for s, grp in cal2.groupby(["Adj Strategy"]):
        alpha_s[s] = (float(grp["omega_adj"].mean()), "s")

    alpha_g = (float(cal2["omega_adj"].mean()), "g")

    # Sigma by (strategy, grade)
    sigma_sg = {}
    sigma_s = {}

    cal2["omega_resid"] = cal2["omega_adj"] - cal2.groupby(["Adj Strategy", "Grade"])["omega_adj"].transform("mean")

    for (s, g), grp in cal2.groupby(["Adj Strategy", "Grade"]):
        if len(grp) < 20:
            continue
        sig = float(grp["omega_resid"].std(ddof=1))
        if not np.isfinite(sig) or sig <= 0:
            continue
        sigma_sg[(s, g)] = (sig, "sg")

    for s, grp in cal2.groupby(["Adj Strategy"]):
        sig = float(grp["omega_adj"].std(ddof=1))
        if np.isfinite(sig) and sig > 0:
            sigma_s[s] = (sig, "s")

    sigma_g = float(cal2["omega_adj"].std(ddof=1))
    if not np.isfinite(sigma_g) or sigma_g <= 0:
        sigma_g = 0.05
    sigma_g = (sigma_g, "g")

    def get_alpha(strategy: str, grade: str, age_bucket: str) -> Tuple[float, str]:
        k = (strategy, grade, age_bucket)
        if k in alpha_sga:
            return alpha_sga[k]
        k2 = (strategy, grade)
        if k2 in alpha_sg:
            return alpha_sg[k2]
        if strategy in alpha_s:
            return alpha_s[strategy]
        return alpha_g

    def get_sigma(strategy: str, grade: str) -> Tuple[float, str]:
        k = (strategy, grade)
        if k in sigma_sg:
            return sigma_sg[k]
        if strategy in sigma_s:
            return sigma_s[strategy]
        return sigma_g

    return get_betas, get_alpha, get_sigma


# -----------------------------
# NAV Logic inline (no file outputs)
# -----------------------------

def run_nav_logic_inline(
    data: pd.DataFrame,
    msci_hist: pd.DataFrame,
    msci_future: pd.DataFrame,
    start_qe: pd.Timestamp,
    data_dir: str,
    alpha_level: float = 0.10,
    min_clusters_for_inference: int = 8,
    seed: int = 1234,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Keep methodology aligned with NAV Logic.ipynb
    try:
        from scipy import stats  # noqa: F401
    except Exception as e:
        raise ImportError("scipy is required for NAV Logic inline runs.") from e

    NAV_COL = "NAV Adjusted EUR"
    DRAW_COL = "Adj Drawdown EUR"
    REPAY_COL = "Adj Repayment EUR"
    SIZE_COL = "Target Fund Size"

    NAV_EPS = 100.0
    OMEGA_CLIP = 0.8
    GRADE_OMEGA_BIAS = {"A": 0.005, "B": 0.0, "C": -0.005, "D": -0.01}

    MIN_FUNDS_BETA = 10
    MIN_OBS_BETA = 80
    MIN_FUNDS_ALPHA_BUCKET = 6
    MIN_OBS_ALPHA_BUCKET = 60
    SIGMA_SHRINK_K = 120.0
    DRAW_EPS = 1000.0
    SIZE_EPS = 1e6
    MIN_OBS_RATIO = 50

    df = data.copy()
    df["quarter_end"] = pd.to_datetime(df["quarter_end"]).dt.to_period("Q").dt.to_timestamp("Q")
    df = df.sort_values(["FundID", "quarter_end"]).reset_index(drop=True)

    required_cols = [
        "FundID", "quarter_end",
        NAV_COL, DRAW_COL, REPAY_COL,
        "Adj Strategy", "Grade",
        SIZE_COL, "Fund_Age_Quarters",
        "Planned end date with add. years as per legal doc",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data for NAV Logic inline: {missing}")

    df["planned_end_qe"] = pd.to_datetime(
        df["Planned end date with add. years as per legal doc"],
        errors="coerce"
    ).dt.to_period("Q").dt.to_timestamp("Q")

    df[NAV_COL] = pd.to_numeric(df[NAV_COL], errors="coerce")
    for c in [DRAW_COL, REPAY_COL, SIZE_COL, "Fund_Age_Quarters"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # MSCI history (ensure lag exists)
    msci_hist = msci_hist.copy()
    msci_hist["quarter_end"] = pd.to_datetime(msci_hist["quarter_end"]).dt.to_period("Q").dt.to_timestamp("Q")
    if "msci_ret_q_lag1" not in msci_hist.columns:
        msci_hist["msci_ret_q_lag1"] = msci_hist["msci_ret_q"].shift(1)

    # MSCI future map (single path)
    msci_future = msci_future.copy()
    msci_future["quarter_end"] = pd.to_datetime(msci_future["quarter_end"]).dt.to_period("Q").dt.to_timestamp("Q")
    if "msci_ret_q_lag1" not in msci_future.columns:
        msci_future["msci_ret_q_lag1"] = msci_future["msci_ret_q"].shift(1)

    msci_future_map = {1: msci_future}
    future_qe_max = msci_future["quarter_end"].iloc[-1]

    # Planned end overrun by strategy (history-based)
    last_obs = df.groupby("FundID")["quarter_end"].max().rename("last_qe")
    fund_static = df.sort_values(["FundID", "quarter_end"]).groupby("FundID").tail(1).copy()
    fund_static = fund_static.merge(last_obs, on="FundID", how="left")

    def quarters_diff(a: pd.Timestamp, b: pd.Timestamp) -> float:
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return float(pd.Period(a, freq="Q").ordinal - pd.Period(b, freq="Q").ordinal)

    fund_static["overrun_q"] = fund_static.apply(
        lambda r: max(quarters_diff(r["last_qe"], r["planned_end_qe"]), 0.0)
        if pd.notna(r["planned_end_qe"]) else np.nan,
        axis=1
    )

    fund_static["ever_overran"] = fund_static["overrun_q"].fillna(0) > 0
    ever_overran_map = fund_static.set_index("FundID")["ever_overran"]
    overran_only = fund_static.loc[fund_static["overrun_q"].notna() & (fund_static["overrun_q"] > 0)].copy()
    avg_overrun_by_strategy = overran_only.groupby("Adj Strategy")["overrun_q"].mean().clip(lower=0.0)

    # Build omega from history for calibration
    df["nav_prev"] = df.groupby("FundID")[NAV_COL].shift(1)
    df["flow_net"] = df[DRAW_COL] - df[REPAY_COL]
    m = df["nav_prev"].abs() > NAV_EPS
    df["omega"] = np.nan
    df.loc[m, "omega"] = ((df.loc[m, NAV_COL] - df.loc[m, "nav_prev"]) - df.loc[m, "flow_net"]) / df.loc[m, "nav_prev"]
    df["omega"] = df["omega"].clip(lower=-OMEGA_CLIP, upper=OMEGA_CLIP)

    cal = df.merge(msci_hist, on="quarter_end", how="left")
    cal = cal.dropna(subset=["omega", "msci_ret_q", "msci_ret_q_lag1"]).copy()
    cal["AgeBucket"] = pd.cut(cal["Fund_Age_Quarters"], bins=AGE_BINS_Q, labels=AGE_LABELS)
    cal = cal[["FundID", "Adj Strategy", "Grade", "AgeBucket", "omega", "msci_ret_q", "msci_ret_q_lag1"]].copy()
    if cal.empty:
        raise ValueError("NAV Logic inline: no calibration rows after filtering.")

    from scipy import stats

    def ols_cluster_robust(df_in, y_col, x_cols, cluster_col):
        d = df_in.dropna(subset=[y_col] + x_cols + [cluster_col]).copy()
        n = len(d)
        if n == 0:
            return None

        y = d[y_col].to_numpy(float)
        X = np.column_stack([np.ones(n)] + [d[c].to_numpy(float) for c in x_cols])
        k = X.shape[1]

        XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ (X.T @ y)

        u = y - X @ beta
        clusters = d[cluster_col].to_numpy()
        uniq = pd.unique(clusters)
        G = len(uniq)
        df_dof = G - 1
        if G <= 1:
            return None

        meat = np.zeros((k, k), dtype=float)
        for g in uniq:
            mask = (clusters == g)
            Xg = X[mask, :]
            ug = u[mask]
            Xgu = Xg.T @ ug
            meat += np.outer(Xgu, Xgu)

        V = XtX_inv @ meat @ XtX_inv
        scale = (G / (G - 1)) * ((n - 1) / max(n - k, 1))
        V *= scale

        se = np.sqrt(np.diag(V))
        tstats = beta / se
        pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tstats), df=df_dof))

        names = ["alpha"] + x_cols
        return {
            "coef": pd.Series(beta, index=names),
            "se": pd.Series(se, index=names),
            "t": pd.Series(tstats, index=names),
            "p": pd.Series(pvals, index=names),
            "n_obs": int(n),
            "n_clusters": int(G),
            "df_dof": int(df_dof),
        }

    def cluster_mean_stats(df_in, y_col, cluster_col):
        res = ols_cluster_robust(df_in, y_col=y_col, x_cols=[], cluster_col=cluster_col)
        if res is None:
            return None
        return {
            "alpha": float(res["coef"]["alpha"]),
            "se_alpha": float(res["se"]["alpha"]),
            "t_alpha": float(res["t"]["alpha"]),
            "p_alpha": float(res["p"]["alpha"]),
            "n_obs": res["n_obs"],
            "n_funds": res["n_clusters"],
            "df": res["df_dof"],
        }

    # Betas
    beta_rows_sg = []
    for (s, g), grp in cal.groupby(["Adj Strategy", "Grade"], dropna=False):
        res = ols_cluster_robust(grp, "omega", ["msci_ret_q", "msci_ret_q_lag1"], "FundID")
        if res is None:
            continue
        n_funds = res["n_clusters"]
        usable = (n_funds >= min_clusters_for_inference)

        b0 = float(res["coef"]["msci_ret_q"])
        b1 = float(res["coef"]["msci_ret_q_lag1"])
        p0 = float(res["p"]["msci_ret_q"])
        p1 = float(res["p"]["msci_ret_q_lag1"])
        use_beta = bool(usable and ((p0 < alpha_level) or (p1 < alpha_level)))

        beta_rows_sg.append({
            "Adj Strategy": s, "Grade": g,
            "b0": b0, "b1": b1,
            "p_b0": p0, "p_b1": p1,
            "n_obs": res["n_obs"], "n_funds": n_funds,
            "use_beta": use_beta
        })
    beta_sg = pd.DataFrame(beta_rows_sg)

    beta_rows_s = []
    for s, grp in cal.groupby(["Adj Strategy"], dropna=False):
        res = ols_cluster_robust(grp, "omega", ["msci_ret_q", "msci_ret_q_lag1"], "FundID")
        if res is None:
            continue
        n_funds = res["n_clusters"]
        usable = (n_funds >= min_clusters_for_inference)

        b0 = float(res["coef"]["msci_ret_q"])
        b1 = float(res["coef"]["msci_ret_q_lag1"])
        p0 = float(res["p"]["msci_ret_q"])
        p1 = float(res["p"]["msci_ret_q_lag1"])
        use_beta = bool(usable and ((p0 < alpha_level) or (p1 < alpha_level)))

        beta_rows_s.append({
            "Adj Strategy": s, "b0": b0, "b1": b1,
            "p_b0": p0, "p_b1": p1,
            "n_obs": res["n_obs"], "n_funds": n_funds,
            "use_beta": use_beta
        })
    beta_s = pd.DataFrame(beta_rows_s)

    res_g = ols_cluster_robust(cal, "omega", ["msci_ret_q", "msci_ret_q_lag1"], "FundID")
    if res_g is None:
        raise ValueError("NAV Logic inline: global beta regression failed.")
    b0_g = float(res_g["coef"]["msci_ret_q"])
    b1_g = float(res_g["coef"]["msci_ret_q_lag1"])

    beta_sg_use = beta_sg.loc[beta_sg["use_beta"]].set_index(["Adj Strategy", "Grade"])[["b0", "b1"]].to_dict("index")
    beta_s_use = beta_s.loc[beta_s["use_beta"]].set_index(["Adj Strategy"])[["b0", "b1"]].to_dict("index")

    def get_betas(strategy, grade):
        k = (strategy, grade)
        if k in beta_sg_use:
            d = beta_sg_use[k]
            return float(d["b0"]), float(d["b1"]), "sg_sig"
        if strategy in beta_s_use:
            d = beta_s_use[strategy]
            return float(d["b0"]), float(d["b1"]), "s_sig"
        return float(b0_g), float(b1_g), "global"

    # Alpha
    cal2 = cal.copy()
    b0_used = []
    b1_used = []
    for _, r in cal2.iterrows():
        b0, b1, _ = get_betas(r["Adj Strategy"], r["Grade"])
        b0_used.append(b0); b1_used.append(b1)
    cal2["b0_used"] = b0_used
    cal2["b1_used"] = b1_used
    cal2["omega_adj"] = cal2["omega"] - cal2["b0_used"]*cal2["msci_ret_q"] - cal2["b1_used"]*cal2["msci_ret_q_lag1"]

    alpha_rows_sga = []
    for (s, g, a), grp in cal2.groupby(["Adj Strategy","Grade","AgeBucket"], dropna=False):
        st = cluster_mean_stats(grp, "omega_adj", "FundID")
        if st is None:
            continue
        use_alpha = bool((st["n_funds"] >= min_clusters_for_inference) and (st["p_alpha"] < alpha_level))
        alpha_rows_sga.append({"Adj Strategy": s, "Grade": g, "AgeBucket": a, **st, "use_alpha": use_alpha})
    alpha_sga = pd.DataFrame(alpha_rows_sga)

    alpha_rows_sg = []
    for (s, g), grp in cal2.groupby(["Adj Strategy","Grade"], dropna=False):
        st = cluster_mean_stats(grp, "omega_adj", "FundID")
        if st is None:
            continue
        use_alpha = bool((st["n_funds"] >= min_clusters_for_inference) and (st["p_alpha"] < alpha_level))
        alpha_rows_sg.append({"Adj Strategy": s, "Grade": g, **st, "use_alpha": use_alpha})
    alpha_sg = pd.DataFrame(alpha_rows_sg)

    alpha_rows_s = []
    for s, grp in cal2.groupby(["Adj Strategy"], dropna=False):
        st = cluster_mean_stats(grp, "omega_adj", "FundID")
        if st is None:
            continue
        use_alpha = bool((st["n_funds"] >= min_clusters_for_inference) and (st["p_alpha"] < alpha_level))
        alpha_rows_s.append({"Adj Strategy": s, **st, "use_alpha": use_alpha})
    alpha_s = pd.DataFrame(alpha_rows_s)

    st_g = cluster_mean_stats(cal2, "omega_adj", "FundID")
    alpha_global = float(st_g["alpha"]) if st_g else 0.0

    alpha_sga_use = alpha_sga.loc[alpha_sga["use_alpha"]].set_index(["Adj Strategy","Grade","AgeBucket"])["alpha"].to_dict()
    alpha_sg_use  = alpha_sg.loc[alpha_sg["use_alpha"]].set_index(["Adj Strategy","Grade"])["alpha"].to_dict()
    alpha_s_use   = alpha_s.loc[alpha_s["use_alpha"]].set_index(["Adj Strategy"])["alpha"].to_dict()

    def get_alpha(strategy, grade, age_bucket):
        k = (strategy, grade, age_bucket)
        if k in alpha_sga_use:
            return float(alpha_sga_use[k]), "sga_sig"
        k2 = (strategy, grade)
        if k2 in alpha_sg_use:
            return float(alpha_sg_use[k2]), "sg_sig"
        if strategy in alpha_s_use:
            return float(alpha_s_use[strategy]), "s_sig"
        return float(alpha_global), "global"

    # Sigma
    resid = []
    for _, r in cal.iterrows():
        b0, b1, _ = get_betas(r["Adj Strategy"], r["Grade"])
        a, _ = get_alpha(r["Adj Strategy"], r["Grade"], r["AgeBucket"])
        pred = a + b0*r["msci_ret_q"] + b1*r["msci_ret_q_lag1"]
        resid.append(float(r["omega"] - pred))

    cal_res = cal.copy()
    cal_res["resid"] = resid

    sigma_sg = (
        cal_res.groupby(["Adj Strategy","Grade"], dropna=False)
               .agg(n_obs=("resid","size"),
                    sigma=("resid", lambda x: float(np.std(x, ddof=1)) if len(x) > 2 else 0.10))
               .reset_index()
    )
    sigma_global = float(np.std(cal_res["resid"], ddof=1))
    sigma_global = max(sigma_global, 0.02)

    sigma_sg_map = sigma_sg.set_index(["Adj Strategy","Grade"])[["sigma","n_obs"]].to_dict("index")

    def get_sigma(strategy, grade):
        k = (strategy, grade)
        if k in sigma_sg_map:
            s = float(sigma_sg_map[k]["sigma"])
            n = float(sigma_sg_map[k]["n_obs"])
            w = n / (n + SIGMA_SHRINK_K)
            return float(w*s + (1.0-w)*sigma_global), "sg_shrunk"
        return float(sigma_global), "global"

    # NAV_start imputation
    hist_upto = df.loc[df["quarter_end"] <= start_qe].copy()
    if hist_upto.empty:
        raise ValueError("NAV Logic inline: no data at or before start quarter.")

    hist_upto = hist_upto.sort_values(["FundID","quarter_end"])
    base_rows = hist_upto.groupby("FundID").tail(1).copy()

    base_rows["ever_overran"] = base_rows["FundID"].map(ever_overran_map).fillna(False)

    caps = []
    for _, r in base_rows.iterrows():
        planned = r["planned_end_qe"]
        if pd.isna(planned):
            caps.append(future_qe_max)
            continue
        if bool(r["ever_overran"]):
            avg_over = float(avg_overrun_by_strategy.get(r["Adj Strategy"], 0.0))
            caps.append(add_quarters(planned, avg_over))
        else:
            caps.append(planned)
    base_rows["cap_qe"] = caps
    base_rows["AgeBucket"] = pd.cut(base_rows["Fund_Age_Quarters"], bins=AGE_BINS_Q, labels=AGE_LABELS)

    hist_upto["draw_cum"] = hist_upto.groupby("FundID")[DRAW_COL].cumsum()
    if "draw_cum" not in base_rows.columns:
        base_rows = base_rows.merge(
            hist_upto.groupby("FundID")["draw_cum"].last().reset_index(),
            on="FundID",
            how="left"
        )

    tmp = hist_upto.copy()
    tmp["AgeBucket"] = pd.cut(tmp["Fund_Age_Quarters"], bins=AGE_BINS_Q, labels=AGE_LABELS)

    tmp["ratio_nav_draw"] = np.where(
        (tmp[NAV_COL].notna()) & (tmp[NAV_COL].abs() > NAV_EPS) & (tmp["draw_cum"] > DRAW_EPS),
        tmp[NAV_COL] / tmp["draw_cum"],
        np.nan
    )
    tmp["ratio_nav_size"] = np.where(
        (tmp[NAV_COL].notna()) & (tmp[NAV_COL].abs() > NAV_EPS) & (tmp[SIZE_COL] > SIZE_EPS),
        tmp[NAV_COL] / tmp[SIZE_COL],
        np.nan
    )

    tmp["log_ratio_nav_draw"] = np.log(tmp["ratio_nav_draw"])
    tmp["log_ratio_nav_size"] = np.log(tmp["ratio_nav_size"])
    tmp.loc[~np.isfinite(tmp["log_ratio_nav_draw"]), "log_ratio_nav_draw"] = np.nan
    tmp.loc[~np.isfinite(tmp["log_ratio_nav_size"]), "log_ratio_nav_size"] = np.nan

    ratio_key = ["Adj Strategy","Grade","AgeBucket"]

    def fit_lognorm(df_in: pd.DataFrame, col: str) -> pd.Series:
        g = df_in[col].dropna()
        if len(g) < MIN_OBS_RATIO:
            return pd.Series({"mu": np.nan, "sig": np.nan, "n": len(g)})
        return pd.Series({"mu": float(g.mean()), "sig": float(g.std(ddof=1)), "n": len(g)})

    ratio_draw = tmp.groupby(ratio_key, dropna=False).apply(lambda g: fit_lognorm(g, "log_ratio_nav_draw")).reset_index()
    ratio_size = tmp.groupby(ratio_key, dropna=False).apply(lambda g: fit_lognorm(g, "log_ratio_nav_size")).reset_index()

    gdraw = ratio_draw.dropna(subset=["mu","sig"])
    gsize = ratio_size.dropna(subset=["mu","sig"])
    fallback_draw = {"mu": float(gdraw["mu"].median()) if len(gdraw) else 0.0,
                     "sig": float(gdraw["sig"].median()) if len(gdraw) else 0.75}
    fallback_size = {"mu": float(gsize["mu"].median()) if len(gsize) else -2.0,
                     "sig": float(gsize["sig"].median()) if len(gsize) else 0.75}

    ratio_draw_map = ratio_draw.set_index(ratio_key)[["mu","sig","n"]].to_dict("index")
    ratio_size_map = ratio_size.set_index(ratio_key)[["mu","sig","n"]].to_dict("index")

    def lookup_ratio(map_, strategy, grade, age_bucket, fallback):
        k = (strategy, grade, age_bucket)
        if k in map_:
            d = map_[k]
            if pd.notna(d["mu"]) and pd.notna(d["sig"]) and d["n"] >= MIN_OBS_RATIO:
                return float(d["mu"]), float(d["sig"]), "bucket"
        return float(fallback["mu"]), float(fallback["sig"]), "global"

    rng_init = np.random.default_rng(2025)
    base_rows["NAV_start"] = base_rows[NAV_COL]
    base_rows["NAV_start_source"] = "observed"

    for idx, r in base_rows.iterrows():
        nav_obs = r["NAV_start"]
        if pd.notna(nav_obs) and abs(nav_obs) > NAV_EPS:
            continue

        draw_cum = r.get("draw_cum", 0.0)
        size = r.get(SIZE_COL, 0.0)
        draw_cum = 0.0 if pd.isna(draw_cum) else float(draw_cum)
        size = 0.0 if pd.isna(size) else float(size)

        strategy = r["Adj Strategy"]
        grade = r.get("AssignedGrade", r["Grade"])
        if pd.isna(grade):
            grade = r["Grade"]
        age_bucket = r["AgeBucket"]

        if draw_cum > DRAW_EPS:
            mu, sig, src = lookup_ratio(ratio_draw_map, strategy, grade, age_bucket, fallback_draw)
            ratio = float(np.exp(mu + sig * rng_init.standard_normal()))
            ratio = float(np.clip(ratio, 0.05, 5.0))
            base_rows.at[idx, "NAV_start"] = ratio * draw_cum
            base_rows.at[idx, "NAV_start_source"] = f"imputed_draw_{src}"
        else:
            base_rows.at[idx, "NAV_start"] = 0.0
            base_rows.at[idx, "NAV_start_source"] = "imputed_zero_nodraw"

    base_rows["NAV_start"] = pd.to_numeric(base_rows["NAV_start"], errors="coerce").fillna(0.0)

    # Grade transitions (yearly)
    GRADE_STATES = ["A","B","C","D"]
    p1_all_path = os.path.join(data_dir, "grade_transition_1y_all.csv")
    p1_pe_path  = os.path.join(data_dir, "grade_transition_1y_pe.csv")
    p1_vc_path  = os.path.join(data_dir, "grade_transition_1y_vc.csv")
    P1_ALL = pd.read_csv(p1_all_path, index_col=0) if os.path.exists(p1_all_path) else None
    P1_PE  = pd.read_csv(p1_pe_path, index_col=0) if os.path.exists(p1_pe_path) else None
    P1_VC  = pd.read_csv(p1_vc_path, index_col=0) if os.path.exists(p1_vc_path) else None

    def _row_norm_df(P):
        P = P.reindex(index=GRADE_STATES, columns=GRADE_STATES).fillna(0.0).clip(lower=0.0)
        rs = P.sum(axis=1).replace(0.0, 1.0)
        return P.div(rs, axis=0)

    if P1_ALL is not None: P1_ALL = _row_norm_df(P1_ALL)
    if P1_PE  is not None: P1_PE  = _row_norm_df(P1_PE)
    if P1_VC  is not None: P1_VC  = _row_norm_df(P1_VC)

    def build_yearly_transition_from_data(df_in, strategy=None):
        d = df_in[["FundID","quarter_end","Grade","Adj Strategy"]].copy()
        if strategy is not None:
            d = d[d["Adj Strategy"] == strategy]
        d = d.dropna(subset=["FundID","quarter_end","Grade"])
        d["Grade"] = d["Grade"].astype(str).str.strip()
        d = d[d["Grade"].isin(GRADE_STATES)]
        d = d.sort_values(["FundID","quarter_end"])

        transitions = []
        for _, g in d.groupby("FundID", sort=False):
            grades = g["Grade"].tolist()
            if len(grades) < 5:
                continue
            yearly = grades[::4]
            if len(yearly) < 2:
                continue
            transitions.extend(zip(yearly[:-1], yearly[1:]))

        if not transitions:
            counts = pd.DataFrame(0.0, index=GRADE_STATES, columns=GRADE_STATES)
            probs = pd.DataFrame(np.eye(4), index=GRADE_STATES, columns=GRADE_STATES)
            return counts, probs, 0

        counts = pd.crosstab(
            [a for a, _ in transitions],
            [b for _, b in transitions]
        ).reindex(index=GRADE_STATES, columns=GRADE_STATES, fill_value=0).astype(float)

        probs = counts.div(counts.sum(axis=1).replace(0.0, 1.0), axis=0)
        return counts, probs, len(transitions)

    all_counts, all_probs, all_n = build_yearly_transition_from_data(df, strategy=None)
    pe_counts, pe_probs, pe_n = build_yearly_transition_from_data(df, strategy="Private Equity")
    vc_counts, vc_probs, vc_n = build_yearly_transition_from_data(df, strategy="Venture Capital")

    def get_transition_matrix(strategy):
        if strategy == "Private Equity" and P1_PE is not None:
            return P1_PE, "PE_1Y"
        if strategy == "Venture Capital" and P1_VC is not None:
            return P1_VC, "VC_1Y"
        if P1_ALL is not None:
            return P1_ALL, "ALL_1Y"
        if strategy == "Private Equity" and pe_n > 0:
            return pe_probs, "PE_DATA"
        if strategy == "Venture Capital" and vc_n > 0:
            return vc_probs, "VC_DATA"
        if all_n > 0:
            return all_probs, "ALL_DATA"
        return pd.DataFrame(np.eye(4), index=GRADE_STATES, columns=GRADE_STATES), "IDENTITY"

    def sample_next_grade(curr_grade, P_df, rng):
        if curr_grade not in GRADE_STATES:
            curr_grade = "D"
        row = P_df.loc[curr_grade].values.astype(float)
        return str(rng.choice(GRADE_STATES, p=row))

    # Projection loop (omega only)
    omega_rows = []
    sim_ids = [1]
    for sim_id in sim_ids:
        rng = np.random.default_rng(seed + int(sim_id))
        msci_future_use = msci_future_map[sim_id]
        for _, r in base_rows.iterrows():
            fund_id = r["FundID"]
            age0 = int(r["Fund_Age_Quarters"]) if pd.notna(r["Fund_Age_Quarters"]) else 0
            strategy = r["Adj Strategy"]
            grade = r["Grade"] if pd.notna(r["Grade"]) else "D"
            cap_qe = r["cap_qe"]
            if pd.isna(cap_qe):
                cap_qe = msci_future_use["quarter_end"].iloc[-1]

            for step, (qe, msci_r, msci_r_lag1) in enumerate(
                zip(msci_future_use["quarter_end"], msci_future_use["msci_ret_q"], msci_future_use["msci_ret_q_lag1"]),
                start=1
            ):
                if qe > cap_qe:
                    break

                msci_r_lag1 = 0.0 if pd.isna(msci_r_lag1) else float(msci_r_lag1)
                age = age0 + step
                age_bucket = pd.cut(pd.Series([age]), bins=AGE_BINS_Q, labels=AGE_LABELS).iloc[0]

                prev_grade = grade
                if step % 4 == 0:
                    P, _ = get_transition_matrix(strategy)
                    grade = sample_next_grade(grade, P, rng)

                b0, b1, _ = get_betas(strategy, grade)
                alpha, _ = get_alpha(strategy, grade, age_bucket)
                sigma, _ = get_sigma(strategy, grade)

                eps = rng.standard_normal()
                omega = alpha + b0*float(msci_r) + b1*msci_r_lag1 + sigma*eps
                omega += float(GRADE_OMEGA_BIAS.get(grade, 0.0))
                if not np.isfinite(omega):
                    omega = 0.0
                omega = float(np.clip(omega, -OMEGA_CLIP, OMEGA_CLIP))

                omega_rows.append({
                    "sim_id": int(sim_id),
                    "FundID": fund_id,
                    "quarter_end": qe,
                    "step_q": step,
                    "msci_ret_q": float(msci_r),
                    "msci_ret_q_lag1": float(msci_r_lag1),
                    "omega": float(omega),
                    "Fund_Age_Quarters": int(age),
                    "Adj Strategy": strategy,
                    "Grade_prev": prev_grade,
                    "Grade": grade,
                    "AgeBucket": age_bucket,
                    "cap_qe": cap_qe,
                })

    omega_proj = pd.DataFrame(omega_rows)
    navstart = base_rows[[
        "FundID","Adj Strategy","Grade","Fund_Age_Quarters","NAV_start","NAV_start_source","cap_qe"
    ]].copy()

    return omega_proj, navstart

# -----------------------------
# NAV Logic inputs (optional)
# -----------------------------

def find_omega_file(data_dir: str, start_qe: Optional[pd.Timestamp] = None,
                    end_qe: Optional[pd.Timestamp] = None) -> str:
    cands = glob.glob(os.path.join(data_dir, "omega_projection_sota_*.parquet")) + \
            glob.glob(os.path.join(data_dir, "omega_projection_sota_*.csv"))
    if not cands:
        raise FileNotFoundError("No omega_projection_sota_* file found. Run NAV Logic first.")

    if start_qe is None or end_qe is None:
        cands.sort(key=os.path.getmtime, reverse=True)
        return cands[0]

    best_path = None
    best_overlap = -1
    best_mtime = -1
    ranges = []

    for p in cands:
        try:
            if p.lower().endswith(".parquet"):
                df = pd.read_parquet(p, columns=["quarter_end"])
            else:
                df = pd.read_csv(p, usecols=["quarter_end"])
            q = pd.to_datetime(df["quarter_end"], errors="coerce").dt.to_period("Q").dt.to_timestamp("Q")
            q = q.dropna()
            if q.empty:
                continue
            qmin = q.min()
            qmax = q.max()
            ranges.append((p, qmin, qmax))

            ov_start = max(start_qe, qmin)
            ov_end = min(end_qe, qmax)
            if ov_end < ov_start:
                overlap = -1
            else:
                overlap = len(pd.period_range(ov_start, ov_end, freq="Q"))

            mtime = os.path.getmtime(p)
            if overlap > best_overlap or (overlap == best_overlap and mtime > best_mtime):
                best_overlap = overlap
                best_mtime = mtime
                best_path = p
        except Exception:
            continue

    if best_path is None or best_overlap <= 0:
        msg = "No omega_projection_sota_* file overlaps the test window."
        if ranges:
            msg += " Available ranges:\n" + "\n".join([f"- {os.path.basename(p)}: {a.date()} to {b.date()}" for p, a, b in ranges])
        raise ValueError(msg)

    return best_path


def find_navstart_file(data_dir: str, year: int, quarter: str) -> str:
    cands = [
        os.path.join(data_dir, f"nav_start_sota_{year}_{quarter}.parquet"),
        os.path.join(data_dir, f"nav_start_sota_{year}_{quarter}.csv"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    c2 = glob.glob(os.path.join(data_dir, "nav_start_sota_*.parquet")) + \
         glob.glob(os.path.join(data_dir, "nav_start_sota_*.csv"))
    if not c2:
        raise FileNotFoundError("No nav_start_sota_* file found. Run NAV Logic first.")
    c2.sort(key=os.path.getmtime, reverse=True)
    return c2[0]


# -----------------------------
# Main backtest
# -----------------------------

def main():
    t0 = time.perf_counter()
    print("=== Backtest config ===")
    print("RUN_NAV_LOGIC_INLINE:", RUN_NAV_LOGIC_INLINE)
    print("NAV_LOGIC_MSCI_MODE:", NAV_LOGIC_MSCI_MODE)
    print("USE_NAV_PROJECTIONS:", USE_NAV_PROJECTIONS)
    print("RUN_CONDITIONAL:", RUN_CONDITIONAL, "RUN_UNCONDITIONAL:", RUN_UNCONDITIONAL)
    print("CALIBRATION_BUCKET_MODE:", CALIBRATION_BUCKET_MODE, "REPORT_BUCKET_MODE:", REPORT_BUCKET_MODE)
    print("=======================")
    year = int(input("Enter year (e.g. 2025): ").strip())
    quarter = input("Enter quarter (Q1, Q2, Q3, Q4): ").strip().upper()

    train_year = int(input("Train end year (e.g. 2018): ").strip())
    train_quarter = input("Train end quarter (Q1, Q2, Q3, Q4): ").strip().upper()

    test_year = input("Test end year (blank => max in data): ").strip()
    test_quarter = ""
    if test_year:
        test_year = int(test_year)
        test_quarter = input("Test end quarter (Q1, Q2, Q3, Q4): ").strip().upper()

    n_sims = int(input("MC simulations [500]: ").strip() or "500")
    seed = int(input("Random seed [1234]: ").strip() or "1234")
    rho_event = float(input("Copula correlation events rho_event [0.25]: ").strip() or "0.25")
    rho_size = float(input("Copula correlation sizes rho_size [0.15]: ").strip() or "0.15")
    scenario_uncond = input("Unconditional scenario (bullish/neutral/bearish) [neutral]: ").strip().lower() or "neutral"
    tilt_strength = float(input("Unconditional tilt strength [1.2]: ").strip() or "1.2")

    rho_event = float(np.clip(rho_event, 0.0, 0.999))
    rho_size = float(np.clip(rho_size, 0.0, 0.999))

    BASE_DIR = os.environ.get(
        "EQUITY_BASE_DIR",
        os.path.join("C:", "Users", os.environ.get("USERNAME", ""), "Documents", "Equity")
    )
    if not os.path.exists(BASE_DIR):
        BASE_DIR = os.path.abspath(os.getcwd())

    HOME = os.path.join(BASE_DIR, f"{year}_{quarter}")
    DATA_DIR = os.path.join(HOME, "data")

    data_path_parquet = os.path.join(DATA_DIR, "data.parquet")
    data_path_csv = os.path.join(DATA_DIR, "data.csv")
    kmp_path_parquet = os.path.join(DATA_DIR, "kmp.parquet")
    kmp_path_csv = os.path.join(DATA_DIR, "kmp.csv")

    if os.path.exists(data_path_parquet):
        data = pd.read_parquet(data_path_parquet)
    elif os.path.exists(data_path_csv):
        data = pd.read_csv(data_path_csv)
    else:
        raise FileNotFoundError(f"Missing data.parquet or data.csv in {DATA_DIR}")

    if os.path.exists(kmp_path_parquet):
        kmp = pd.read_parquet(kmp_path_parquet)
    elif os.path.exists(kmp_path_csv):
        kmp = pd.read_csv(kmp_path_csv)
    else:
        raise FileNotFoundError("Missing kmp.parquet or kmp.csv")

    # normalize column names (trim/collapse spaces) to avoid hidden KeyErrors
    def _norm_col(c):
        if not isinstance(c, str):
            return c
        return " ".join(c.strip().split())

    data.columns = [_norm_col(c) for c in data.columns]
    kmp.columns = [_norm_col(c) for c in kmp.columns]

    req = [
        "FundID", "Adj Strategy", "Grade", "Fund_Age_Quarters",
        "Year of Transaction Date", "Quarter of Transaction Date",
        "Adj Drawdown EUR", "Adj Repayment EUR",
        "NAV Adjusted EUR", "Recallable",
        "Planned end date with add. years as per legal doc",
    ]
    missing = [c for c in req if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    # Commitment column aliases (some datasets use different naming)
    def _norm_key(s: str) -> str:
        return " ".join(s.strip().lower().replace("_", " ").split())

    col_by_key = {_norm_key(c): c for c in data.columns if isinstance(c, str)}

    commit_level_col = None
    commit_flow_col = None
    for c in ["Commitment_Level", "Commitment Level", "commitment level"]:
        key = _norm_key(c)
        if key in col_by_key:
            commit_level_col = col_by_key[key]
            break
    for c in ["Commitment EUR", "Commitment", "commitment eur", "commitment"]:
        key = _norm_key(c)
        if key in col_by_key:
            commit_flow_col = col_by_key[key]
            break
    if commit_level_col is None and commit_flow_col is None:
        raise ValueError("Missing commitment column: expected one of Commitment_Level / Commitment Level / Commitment EUR / Commitment")

    # Clean numeric
    num_cols = ["Adj Drawdown EUR", "Adj Repayment EUR", "NAV Adjusted EUR", "Recallable", "Fund_Age_Quarters"]
    if commit_level_col is not None:
        num_cols.append(commit_level_col)
    if commit_flow_col is not None:
        num_cols.append(commit_flow_col)
    for c in num_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data["Adj Drawdown EUR"] = data["Adj Drawdown EUR"].fillna(0.0).clip(lower=0.0)
    data["Adj Repayment EUR"] = data["Adj Repayment EUR"].fillna(0.0).clip(lower=0.0)
    data["Recallable"] = data["Recallable"].fillna(0.0).clip(lower=0.0)
    data["NAV Adjusted EUR"] = data["NAV Adjusted EUR"].fillna(0.0).clip(lower=0.0)
    data["Fund_Age_Quarters"] = pd.to_numeric(data["Fund_Age_Quarters"], errors="coerce")

    q_year = pd.to_numeric(data["Year of Transaction Date"], errors="coerce")
    q_qtr = pd.to_numeric(data["Quarter of Transaction Date"], errors="coerce")
    if q_year.isna().any() or q_qtr.isna().any():
        raise ValueError("Year/Quarter of Transaction Date contains non-numeric values.")

    data["quarter_end"] = pd.PeriodIndex(
        q_year.astype("int64").astype(str) + "Q" + q_qtr.astype("int64").astype(str),
        freq="Q"
    ).to_timestamp("Q")
    data = data.sort_values(["FundID", "quarter_end"]).reset_index(drop=True)

    # First closing date (age baseline) if available; fallback to first observed quarter_end
    first_close_col = None
    for key, col in col_by_key.items():
        if "first closing" in key:
            first_close_col = col
            break
    if first_close_col is not None:
        data["first_close_qe"] = pd.to_datetime(data[first_close_col], errors="coerce").dt.to_period("Q").dt.to_timestamp("Q")
    else:
        data["first_close_qe"] = pd.NaT
    fc_map = data.groupby("FundID")["first_close_qe"].min()
    fallback_fc = data.groupby("FundID")["quarter_end"].min()
    fc_map = fc_map.fillna(fallback_fc)
    data["first_close_qe"] = data["FundID"].map(fc_map)
    # Age in quarters since first close (Q2 for vintage handled upstream in data prep)
    p_qe = pd.PeriodIndex(data["quarter_end"], freq="Q")
    p_fc = pd.PeriodIndex(data["first_close_qe"], freq="Q")
    data["age_q_fc"] = (p_qe.astype("int64") - p_fc.astype("int64")).astype(int)
    data.loc[data["age_q_fc"] < 0, "age_q_fc"] = 0
    age_from_fund = pd.to_numeric(data["Fund_Age_Quarters"], errors="coerce")
    if str(AGE_SOURCE).lower() in {"fund_age", "fund_age_quarters", "fund"}:
        data["age_q_model"] = age_from_fund
        data.loc[data["age_q_model"].isna(), "age_q_model"] = data.loc[data["age_q_model"].isna(), "age_q_fc"]
    else:
        data["age_q_model"] = data["age_q_fc"]
    data["age_q_model"] = pd.to_numeric(data["age_q_model"], errors="coerce").fillna(0.0)
    data["Fund_Age_Quarters"] = data["Fund_Age_Quarters"].fillna(0.0)

    train_end_qe = quarter_end_from_year_quarter(train_year, train_quarter)
    test_start_qe = add_quarters(train_end_qe, 1)

    if test_year:
        test_end_qe = quarter_end_from_year_quarter(test_year, test_quarter)
    else:
        test_end_qe = data["quarter_end"].max()

    if test_end_qe < test_start_qe:
        raise ValueError("Test end is before test start. Check dates.")

    test_quarters = quarter_range(test_start_qe, test_end_qe)
    if not test_quarters:
        raise ValueError("No test quarters found. Check dates.")

    # Optional: load NAV Logic omega + nav_start for aligned NAV path
    omega_df = None
    navstart = None
    omega_map = {}
    grade_map = {}
    age_map = {}
    strategy_map = {}
    msci_map_sim = {}
    msci_stats_by_sim = {}
    sim_ids = None
    nav_start_by_fund = {}
    nav_cap_by_fund = {}
    nav_grade_by_fund = {}
    nav_age_by_fund = {}
    nav_strategy_by_fund = {}
    draw_cum_hist = {}

    if USE_NAV_PROJECTIONS:
        if RUN_NAV_LOGIC_INLINE:
            # Load MSCI (quarterly) for inline NAV Logic
            msci_path = os.path.join(DATA_DIR, "msci.xlsx")
            if not os.path.exists(msci_path):
                msci_path = os.path.join(DATA_DIR, "MSCI.xlsx")
            if not os.path.exists(msci_path):
                raise FileNotFoundError("Missing msci.xlsx / MSCI.xlsx in DATA_DIR")
            msci_q = load_msci_quarterly(msci_path)
            msci_q = msci_q.sort_values("quarter_end").reset_index(drop=True)
            msci_q["msci_ret_q_lag1"] = msci_q["msci_ret_q"].shift(1)

            # Build MSCI history + future for NAV Logic
            start_qe = train_end_qe
            msci_hist_inline = msci_q[msci_q["quarter_end"] <= start_qe].copy()
            last_hist_ret = msci_hist_inline["msci_ret_q"].tail(1)
            last_hist_ret_val = float(last_hist_ret.iloc[0]) if len(last_hist_ret) else 0.0
            if NAV_LOGIC_MSCI_MODE == "unconditional":
                rng_nav = np.random.default_rng(seed)
                msci_future = simulate_msci_path(
                    msci_q[msci_q["quarter_end"] <= start_qe],
                    start_qe=start_qe,
                    n_quarters=len(test_quarters),
                    scenario=scenario_uncond,
                    tilt_strength=tilt_strength,
                    rng=rng_nav,
                )
                msci_future = msci_future.sort_values("quarter_end").reset_index(drop=True)
            else:
                msci_future = msci_q[(msci_q["quarter_end"] > start_qe) &
                                     (msci_q["quarter_end"] <= test_end_qe)].copy()
                if len(msci_future) != len(test_quarters):
                    raise ValueError("MSCI history does not fully cover the backtest window.")
                msci_future = msci_future.sort_values("quarter_end").reset_index(drop=True)

            msci_future["msci_ret_q_lag1"] = msci_future["msci_ret_q"].shift(1)
            if len(msci_future):
                msci_future.loc[0, "msci_ret_q_lag1"] = last_hist_ret_val
            msci_future["msci_ret_q_lag1"] = msci_future["msci_ret_q_lag1"].fillna(0.0)

            omega_df, navstart = run_nav_logic_inline(
                data=data,
                msci_hist=msci_hist_inline,
                msci_future=msci_future,
                start_qe=start_qe,
                data_dir=DATA_DIR,
                alpha_level=NAV_LOGIC_ALPHA_LEVEL,
                min_clusters_for_inference=NAV_LOGIC_MIN_CLUSTERS,
                seed=seed,
            )
        else:
            omega_path = find_omega_file(DATA_DIR, test_start_qe, test_end_qe)
            navstart_path = find_navstart_file(DATA_DIR, train_year, train_quarter)
            print("Using omega file:", omega_path)
            print("Using nav_start file:", navstart_path)

            omega_df = pd.read_parquet(omega_path) if omega_path.lower().endswith(".parquet") else pd.read_csv(omega_path)
            navstart = pd.read_parquet(navstart_path) if navstart_path.lower().endswith(".parquet") else pd.read_csv(navstart_path)

        need_omega = {"FundID", "quarter_end", "omega"}
        if not need_omega.issubset(omega_df.columns):
            raise ValueError(f"omega_projection must contain columns: {need_omega}")
        need_ns = {"FundID", "NAV_start", "cap_qe"}
        if not need_ns.issubset(navstart.columns):
            raise ValueError(f"nav_start must contain columns: {need_ns}")

        omega_df = omega_df.copy()
        omega_df["quarter_end"] = pd.to_datetime(omega_df["quarter_end"]).dt.to_period("Q").dt.to_timestamp("Q")
        omega_df["omega"] = pd.to_numeric(omega_df["omega"], errors="coerce").fillna(0.0)
        if "msci_ret_q" in omega_df.columns:
            omega_df["msci_ret_q"] = pd.to_numeric(omega_df["msci_ret_q"], errors="coerce").fillna(0.0)
        if "Fund_Age_Quarters" in omega_df.columns:
            omega_df["Fund_Age_Quarters"] = pd.to_numeric(omega_df["Fund_Age_Quarters"], errors="coerce")
        if "Grade" in omega_df.columns:
            omega_df["Grade"] = omega_df["Grade"].astype(str).str.strip()
            omega_df.loc[~omega_df["Grade"].isin(GRADE_STATES), "Grade"] = "D"
        if "sim_id" not in omega_df.columns:
            omega_df["sim_id"] = 1
        omega_df["sim_id"] = pd.to_numeric(omega_df["sim_id"], errors="coerce").fillna(1).astype(int)

        # require omega to fully cover the test window
        omega_min = omega_df["quarter_end"].min()
        omega_max = omega_df["quarter_end"].max()
        if omega_min > test_start_qe or omega_max < test_end_qe:
            raise ValueError(
                "omega_projection does not fully cover the test window. "
                f"omega range: {omega_min.date()} to {omega_max.date()}, "
                f"test window: {test_start_qe.date()} to {test_end_qe.date()}. "
                "Run NAV Logic for the backtest window and re-run."
            )

        # restrict to test window
        omega_df = omega_df[(omega_df["quarter_end"] >= test_start_qe) &
                            (omega_df["quarter_end"] <= test_end_qe)].copy()

        # ensure omega has every quarter in the test window
        omega_quarters = set(omega_df["quarter_end"].unique())
        missing_q = [qe for qe in test_quarters if qe not in omega_quarters]
        if missing_q:
            miss = ", ".join([d.strftime("%Y-%m-%d") for d in missing_q[:8]])
            extra = "" if len(missing_q) <= 8 else f" (+{len(missing_q) - 8} more)"
            raise ValueError(
                "omega_projection is missing quarters inside the test window: "
                f"{miss}{extra}. Align omega projection or adjust test dates."
            )

        sim_ids = sorted(omega_df["sim_id"].unique().tolist())

        navstart = navstart.copy()
        navstart["NAV_start"] = pd.to_numeric(navstart["NAV_start"], errors="coerce").fillna(0.0).clip(lower=0.0)
        navstart["cap_qe"] = pd.to_datetime(navstart["cap_qe"], errors="coerce")

        nav_start_by_fund = navstart.set_index("FundID")["NAV_start"].to_dict()
        nav_cap_by_fund = navstart.set_index("FundID")["cap_qe"].to_dict()
        if "Grade" in navstart.columns:
            nav_grade_by_fund = navstart.set_index("FundID")["Grade"].astype(str).to_dict()
        if "Fund_Age_Quarters" in navstart.columns:
            nav_age_by_fund = pd.to_numeric(navstart.set_index("FundID")["Fund_Age_Quarters"], errors="coerce").to_dict()
        if "Adj Strategy" in navstart.columns:
            nav_strategy_by_fund = navstart.set_index("FundID")["Adj Strategy"].astype(str).to_dict()

        for sim_id, g in omega_df.groupby("sim_id"):
            key = list(zip(g["FundID"], g["quarter_end"]))
            omega_map[sim_id] = dict(zip(key, g["omega"]))
            if "Adj Strategy" in g.columns:
                strategy_map[sim_id] = dict(zip(key, g["Adj Strategy"].astype(str)))
            if "Grade" in g.columns:
                grade_map[sim_id] = dict(zip(key, g["Grade"]))
            if "Fund_Age_Quarters" in g.columns:
                age_map[sim_id] = dict(zip(key, g["Fund_Age_Quarters"]))
            if "msci_ret_q" in g.columns:
                msci_map_sim[sim_id] = dict(zip(g["quarter_end"], g["msci_ret_q"]))

        if "msci_ret_q" in omega_df.columns:
            msci_stats_by_sim = omega_df.groupby("sim_id")["msci_ret_q"].agg(["mean", "std"]).to_dict("index")

        # historical drawdowns up to first projection quarter (for DD_cum_commit init)
        draw_cum_hist = (data[data["quarter_end"] <= test_start_qe]
                         .groupby("FundID")["Adj Drawdown EUR"].sum()
                         .to_dict())

    if USE_NAV_PROJECTIONS and not draw_cum_hist:
        draw_cum_hist = (data[data["quarter_end"] <= test_start_qe]
                         .groupby("FundID")["Adj Drawdown EUR"].sum()
                         .to_dict())

    # MSCI data (load if not already loaded for inline NAV Logic)
    if "msci_q" not in locals():
        msci_path = os.path.join(DATA_DIR, "msci.xlsx")
        if not os.path.exists(msci_path):
            msci_path = os.path.join(DATA_DIR, "MSCI.xlsx")
        if not os.path.exists(msci_path):
            raise FileNotFoundError("Missing msci.xlsx / MSCI.xlsx in DATA_DIR")

        msci_q = load_msci_quarterly(msci_path)
        msci_q = msci_q.sort_values("quarter_end").reset_index(drop=True)
        msci_q["msci_ret_q_lag1"] = msci_q["msci_ret_q"].shift(1)

    # Train/test splits (initial)
    train = data[data["quarter_end"] <= train_end_qe].copy()
    test = data[data["quarter_end"] >= test_start_qe].copy()

    if train.empty or test.empty:
        raise ValueError("Train or test split is empty. Adjust dates.")

    # Commitment proxy (full history)
    if commit_level_col is not None:
        data["Commitment_Level"] = data[commit_level_col].fillna(0.0)
    else:
        data["Commitment_Level"] = data.groupby("FundID")[commit_flow_col].cumsum().fillna(0.0)
    fund_commit = data.groupby("FundID")["Commitment_Level"].max().fillna(0.0).to_dict()
    # Refresh train/test to include Commitment_Level
    train = data[data["quarter_end"] <= train_end_qe].copy()
    test = data[data["quarter_end"] >= test_start_qe].copy()
    if train.empty or test.empty:
        raise ValueError("Train or test split is empty after commitment prep. Adjust dates.")

    # Funds in test window
    test_funds = test["FundID"].unique().tolist()
    if USE_NAV_PROJECTIONS:
        nav_funds = set(nav_start_by_fund.keys())
        omega_funds = set(omega_df["FundID"].unique()) if omega_df is not None else set()
        test_funds = [fid for fid in test_funds if fid in nav_funds and fid in omega_funds]
        if not test_funds:
            raise ValueError("No overlap between test funds and NAV Logic projection inputs.")
        if sim_ids:
            if len(sim_ids) > 1:
                n_sims = len(sim_ids)
                print(f"Using {n_sims} NAV Logic sim_id(s) from omega projections.")
            else:
                print(f"Using single NAV Logic omega path; cashflow sims = {n_sims}.")

    # Planned end date (cap)
    data["planned_end_qe"] = pd.to_datetime(
        data["Planned end date with add. years as per legal doc"],
        errors="coerce"
    ).dt.to_period("Q").dt.to_timestamp("Q")

    cap_by_fund = (data.dropna(subset=["planned_end_qe"])
                   .sort_values(["FundID", "planned_end_qe"])
                   .groupby("FundID")["planned_end_qe"].last().to_dict())
    if USE_NAV_PROJECTIONS and nav_cap_by_fund:
        for fid, cap in nav_cap_by_fund.items():
            if pd.notna(cap):
                cap_by_fund[fid] = cap

    # Age buckets
    data["AgeBucket"] = pd.cut(data["age_q_model"], bins=AGE_BINS_Q, labels=AGE_LABELS)
    if "AgeBucket" not in train.columns:
        train["AgeBucket"] = pd.cut(train["age_q_model"], bins=AGE_BINS_Q, labels=AGE_LABELS)
    if "AgeBucket" not in test.columns:
        test["AgeBucket"] = pd.cut(test["age_q_model"], bins=AGE_BINS_Q, labels=AGE_LABELS)

    # Calibration bucketing (optionally collapse age buckets)
    if CALIBRATION_BUCKET_MODE == "strategy_grade":
        train["AgeBucketCalib"] = "ALL"
    else:
        train["AgeBucketCalib"] = train["AgeBucket"].astype(str)

    # Build MSCI lookup for conditional path
    msci_map = msci_q.set_index("quarter_end")["msci_ret_q"].to_dict()
    msci_lag_map = msci_q.set_index("quarter_end")["msci_ret_q_lag1"].to_dict()
    msci_hist = msci_q[msci_q["quarter_end"] <= train_end_qe].copy()
    msci_mu_all = float(msci_hist["msci_ret_q"].mean()) if len(msci_hist) else 0.0
    msci_sigma_all = float(msci_hist["msci_ret_q"].std(ddof=1)) if len(msci_hist) > 1 else 1.0
    msci_sigma_all = max(msci_sigma_all, 1e-6)

    # Ensure MSCI covers test quarters for conditional
    for qe in test_quarters:
        if qe not in msci_map:
            raise ValueError(f"Missing MSCI return for quarter {qe} in MSCI file.")

    # =============================
    # Calibrate omega model on train
    # =============================
    train = train.copy()
    train["nav_prev"] = train.groupby("FundID")["NAV Adjusted EUR"].shift(1)
    train["flow_net"] = train["Adj Drawdown EUR"] - train["Adj Repayment EUR"]
    m = train["nav_prev"].fillna(0.0) > 0
    train["omega"] = np.nan
    train.loc[m, "omega"] = ((train.loc[m, "NAV Adjusted EUR"] - train.loc[m, "nav_prev"]) - train.loc[m, "flow_net"]) / train.loc[m, "nav_prev"]
    train["omega"] = train["omega"].clip(lower=-OMEGA_CLIP, upper=OMEGA_CLIP)

    # attach MSCI to train
    train["msci_ret_q"] = train["quarter_end"].map(msci_map)
    train["msci_ret_q_lag1"] = train["quarter_end"].map(msci_lag_map)

    cal = train.dropna(subset=["omega", "msci_ret_q", "msci_ret_q_lag1"]).copy()
    cal["AgeBucket"] = cal["AgeBucket"].astype(str)
    if cal.empty:
        raise ValueError("No omega calibration data after filtering.")

    # Omega validation (train vs test, actual vs model input)
    def _omega_stats(df_in: pd.DataFrame, label: str):
        if df_in is None or df_in.empty:
            print(f"[omega] {label}: no data")
            return
        m = df_in["nav_prev"].abs() > NAV_EPS
        if "omega" not in df_in.columns:
            df_in = df_in.copy()
            df_in["flow_net"] = df_in["Adj Drawdown EUR"] - df_in["Adj Repayment EUR"]
            df_in["omega"] = np.nan
            df_in.loc[m, "omega"] = ((df_in.loc[m, "NAV Adjusted EUR"] - df_in.loc[m, "nav_prev"]) - df_in.loc[m, "flow_net"]) / df_in.loc[m, "nav_prev"]
        w = df_in.loc[m, "omega"].dropna()
        if w.empty:
            print(f"[omega] {label}: no omega values")
            return
        print(f"[omega] {label}: mean={w.mean():.4f} median={w.median():.4f} p10={w.quantile(0.1):.4f} p90={w.quantile(0.9):.4f} n={len(w)}")

    _omega_stats(train, "train_actual")
    _omega_stats(test, "test_actual")

    if USE_NAV_PROJECTIONS and omega_df is not None and not omega_df.empty:
        om = omega_df["omega"].dropna()
        if len(om):
            print(f"[omega] model_input (omega_df): mean={om.mean():.4f} median={om.median():.4f} p10={om.quantile(0.1):.4f} p90={om.quantile(0.9):.4f} n={len(om)}")
        else:
            print("[omega] model_input (omega_df): no omega values")

    get_betas, get_alpha, get_sigma = build_omega_models(cal)

    # =============================
    # Calibrate cashflow model on train
    # =============================
    # investment period calibration by strategy
    ip_by_strategy = {}
    df_ip = train[(train["Adj Drawdown EUR"] > 0) & train["Adj Strategy"].notna() & train["age_q_model"].notna()].copy()
    if len(df_ip):
        df_ip["age_q"] = pd.to_numeric(df_ip["age_q_model"], errors="coerce").round().astype("Int64")
        df_ip = df_ip[df_ip["age_q"].notna() & (df_ip["age_q"] >= 0)]
        for strat, g in df_ip.groupby("Adj Strategy", dropna=False):
            s = g.groupby("age_q")["Adj Drawdown EUR"].sum().sort_index()
            total = float(s.sum())
            if total <= 0:
                continue
            cum = s.cumsum()
            thr = IP_CUM_PCTL * total
            ip_q = int(cum.index[cum.values >= thr][0])
            ip_q = int(np.clip(ip_q, IP_Q_MIN, IP_Q_MAX))
            ip_by_strategy[strat] = ip_q

    # hazard dataset
    haz = train.copy()
    haz["age_q"] = pd.to_numeric(haz["age_q_model"], errors="coerce").round()
    haz = haz[haz["age_q"].notna()].copy()
    haz["age_q"] = haz["age_q"].astype(int)
    haz["log_nav_prev"] = np.log1p(haz["nav_prev"].abs().fillna(0.0))
    haz["ip_q"] = haz["Adj Strategy"].map(ip_by_strategy).fillna(IP_Q_DEFAULT).astype(int)
    draw_haz = haz.copy()
    if ENFORCE_IP_LIMITS:
        draw_haz = haz[haz["age_q"] <= haz["ip_q"]].copy()

    X_draw = build_feature_matrix(draw_haz, include_nav=False)
    X_rep = build_feature_matrix(haz, include_nav=True)

    cont_draw = ["age_q", "age_q2"]
    cont_rep = ["age_q", "age_q2", "log_nav_prev"]

    X_draw, draw_means, draw_stds = standardize_X(X_draw, cont_draw)
    X_rep, rep_means, rep_stds = standardize_X(X_rep, cont_rep)

    Y_draw = (draw_haz["Adj Drawdown EUR"] > 0).astype(int).to_numpy(dtype=float)
    Y_rep = (haz["Adj Repayment EUR"] > 0).astype(int).to_numpy(dtype=float)

    beta_draw = None
    beta_rep = None
    try:
        if USE_HAZARD_MODELS and len(X_draw) and len(Y_draw):
            beta_draw = fit_logit(X_draw.to_numpy(), Y_draw)
        if USE_HAZARD_MODELS and len(X_rep) and len(Y_rep):
            beta_rep = fit_logit(X_rep.to_numpy(), Y_rep)
    except Exception as e:
        print("Warning: hazard model fit failed, falling back to group means:", e)
        beta_draw = None
        beta_rep = None

    hazard_meta = {
        "draw_cols": list(X_draw.columns),
        "rep_cols": list(X_rep.columns),
        "draw_means": draw_means, "draw_stds": draw_stds,
        "rep_means": rep_means, "rep_stds": rep_stds,
    }

    # cap proxy for draw ratios
    if "Commitment_Level" not in train.columns:
        if commit_level_col is not None and commit_level_col in train.columns:
            train["Commitment_Level"] = train[commit_level_col].fillna(0.0)
        elif commit_flow_col is not None and commit_flow_col in train.columns:
            train["Commitment_Level"] = train.groupby("FundID")[commit_flow_col].cumsum().fillna(0.0)
        else:
            raise ValueError("Commitment_Level missing in train and no usable commitment column found.")
    cap_proxy_col = "Capacity" if "Capacity" in train.columns else None
    if cap_proxy_col is None:
        print("WARNING: data has no 'Capacity' column; draw calibration uses Commitment_Level proxy.")
    train["cap_proxy"] = pd.to_numeric(train[cap_proxy_col], errors="coerce").fillna(0.0) if cap_proxy_col else train["Commitment_Level"].fillna(0.0)
    train["draw_event"] = (train["Adj Drawdown EUR"] > 0).astype(int)
    train["rep_event"] = (train["Adj Repayment EUR"] > 0).astype(int)

    train["draw_ratio"] = np.where(train["cap_proxy"] > CAP_EPS, train["Adj Drawdown EUR"] / train["cap_proxy"], np.nan)
    train.loc[train["draw_ratio"] <= 0, "draw_ratio"] = np.nan

    train["rep_ratio"] = np.where(train["nav_prev"].abs() > NAV_EPS, train["Adj Repayment EUR"] / train["nav_prev"].abs(), np.nan)
    train.loc[train["rep_ratio"] <= 0, "rep_ratio"] = np.nan

    train["rc_given_rep_event"] = ((train["Adj Repayment EUR"] > 0) & (train["Recallable"] > 0)).astype(int)
    train["rc_ratio_given_rep"] = np.where(train["Adj Repayment EUR"] > 0, train["Recallable"] / train["Adj Repayment EUR"], np.nan)
    train.loc[train["rc_ratio_given_rep"] <= 0, "rc_ratio_given_rep"] = np.nan

    # Calibration tables
    group_keys = ["Adj Strategy", "Grade", "AgeBucketCalib"]
    rows = []
    for (s, g, a), grp in train.groupby(group_keys, dropna=False):
        p_draw = float(grp["draw_event"].mean()) if len(grp) else 0.0
        p_rep = float(grp["rep_event"].mean()) if len(grp) else 0.0
        rep_q = grp[grp["Adj Repayment EUR"] > 0]
        p_rc_given_rep = float(rep_q["rc_given_rep_event"].mean()) if len(rep_q) else 0.0
        stats_d = fit_lognormal_stats(grp["draw_ratio"], grp["FundID"])
        stats_r = fit_lognormal_stats(grp["rep_ratio"], grp["FundID"])
        stats_c = fit_lognormal_stats(rep_q["rc_ratio_given_rep"], rep_q["FundID"]) if len(rep_q) else {
            "mu": 0.0, "sig": SIGMA_FLOOR, "n": 0, "n_funds": 0, "ks_D": float("nan"), "ks_pass": False,
        }
        rows.append({
            "Adj Strategy": s, "Grade": g, "AgeBucket": a,
            "p_draw": p_draw, "p_rep": p_rep, "p_rc_given_rep": p_rc_given_rep,
            "mu_draw": stats_d["mu"], "sig_draw": stats_d["sig"],
            "n_draw": stats_d["n"], "n_funds_draw": stats_d["n_funds"], "ks_draw": stats_d["ks_D"], "ks_pass_draw": stats_d["ks_pass"],
            "mu_rep": stats_r["mu"], "sig_rep": stats_r["sig"],
            "n_rep": stats_r["n"], "n_funds_rep": stats_r["n_funds"], "ks_rep": stats_r["ks_D"], "ks_pass_rep": stats_r["ks_pass"],
            "mu_rc": stats_c["mu"], "sig_rc": stats_c["sig"],
            "n_rc": stats_c["n"], "n_funds_rc": stats_c["n_funds"], "ks_rc": stats_c["ks_D"], "ks_pass_rc": stats_c["ks_pass"],
            "n_obs": int(len(grp)), "n_funds": int(grp["FundID"].nunique())
        })
    cal = pd.DataFrame(rows)

    # strategy fallback
    rows_s = []
    for s, grp in train.groupby(["Adj Strategy"], dropna=False):
        p_draw = float(grp["draw_event"].mean()) if len(grp) else 0.0
        p_rep = float(grp["rep_event"].mean()) if len(grp) else 0.0
        rep_q = grp[grp["Adj Repayment EUR"] > 0]
        p_rc_given_rep = float(rep_q["rc_given_rep_event"].mean()) if len(rep_q) else 0.0
        stats_d = fit_lognormal_stats(grp["draw_ratio"], grp["FundID"])
        stats_r = fit_lognormal_stats(grp["rep_ratio"], grp["FundID"])
        stats_c = fit_lognormal_stats(rep_q["rc_ratio_given_rep"], rep_q["FundID"]) if len(rep_q) else {
            "mu": 0.0, "sig": SIGMA_FLOOR, "n": 0, "n_funds": 0, "ks_D": float("nan"), "ks_pass": False,
        }
        rows_s.append({
            "Adj Strategy": s,
            "p_draw": p_draw, "p_rep": p_rep, "p_rc_given_rep": p_rc_given_rep,
            "mu_draw": stats_d["mu"], "sig_draw": stats_d["sig"],
            "n_draw": stats_d["n"], "n_funds_draw": stats_d["n_funds"], "ks_draw": stats_d["ks_D"], "ks_pass_draw": stats_d["ks_pass"],
            "mu_rep": stats_r["mu"], "sig_rep": stats_r["sig"],
            "n_rep": stats_r["n"], "n_funds_rep": stats_r["n_funds"], "ks_rep": stats_r["ks_D"], "ks_pass_rep": stats_r["ks_pass"],
            "mu_rc": stats_c["mu"], "sig_rc": stats_c["sig"],
            "n_rc": stats_c["n"], "n_funds_rc": stats_c["n_funds"], "ks_rc": stats_c["ks_D"], "ks_pass_rc": stats_c["ks_pass"],
            "n_obs": int(len(grp)), "n_funds": int(grp["FundID"].nunique())
        })
    cal_s = pd.DataFrame(rows_s)

    # strategy + grade fallback
    rows_sg = []
    for (s, g), grp in train.groupby(["Adj Strategy", "Grade"], dropna=False):
        p_draw = float(grp["draw_event"].mean()) if len(grp) else 0.0
        p_rep = float(grp["rep_event"].mean()) if len(grp) else 0.0
        rep_q = grp[grp["Adj Repayment EUR"] > 0]
        p_rc_given_rep = float(rep_q["rc_given_rep_event"].mean()) if len(rep_q) else 0.0
        stats_d = fit_lognormal_stats(grp["draw_ratio"], grp["FundID"])
        stats_r = fit_lognormal_stats(grp["rep_ratio"], grp["FundID"])
        stats_c = fit_lognormal_stats(rep_q["rc_ratio_given_rep"], rep_q["FundID"]) if len(rep_q) else {
            "mu": 0.0, "sig": SIGMA_FLOOR, "n": 0, "n_funds": 0, "ks_D": float("nan"), "ks_pass": False,
        }
        rows_sg.append({
            "Adj Strategy": s, "Grade": g,
            "p_draw": p_draw, "p_rep": p_rep, "p_rc_given_rep": p_rc_given_rep,
            "mu_draw": stats_d["mu"], "sig_draw": stats_d["sig"],
            "n_draw": stats_d["n"], "n_funds_draw": stats_d["n_funds"], "ks_draw": stats_d["ks_D"], "ks_pass_draw": stats_d["ks_pass"],
            "mu_rep": stats_r["mu"], "sig_rep": stats_r["sig"],
            "n_rep": stats_r["n"], "n_funds_rep": stats_r["n_funds"], "ks_rep": stats_r["ks_D"], "ks_pass_rep": stats_r["ks_pass"],
            "mu_rc": stats_c["mu"], "sig_rc": stats_c["sig"],
            "n_rc": stats_c["n"], "n_funds_rc": stats_c["n_funds"], "ks_rc": stats_c["ks_D"], "ks_pass_rc": stats_c["ks_pass"],
            "n_obs": int(len(grp)), "n_funds": int(grp["FundID"].nunique())
        })
    cal_sg = pd.DataFrame(rows_sg)

    # strategy + age fallback
    rows_sa = []
    for (s, a), grp in train.groupby(["Adj Strategy", "AgeBucketCalib"], dropna=False):
        p_draw = float(grp["draw_event"].mean()) if len(grp) else 0.0
        p_rep = float(grp["rep_event"].mean()) if len(grp) else 0.0
        rep_q = grp[grp["Adj Repayment EUR"] > 0]
        p_rc_given_rep = float(rep_q["rc_given_rep_event"].mean()) if len(rep_q) else 0.0
        stats_d = fit_lognormal_stats(grp["draw_ratio"], grp["FundID"])
        stats_r = fit_lognormal_stats(grp["rep_ratio"], grp["FundID"])
        stats_c = fit_lognormal_stats(rep_q["rc_ratio_given_rep"], rep_q["FundID"]) if len(rep_q) else {
            "mu": 0.0, "sig": SIGMA_FLOOR, "n": 0, "n_funds": 0, "ks_D": float("nan"), "ks_pass": False,
        }
        rows_sa.append({
            "Adj Strategy": s, "AgeBucket": a,
            "p_draw": p_draw, "p_rep": p_rep, "p_rc_given_rep": p_rc_given_rep,
            "mu_draw": stats_d["mu"], "sig_draw": stats_d["sig"],
            "n_draw": stats_d["n"], "n_funds_draw": stats_d["n_funds"], "ks_draw": stats_d["ks_D"], "ks_pass_draw": stats_d["ks_pass"],
            "mu_rep": stats_r["mu"], "sig_rep": stats_r["sig"],
            "n_rep": stats_r["n"], "n_funds_rep": stats_r["n_funds"], "ks_rep": stats_r["ks_D"], "ks_pass_rep": stats_r["ks_pass"],
            "mu_rc": stats_c["mu"], "sig_rc": stats_c["sig"],
            "n_rc": stats_c["n"], "n_funds_rc": stats_c["n_funds"], "ks_rc": stats_c["ks_D"], "ks_pass_rc": stats_c["ks_pass"],
            "n_obs": int(len(grp)), "n_funds": int(grp["FundID"].nunique())
        })
    cal_sa = pd.DataFrame(rows_sa)

    global_p_draw = float(train["draw_event"].mean())
    global_p_rep = float(train["rep_event"].mean())
    rep_all = train[train["Adj Repayment EUR"] > 0]
    global_p_rc_given_rep = float(rep_all["rc_given_rep_event"].mean()) if len(rep_all) else 0.0

    stats_g_draw = fit_lognormal_stats(train["draw_ratio"], train["FundID"])
    stats_g_rep = fit_lognormal_stats(train["rep_ratio"], train["FundID"])
    stats_g_rc = fit_lognormal_stats(rep_all["rc_ratio_given_rep"], rep_all["FundID"]) if len(rep_all) else {
        "mu": 0.0, "sig": SIGMA_FLOOR, "n": 0, "n_funds": 0, "ks_D": float("nan"), "ks_pass": False,
    }

    def lookup_params(strategy, grade, age_bucket) -> Dict[str, float]:
        age_bucket_calib = "ALL" if CALIBRATION_BUCKET_MODE == "strategy_grade" else age_bucket
        m = (cal["Adj Strategy"].eq(strategy)) & (cal["Grade"].eq(grade)) & (cal["AgeBucket"].eq(age_bucket_calib))
        child = cal[m].iloc[0].to_dict() if m.any() else {}

        ss = cal_s[cal_s["Adj Strategy"].eq(strategy)]
        parent_s = ss.iloc[0].to_dict() if len(ss) else {}

        ssg = cal_sg[(cal_sg["Adj Strategy"].eq(strategy)) & (cal_sg["Grade"].eq(grade))]
        parent_sg = ssg.iloc[0].to_dict() if len(ssg) else {}

        ssa = cal_sa[(cal_sa["Adj Strategy"].eq(strategy)) & (cal_sa["AgeBucket"].eq(age_bucket_calib))]
        parent_sa = ssa.iloc[0].to_dict() if len(ssa) else {}

        p_s = float(np.clip(parent_s.get("p_draw", global_p_draw), 0.0, 1.0))
        p_sg = float(np.clip(parent_sg.get("p_draw", p_s), 0.0, 1.0))
        p_sa = float(np.clip(parent_sa.get("p_draw", p_s), 0.0, 1.0))
        p_draw_parent = _combine_p(p_sg, parent_sg.get("n_obs", 0), parent_sg.get("n_funds", 0),
                                   p_sa, parent_sa.get("n_obs", 0), parent_sa.get("n_funds", 0), p_s)
        p_draw = _blend_p(child.get("p_draw", p_draw_parent), child.get("n_obs", 0), child.get("n_funds", 0), p_draw_parent)

        p_s = float(np.clip(parent_s.get("p_rep", global_p_rep), 0.0, 1.0))
        p_sg = float(np.clip(parent_sg.get("p_rep", p_s), 0.0, 1.0))
        p_sa = float(np.clip(parent_sa.get("p_rep", p_s), 0.0, 1.0))
        p_rep_parent = _combine_p(p_sg, parent_sg.get("n_obs", 0), parent_sg.get("n_funds", 0),
                                  p_sa, parent_sa.get("n_obs", 0), parent_sa.get("n_funds", 0), p_s)
        p_rep = _blend_p(child.get("p_rep", p_rep_parent), child.get("n_obs", 0), child.get("n_funds", 0), p_rep_parent)

        p_s = float(np.clip(parent_s.get("p_rc_given_rep", global_p_rc_given_rep), 0.0, 1.0))
        p_sg = float(np.clip(parent_sg.get("p_rc_given_rep", p_s), 0.0, 1.0))
        p_sa = float(np.clip(parent_sa.get("p_rc_given_rep", p_s), 0.0, 1.0))
        p_rc_parent = _combine_p(p_sg, parent_sg.get("n_rep", 0), parent_sg.get("n_funds_rep", 0),
                                 p_sa, parent_sa.get("n_rep", 0), parent_sa.get("n_funds_rep", 0), p_s)
        p_rc = _blend_p(child.get("p_rc_given_rep", p_rc_parent), child.get("n_rep", 0), child.get("n_funds_rep", 0), p_rc_parent)

        mu_draw_s, sig_draw_s = _blend(
            parent_s.get("mu_draw", stats_g_draw["mu"]), parent_s.get("sig_draw", stats_g_draw["sig"]),
            parent_s.get("n_draw", 0), parent_s.get("n_funds_draw", 0), parent_s.get("ks_pass_draw", False),
            stats_g_draw["mu"], stats_g_draw["sig"], stats_g_draw["n"], stats_g_draw["n_funds"], stats_g_draw["ks_pass"]
        )
        mu_rep_s, sig_rep_s = _blend(
            parent_s.get("mu_rep", stats_g_rep["mu"]), parent_s.get("sig_rep", stats_g_rep["sig"]),
            parent_s.get("n_rep", 0), parent_s.get("n_funds_rep", 0), parent_s.get("ks_pass_rep", False),
            stats_g_rep["mu"], stats_g_rep["sig"], stats_g_rep["n"], stats_g_rep["n_funds"], stats_g_rep["ks_pass"]
        )
        mu_rc_s, sig_rc_s = _blend(
            parent_s.get("mu_rc", stats_g_rc["mu"]), parent_s.get("sig_rc", stats_g_rc["sig"]),
            parent_s.get("n_rc", 0), parent_s.get("n_funds_rc", 0), parent_s.get("ks_pass_rc", False),
            stats_g_rc["mu"], stats_g_rc["sig"], stats_g_rc["n"], stats_g_rc["n_funds"], stats_g_rc["ks_pass"]
        )

        mu_draw_sg, sig_draw_sg = _blend(
            parent_sg.get("mu_draw", mu_draw_s), parent_sg.get("sig_draw", sig_draw_s),
            parent_sg.get("n_draw", 0), parent_sg.get("n_funds_draw", 0), parent_sg.get("ks_pass_draw", False),
            mu_draw_s, sig_draw_s, parent_s.get("n_draw", 0), parent_s.get("n_funds_draw", 0), parent_s.get("ks_pass_draw", False)
        )
        mu_rep_sg, sig_rep_sg = _blend(
            parent_sg.get("mu_rep", mu_rep_s), parent_sg.get("sig_rep", sig_rep_s),
            parent_sg.get("n_rep", 0), parent_sg.get("n_funds_rep", 0), parent_sg.get("ks_pass_rep", False),
            mu_rep_s, sig_rep_s, parent_s.get("n_rep", 0), parent_s.get("n_funds_rep", 0), parent_s.get("ks_pass_rep", False)
        )
        mu_rc_sg, sig_rc_sg = _blend(
            parent_sg.get("mu_rc", mu_rc_s), parent_sg.get("sig_rc", sig_rc_s),
            parent_sg.get("n_rc", 0), parent_sg.get("n_funds_rc", 0), parent_sg.get("ks_pass_rc", False),
            mu_rc_s, sig_rc_s, parent_s.get("n_rc", 0), parent_s.get("n_funds_rc", 0), parent_s.get("ks_pass_rc", False)
        )

        mu_draw_sa, sig_draw_sa = _blend(
            parent_sa.get("mu_draw", mu_draw_s), parent_sa.get("sig_draw", sig_draw_s),
            parent_sa.get("n_draw", 0), parent_sa.get("n_funds_draw", 0), parent_sa.get("ks_pass_draw", False),
            mu_draw_s, sig_draw_s, parent_s.get("n_draw", 0), parent_s.get("n_funds_draw", 0), parent_s.get("ks_pass_draw", False)
        )
        mu_rep_sa, sig_rep_sa = _blend(
            parent_sa.get("mu_rep", mu_rep_s), parent_sa.get("sig_rep", sig_rep_s),
            parent_sa.get("n_rep", 0), parent_sa.get("n_funds_rep", 0), parent_sa.get("ks_pass_rep", False),
            mu_rep_s, sig_rep_s, parent_s.get("n_rep", 0), parent_s.get("n_funds_rep", 0), parent_s.get("ks_pass_rep", False)
        )
        mu_rc_sa, sig_rc_sa = _blend(
            parent_sa.get("mu_rc", mu_rc_s), parent_sa.get("sig_rc", sig_rc_s),
            parent_sa.get("n_rc", 0), parent_sa.get("n_funds_rc", 0), parent_sa.get("ks_pass_rc", False),
            mu_rc_s, sig_rc_s, parent_s.get("n_rc", 0), parent_s.get("n_funds_rc", 0), parent_s.get("ks_pass_rc", False)
        )

        mu_draw_p, sig_draw_p = _combine_mu_sig(
            mu_draw_sg, sig_draw_sg, parent_sg.get("n_draw", 0), parent_sg.get("n_funds_draw", 0), parent_sg.get("ks_pass_draw", False),
            mu_draw_sa, sig_draw_sa, parent_sa.get("n_draw", 0), parent_sa.get("n_funds_draw", 0), parent_sa.get("ks_pass_draw", False),
            mu_draw_s, sig_draw_s
        )
        mu_rep_p, sig_rep_p = _combine_mu_sig(
            mu_rep_sg, sig_rep_sg, parent_sg.get("n_rep", 0), parent_sg.get("n_funds_rep", 0), parent_sg.get("ks_pass_rep", False),
            mu_rep_sa, sig_rep_sa, parent_sa.get("n_rep", 0), parent_sa.get("n_funds_rep", 0), parent_sa.get("ks_pass_rep", False),
            mu_rep_s, sig_rep_s
        )
        mu_rc_p, sig_rc_p = _combine_mu_sig(
            mu_rc_sg, sig_rc_sg, parent_sg.get("n_rc", 0), parent_sg.get("n_funds_rc", 0), parent_sg.get("ks_pass_rc", False),
            mu_rc_sa, sig_rc_sa, parent_sa.get("n_rc", 0), parent_sa.get("n_funds_rc", 0), parent_sa.get("ks_pass_rc", False),
            mu_rc_s, sig_rc_s
        )

        mu_draw, sig_draw = _blend(
            child.get("mu_draw", mu_draw_p), child.get("sig_draw", sig_draw_p),
            child.get("n_draw", 0), child.get("n_funds_draw", 0), child.get("ks_pass_draw", False),
            mu_draw_p, sig_draw_p, 0, 0, True
        )
        mu_rep, sig_rep = _blend(
            child.get("mu_rep", mu_rep_p), child.get("sig_rep", sig_rep_p),
            child.get("n_rep", 0), child.get("n_funds_rep", 0), child.get("ks_pass_rep", False),
            mu_rep_p, sig_rep_p, 0, 0, True
        )
        mu_rc, sig_rc = _blend(
            child.get("mu_rc", mu_rc_p), child.get("sig_rc", sig_rc_p),
            child.get("n_rc", 0), child.get("n_funds_rc", 0), child.get("ks_pass_rc", False),
            mu_rc_p, sig_rc_p, 0, 0, True
        )

        return {
            "p_draw": p_draw,
            "p_rep": p_rep,
            "p_rc_given_rep": p_rc,
            "mu_draw": float(mu_draw), "sig_draw": float(sig_draw),
            "mu_rep": float(mu_rep), "sig_rep": float(sig_rep),
            "mu_rc": float(mu_rc), "sig_rc": float(sig_rc),
        }

    # =============================
    # Drawdown scaling calibration (censored funds, by strategy+grade)
    # =============================
    draw_scale_sg = {}
    draw_scale_s = {}
    draw_scale_g = 1.0

    def build_cum_draw_obs(df: pd.DataFrame, fund_commit: dict) -> pd.DataFrame:
        rows = []
        for fid, g in df.groupby("FundID"):
            C = float(fund_commit.get(fid, 0.0) or 0.0)
            if not np.isfinite(C) or C <= 0:
                continue
            g2 = g.sort_values("age_q_fc")
            g2 = (g2.groupby("age_q_fc", as_index=False)
                    .agg(draw=("Adj Drawdown EUR", "sum"),
                         **{"Adj Strategy": ("Adj Strategy", "last"),
                            "Grade": ("Grade", "last")}))
            g2["cum_draw"] = g2["draw"].cumsum()
            g2["ratio"] = g2["cum_draw"] / C
            g2["Adj Strategy"] = g2["Adj Strategy"].fillna("Unknown")
            g2["Grade"] = g2["Grade"].fillna("D").astype(str).str.strip()
            g2["FundID"] = fid
            rows.append(g2[["FundID", "Adj Strategy", "Grade", "age_q_fc", "ratio"]])
        if not rows:
            return pd.DataFrame(columns=["FundID", "Adj Strategy", "Grade", "age_q_fc", "ratio"])
        return pd.concat(rows, ignore_index=True)

    def expected_draw_increment(strategy: str, grade: str, age_q: int) -> float:
        age_bucket = make_age_bucket_q(age_q)
        ip_q = int(ip_by_strategy.get(strategy, IP_Q_DEFAULT))
        draw_mult = 1.0
        if USE_DRAW_AGE_SHAPE and ip_q > 0:
            if ENFORCE_IP_LIMITS and age_q > ip_q:
                draw_mult = 0.0
            else:
                frac = float(age_q) / float(ip_q)
                draw_mult = max(DRAW_AGE_MIN_MULT, (1.0 - frac) ** DRAW_AGE_DECAY_POWER)

        params = lookup_params(strategy, grade, age_bucket)
        if USE_HAZARD_MODELS and beta_draw is not None:
            Xr = build_feature_row(strategy, grade, age_q, 0.0, False,
                                   hazard_meta["draw_cols"], hazard_meta["draw_means"], hazard_meta["draw_stds"])
            p_draw_base = float(_sigmoid(Xr @ beta_draw)[0])
        else:
            p_draw_base = float(params.get("p_draw", 0.0))

        grade_key = grade if grade in GRADE_DRAW_P_MULT else "D"
        p_draw_adj = p_draw_base * draw_mult * float(GRADE_DRAW_P_MULT.get(grade_key, 1.0))
        p_draw_adj = min(p_draw_adj, 1.0)

        mu = float(params.get("mu_draw", 0.0))
        sig = float(params.get("sig_draw", SIGMA_FLOOR))
        mean_ratio = float(np.exp(mu + 0.5 * sig * sig))
        mean_ratio = min(mean_ratio * float(GRADE_DRAW_SIZE_MULT.get(grade_key, 1.0)), 1.0)
        return float(p_draw_adj * mean_ratio)

    def baseline_curve(strategy: str, grade: str, ages: List[int]) -> np.ndarray:
        ages_sorted = sorted(set(int(a) for a in ages))
        out = []
        cum = 0.0
        for a in ages_sorted:
            cum += expected_draw_increment(strategy, grade, a)
            out.append(cum)
        return np.array(out, dtype=float), ages_sorted

    def fit_scale_for_group(grp: pd.DataFrame, target_col: str) -> Tuple[float, float]:
        ages = grp["age_q_fc"].astype(int).tolist()
        base, ages_sorted = baseline_curve(grp["Adj Strategy"].iloc[0], grp["Grade"].iloc[0], ages)
        g2 = grp.set_index("age_q_fc").loc[ages_sorted]
        target = g2[target_col].to_numpy(dtype=float)
        weights = g2["n_funds"].to_numpy(dtype=float)
        denom = float(np.sum(weights * base * base))
        if denom <= 0:
            return 1.0, float("inf")
        scale_raw = float(np.sum(weights * target * base) / denom)
        n_funds = float(g2["n_funds"].max())
        w_shrink = n_funds / (n_funds + SHRINK_FUNDS)
        scale = 1.0 + w_shrink * (scale_raw - 1.0)
        sse = float(np.sum(weights * (scale * base - target) ** 2))
        return max(scale, 0.0), sse

    if USE_DRAWDOWN_CALIBRATION:
        obs = build_cum_draw_obs(train, fund_commit)
        if len(obs):
            curves_sg = (obs.groupby(["Adj Strategy", "Grade", "age_q_fc"])
                            .agg(mean_ratio=("ratio", "mean"),
                                 median_ratio=("ratio", "median"),
                                 n_funds=("FundID", "nunique"))
                            .reset_index())
            curves_s = (obs.groupby(["Adj Strategy", "age_q_fc"])
                           .agg(mean_ratio=("ratio", "mean"),
                                median_ratio=("ratio", "median"),
                                n_funds=("FundID", "nunique"))
                           .reset_index())
            # strategy+grade
            for (s, g), grp in curves_sg.groupby(["Adj Strategy", "Grade"]):
                g2 = grp[grp["n_funds"] >= DRAW_CALIB_MIN_FUNDS]
                if len(g2) < DRAW_CALIB_MIN_AGES:
                    continue
                scale_mean, sse_mean = fit_scale_for_group(g2, "mean_ratio")
                scale_med, sse_med = fit_scale_for_group(g2, "median_ratio")
                if DRAW_CALIB_TARGET == "mean":
                    draw_scale_sg[(s, g)] = scale_mean
                elif DRAW_CALIB_TARGET == "median":
                    draw_scale_sg[(s, g)] = scale_med
                else:
                    draw_scale_sg[(s, g)] = scale_mean if sse_mean <= sse_med else scale_med

            # strategy fallback: weighted average of grade-level scales
            for s, grp in curves_sg.groupby(["Adj Strategy"]):
                scales = []
                for g in grp["Grade"].unique():
                    key = (s, g)
                    if key in draw_scale_sg:
                        n_funds = float(grp[grp["Grade"] == g]["n_funds"].max())
                        scales.append((draw_scale_sg[key], n_funds))
                if scales:
                    wsum = sum(w for _, w in scales)
                    if wsum > 0:
                        draw_scale_s[s] = sum(scale * w for scale, w in scales) / wsum

            # global fallback: weighted average of strategy scales
            if draw_scale_s:
                wsum = 0.0
                tot = 0.0
                for s, scale in draw_scale_s.items():
                    n_funds = float(curves_s[curves_s["Adj Strategy"] == s]["n_funds"].max())
                    if not np.isfinite(n_funds) or n_funds <= 0:
                        continue
                    wsum += n_funds
                    tot += scale * n_funds
                if wsum > 0:
                    draw_scale_g = tot / wsum

            print(f"Drawdown calibration: sg_scales={len(draw_scale_sg)}, "
                  f"s_scales={len(draw_scale_s)}, global_scale={round(draw_scale_g, 4)} "
                  f"(target={DRAW_CALIB_TARGET})")

    # Runoff calibration
    runoff_mult_by_strategy = {}
    if USE_RUNOFF_CALIBRATION:
        tail = train[train["AgeBucket"].isin(["15-20y", "20y+"])].copy()
        mid = train[train["AgeBucket"].isin(["6-8y", "8-10y", "10-15y"])].copy()
        for s in train["Adj Strategy"].dropna().unique():
            t = tail[tail["Adj Strategy"] == s]
            m = mid[mid["Adj Strategy"] == s]
            if len(t) and len(m) and t["rep_ratio"].notna().any() and m["rep_ratio"].notna().any():
                num = t["rep_ratio"].mean()
                den = m["rep_ratio"].mean()
                if den > 0:
                    runoff_mult_by_strategy[s] = float(np.clip(num / den, RUNOFF_MULT_MIN, RUNOFF_MULT_MAX))

    # =============================
    # Recallable soft limits
    # =============================
    kmp_needed = ["FundID", "Recallable_Percentage_Decimal", "Expiration_Quarters"]
    missing_k = [c for c in kmp_needed if c not in kmp.columns]
    if missing_k:
        raise ValueError(f"Missing columns in kmp: {missing_k}")

    kmp2 = kmp[kmp_needed].copy()
    kmp2["Recallable_Percentage_Decimal"] = pd.to_numeric(kmp2["Recallable_Percentage_Decimal"], errors="coerce")
    kmp2["Expiration_Quarters"] = pd.to_numeric(kmp2["Expiration_Quarters"], errors="coerce")
    kmp2 = kmp2.set_index("FundID")

    tmp_rho = (
        data.groupby("FundID", as_index=False)
        .agg(sum_rc=("Recallable", "sum"), C_last=("Commitment_Level", "max"))
    )
    tmp_rho["rho_emp"] = np.where(tmp_rho["C_last"] > 0, tmp_rho["sum_rc"] / tmp_rho["C_last"], np.nan)
    rho_emp = tmp_rho.set_index("FundID")["rho_emp"].to_dict()

    def soft_params(strategy: str) -> Tuple[float, int]:
        s = data[data["Adj Strategy"].eq(strategy)]
        if len(s):
            rho = float(np.nanquantile(s["Recallable"].fillna(0.0), SOFT_RHO_PCTL))
            rho = float(np.clip(rho, 0.0, 0.9))
        else:
            rho = 0.0
        return rho, SOFT_EXPIRY_FALLBACK

    def get_rho_E(fid: str, strategy: str) -> Tuple[float, int]:
        if fid in kmp2.index:
            r = kmp2.loc[fid]
            rho = float(r.get("Recallable_Percentage_Decimal", np.nan))
            E = int(r.get("Expiration_Quarters", np.nan)) if pd.notna(r.get("Expiration_Quarters", np.nan)) else SOFT_EXPIRY_FALLBACK
            if not np.isfinite(rho):
                rho = rho_emp.get(fid, np.nan)
            if not np.isfinite(rho):
                rho, _ = soft_params(strategy)
            return float(np.clip(rho, 0.0, 0.9)), int(E)
        rho = rho_emp.get(fid, np.nan)
        if not np.isfinite(rho):
            rho, E = soft_params(strategy)
        else:
            _, E = soft_params(strategy)
        return float(np.clip(rho, 0.0, 0.9)), int(E)

    # =============================
    # Build initial states at test start
    # =============================
    fund_states = {}
    fund_start_qe = {}
    fund_bucket = {}

    # Build transition matrices
    all_counts, all_probs, all_n = build_yearly_transition_from_data(train, strategy=None)
    strat_probs = {}
    strat_n = {}
    for s in train["Adj Strategy"].dropna().unique():
        _, probs_s, n_s = build_yearly_transition_from_data(train, strategy=s)
        strat_probs[s] = probs_s
        strat_n[s] = n_s

    def get_transition_matrix(strategy: str) -> pd.DataFrame:
        if strategy in strat_probs and strat_n.get(strategy, 0) >= 30:
            return strat_probs[strategy]
        return all_probs

    # Pre-build actual fund history for replay
    data_by_fund = {fid: g.copy() for fid, g in data.groupby("FundID")}

    for fid in test_funds:
        hist = data_by_fund.get(fid)
        if hist is None or hist.empty:
            continue
        hist = hist.sort_values("quarter_end")
        hist_pre = hist[hist["quarter_end"] <= train_end_qe]
        if hist_pre.empty:
            continue

        last = hist_pre.iloc[-1]
        strategy = nav_strategy_by_fund.get(fid, str(last.get("Adj Strategy") or "Unknown")) if USE_NAV_PROJECTIONS else str(last.get("Adj Strategy") or "Unknown")
        grade0 = nav_grade_by_fund.get(fid, str(last.get("Grade") or "D")).strip() if USE_NAV_PROJECTIONS else str(last.get("Grade") or "D").strip()
        if grade0 not in GRADE_STATES:
            grade0 = "D"
        age0_src = nav_age_by_fund.get(fid, None) if USE_NAV_PROJECTIONS else None
        if age0_src is None or pd.isna(age0_src):
            age0_src = last.get("age_q_model") or 0.0
        age0 = int(round(float(age0_src)))
        nav0 = float(nav_start_by_fund.get(fid, last.get("NAV Adjusted EUR") or 0.0)) if USE_NAV_PROJECTIONS else float(last.get("NAV Adjusted EUR") or 0.0)

        fund_start_qe[fid] = hist["quarter_end"].iloc[0]
        start_qe = fund_start_qe[fid]

        # initialize recallables and DD_cum_commit
        C = float(fund_commit.get(fid, 0.0) or 0.0)
        rho, E = get_rho_E(fid, strategy)
        ledger = RecallableLedger(rho=rho, expiry_quarters=E, commitment=C)
        dd_cum_commit = 0.0

        if USE_NAV_PROJECTIONS:
            draw_hist = float(draw_cum_hist.get(fid, 0.0) or 0.0)
            dd_cum_commit = min(draw_hist, C)
        else:
            # replay history to reconstruct DD_commit and recallables
            for _, row in hist_pre.iterrows():
                qe = row["quarter_end"]
                step = quarter_diff(qe, start_qe)
                draw = float(row.get("Adj Drawdown EUR") or 0.0)
                rep = float(row.get("Adj Repayment EUR") or 0.0)
                rc = float(row.get("Recallable") or 0.0)
                if rep > 0 and rc > 0:
                    ledger.add_recallable(step, rc, enforce_cap=True)
                if draw > 0:
                    cons = ledger.consume_for_drawdown(step, draw)
                    dd_cum_commit += cons["use_commitment"]

        # bucket (based on start grade/age)
        age_bucket0 = make_age_bucket_q(age0)
        if REPORT_BUCKET_MODE == "strategy":
            bucket_key = (strategy, "ALL", "ALL")
        elif REPORT_BUCKET_MODE == "strategy_grade":
            bucket_key = (strategy, grade0, "ALL")
        else:
            bucket_key = (strategy, grade0, str(age_bucket0))
        fund_bucket[fid] = bucket_key

        fund_states[fid] = {
            "strategy": strategy,
            "grade": grade0,
            "age0": age0,
            "nav": nav0,
            "dd_commit": dd_cum_commit,
            "ledger": ledger,
            "commitment": C,
            "cap_qe": cap_by_fund.get(fid, pd.NaT),
        }

    # bucket index mapping
    bucket_keys = sorted(set(fund_bucket.values()))
    bucket_index = {b: i for i, b in enumerate(bucket_keys)}

    # =============================
    # Actual portfolio series (test)
    # =============================
    T = len(test_quarters)
    B = len(bucket_keys)

    actual_draw = np.zeros(T)
    actual_rep = np.zeros(T)
    actual_nav = np.zeros(T)

    actual_draw_b = np.zeros((T, B))
    actual_rep_b = np.zeros((T, B))
    actual_nav_b = np.zeros((T, B))

    # Build per-fund actual grid
    for fid in fund_states.keys():
        hist = data_by_fund[fid].copy().sort_values("quarter_end")
        # collapse any duplicate quarter_end rows before aligning to grid
        hist_q = (hist.groupby("quarter_end", as_index=True)
                  .agg({
                      "Adj Drawdown EUR": "sum",
                      "Adj Repayment EUR": "sum",
                      "NAV Adjusted EUR": "last",
                  }))
        # grid for test quarters
        grid = pd.DataFrame(index=pd.Index(test_quarters, name="quarter_end"))
        grid["Draw"] = hist_q["Adj Drawdown EUR"]
        grid["Rep"] = hist_q["Adj Repayment EUR"]
        grid["NAV"] = hist_q["NAV Adjusted EUR"]

        # Fill missing: Draw/Rep = 0, NAV = ffill from last known (including pre-test)
        grid["Draw"] = grid["Draw"].fillna(0.0)
        grid["Rep"] = grid["Rep"].fillna(0.0)

        # seed NAV from last known before test start
        nav_seed = hist_q.loc[hist_q.index <= train_end_qe, "NAV Adjusted EUR"]
        nav_seed = float(nav_seed.iloc[-1]) if len(nav_seed) else 0.0
        grid["NAV"] = grid["NAV"].ffill().fillna(nav_seed)

        bkey = fund_bucket[fid]
        bidx = bucket_index[bkey]

        actual_draw += grid["Draw"].to_numpy(dtype=float)
        actual_rep += grid["Rep"].to_numpy(dtype=float)
        actual_nav += grid["NAV"].to_numpy(dtype=float)

        actual_draw_b[:, bidx] += grid["Draw"].to_numpy(dtype=float)
        actual_rep_b[:, bidx] += grid["Rep"].to_numpy(dtype=float)
        actual_nav_b[:, bidx] += grid["NAV"].to_numpy(dtype=float)

    # =============================
    # Simulation runner
    # =============================

    def run_backtest(scenario: str, conditional: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        sim_draw = np.zeros((n_sims, T))
        sim_rep = np.zeros((n_sims, T))
        sim_nav = np.zeros((n_sims, T))

        sim_draw_b = np.zeros((n_sims, T, B))
        sim_rep_b = np.zeros((n_sims, T, B))
        sim_nav_b = np.zeros((n_sims, T, B))

        fund_ids = list(fund_states.keys())
        n_funds = len(fund_ids)
        fund_index = {fid: i for i, fid in enumerate(fund_ids)}

        omega_key = None
        if USE_NAV_PROJECTIONS and sim_ids:
            if len(sim_ids) == 1:
                sim_id_list = list(range(n_sims))
                omega_key = sim_ids[0]
            else:
                sim_id_list = sim_ids
        else:
            sim_id_list = list(range(n_sims))

        for s_idx, sim_id in enumerate(sim_id_list):
            rng = np.random.default_rng(seed + s_idx)
            omega_sim_id = omega_key if omega_key is not None else sim_id

            # MSCI path: conditional uses actual MSCI; unconditional may use NAV Logic or simulated
            if conditional:
                msci_series = pd.Series([msci_map[qe] for qe in test_quarters],
                                        index=pd.Index(test_quarters, name="quarter_end"))
                msci_mu = msci_mu_all
                msci_sigma = msci_sigma_all
            else:
                if USE_NAV_PROJECTIONS and omega_sim_id in msci_map_sim:
                    msci_series = pd.Series([float(msci_map_sim[omega_sim_id].get(qe, 0.0)) for qe in test_quarters],
                                            index=pd.Index(test_quarters, name="quarter_end"))
                    stats = msci_stats_by_sim.get(omega_sim_id, {})
                    msci_mu = float(stats.get("mean", msci_mu_all))
                    msci_sigma = float(stats.get("std", msci_sigma_all))
                else:
                    msci_path = simulate_msci_path(msci_q[msci_q["quarter_end"] <= train_end_qe],
                                                   start_qe=train_end_qe,
                                                   n_quarters=T,
                                                   scenario=scenario,
                                                   tilt_strength=tilt_strength,
                                                   rng=rng)
                    msci_path = msci_path.set_index("quarter_end")
                    msci_series = msci_path["msci_ret_q"]
                    msci_mu = msci_mu_all
                    msci_sigma = msci_sigma_all

            if not np.isfinite(msci_sigma) or msci_sigma <= 0:
                msci_sigma = 1e-6

            # Prepare per-quarter copula uniforms
            U_by_q = {}
            for qe in test_quarters:
                U_by_q[qe] = {
                    "draw_event": one_factor_uniforms(n_funds, rng, rho_event),
                    "draw_size": one_factor_uniforms(n_funds, rng, rho_size),
                    "rep_event": one_factor_uniforms(n_funds, rng, rho_event),
                    "rep_size": one_factor_uniforms(n_funds, rng, rho_size),
                    "rc_event": one_factor_uniforms(n_funds, rng, rho_event),
                    "rc_size": one_factor_uniforms(n_funds, rng, rho_size),
                }

            # initialize fund states for this sim
            sim_state = {}
            for fid in fund_ids:
                st0 = fund_states[fid]
                sim_state[fid] = {
                    "strategy": st0["strategy"],
                    "grade": st0["grade"],
                    "age0": st0["age0"],
                    "nav": float(st0["nav"]),
                    "dd_commit": float(st0["dd_commit"]),
                    "ip_catchup_done": False,
                    "ledger": RecallableLedger(
                        rho=st0["ledger"].rho,
                        expiry_quarters=st0["ledger"].expiry_quarters,
                        commitment=st0["ledger"].commitment,
                        buckets=[RecallableBucket(b.created_q, b.expiry_q, b.amount_remaining) for b in st0["ledger"].buckets]
                    ),
                    "commitment": float(st0["commitment"]),
                    "cap_qe": st0["cap_qe"],
                    "alive": True,
                }

            for t_idx, qe in enumerate(test_quarters):
                msci_r = float(msci_series.get(qe, 0.0))
                msci_r_lag1 = float(msci_series.get(test_quarters[t_idx - 1], 0.0)) if t_idx > 0 else float(msci_lag_map.get(qe, 0.0) or 0.0)

                msci_z = (msci_r - msci_mu) / msci_sigma
                msci_z = float(np.clip(msci_z, -MSCI_Z_CLIP, MSCI_Z_CLIP))
                msci_z_eff = max(msci_z, 0.0) if MSCI_REP_POS_ONLY else msci_z

                U = U_by_q[qe]

                for fid in fund_ids:
                    st = sim_state[fid]
                    if not st["alive"]:
                        continue

                    age_q = int(st["age0"] + t_idx)
                    if USE_NAV_PROJECTIONS and omega_sim_id in age_map:
                        age_q = age_map[omega_sim_id].get((fid, qe), age_q)
                        age_q = int(round(float(age_q)))
                    age_bucket = make_age_bucket_q(age_q)

                    # annual grade transition (disabled when using NAV projections)
                    if not USE_NAV_PROJECTIONS:
                        if t_idx > 0 and t_idx % 4 == 0:
                            P = get_transition_matrix(st["strategy"])
                            st["grade"] = sample_next_grade(st["grade"], P, rng)

                    grade = st["grade"]
                    if USE_NAV_PROJECTIONS and omega_sim_id in grade_map:
                        grade = grade_map[omega_sim_id].get((fid, qe), grade)
                        st["grade"] = grade

                    strategy = st["strategy"]
                    if USE_NAV_PROJECTIONS and omega_sim_id in strategy_map:
                        strategy = strategy_map[omega_sim_id].get((fid, qe), strategy)
                        st["strategy"] = strategy

                    # omega
                    if USE_NAV_PROJECTIONS:
                        omega = float(omega_map.get(omega_sim_id, {}).get((fid, qe), 0.0))
                    else:
                        a, b0, b1 = get_betas(strategy, grade)
                        alpha, _ = get_alpha(strategy, grade, str(age_bucket))
                        sigma, _ = get_sigma(strategy, grade)
                        eps = rng.standard_normal()
                        omega = alpha + b0 * msci_r + b1 * msci_r_lag1 + sigma * eps
                        omega += float(GRADE_OMEGA_BIAS.get(grade, 0.0))
                        omega = float(np.clip(omega, -OMEGA_CLIP, OMEGA_CLIP))

                    # capacity
                    ledger = st["ledger"]
                    start_qe = fund_start_qe[fid]
                    if USE_NAV_PROJECTIONS:
                        step = t_idx + 1
                    else:
                        step = quarter_diff(qe, start_qe)
                    rc_avail_pre = ledger.available(step)
                    remaining_commit_pre = max(st["commitment"] - st["dd_commit"], 0.0)
                    capacity_pre = remaining_commit_pre + rc_avail_pre

                    params = lookup_params(strategy, grade, age_bucket)

                    # drawdown probability
                    ip_q = int(ip_by_strategy.get(strategy, IP_Q_DEFAULT))
                    draw_mult = 1.0
                    if USE_DRAW_AGE_SHAPE and ip_q > 0:
                        if ENFORCE_IP_LIMITS and age_q > ip_q:
                            draw_mult = 0.0
                        else:
                            frac = float(age_q) / float(ip_q)
                            draw_mult = max(DRAW_AGE_MIN_MULT, (1.0 - frac) ** DRAW_AGE_DECAY_POWER)

                    if USE_HAZARD_MODELS and beta_draw is not None:
                        Xr = build_feature_row(strategy, grade, age_q, 0.0, False, hazard_meta["draw_cols"], hazard_meta["draw_means"], hazard_meta["draw_stds"])
                        p_draw_base = float(_sigmoid(Xr @ beta_draw)[0])
                    else:
                        p_draw_base = float(params.get("p_draw", 0.0))

                    grade_key = grade if grade in GRADE_DRAW_P_MULT else "D"
                    p_draw_adj = p_draw_base * draw_mult * float(GRADE_DRAW_P_MULT.get(grade_key, 1.0))
                    p_draw_adj = min(p_draw_adj, 1.0)

                    i = fund_index[fid]
                    draw_event = (U["draw_event"][i] < p_draw_adj) and (capacity_pre > 0.0)
                    draw_amt = 0.0
                    use_rc = 0.0
                    use_commit = 0.0

                    if draw_event:
                        ratio = lognormal_from_u(params["mu_draw"], params["sig_draw"], float(U["draw_size"][i]))
                        ratio = float(np.clip(ratio, 0.0, 1.0))
                        ratio = min(ratio * float(GRADE_DRAW_SIZE_MULT.get(grade_key, 1.0)), 1.0)
                        if USE_DRAWDOWN_CALIBRATION:
                            scale = draw_scale_sg.get((strategy, grade),
                                                      draw_scale_s.get(strategy, draw_scale_g))
                            ratio = ratio * float(scale)
                        ratio = float(np.clip(ratio, 0.0, 1.0))
                        draw_amt = ratio * capacity_pre
                        cons = ledger.consume_for_drawdown(step, draw_amt)
                        use_rc = cons["use_rc"]
                        use_commit = cons["use_commitment"]
                        st["dd_commit"] += use_commit

                    # Optional: enforce full commitment call at IP end (aligns with Structural-cashflows)
                    if ENFORCE_IP_LIMITS and (not st["ip_catchup_done"]) and (age_q >= ip_q):
                        remaining_commit_after = max(st["commitment"] - st["dd_commit"], 0.0)
                        if remaining_commit_after > 0.0:
                            draw_amt += remaining_commit_after
                            use_commit += remaining_commit_after
                            st["dd_commit"] += remaining_commit_after
                            draw_event = True
                        st["ip_catchup_done"] = True

                    # repayment
                    NAV_prev = float(st["nav"])
                    rep_regular = 0.0
                    if USE_HAZARD_MODELS and beta_rep is not None:
                        Xr = build_feature_row(strategy, grade, age_q, np.log1p(abs(NAV_prev)), True, hazard_meta["rep_cols"], hazard_meta["rep_means"], hazard_meta["rep_stds"])
                        p_rep_base = float(_sigmoid(Xr @ beta_rep)[0])
                    else:
                        p_rep_base = float(params.get("p_rep", 0.0))

                    grade_key = grade if grade in GRADE_P_MULT else "D"
                    runoff_mult = float(runoff_mult_by_strategy.get(strategy, 1.0)) if USE_RUNOFF_CALIBRATION else 1.0

                    cap_qe = st["cap_qe"]
                    if USE_NAV_PROJECTIONS:
                        q_left = max(T - (t_idx + 1), 0)
                    else:
                        q_left = 9999
                        if pd.notna(cap_qe):
                            q_left = max(quarter_diff(cap_qe, qe), 0)

                    tail_factor = 0.0
                    if RUNOFF_Q > 0:
                        tail_factor = float(max(0.0, (RUNOFF_Q - q_left) / RUNOFF_Q))

                    p_rep_adj = 1.0 - (1.0 - p_rep_base) ** (1.0 + (REP_RAMP_P * runoff_mult) * tail_factor)
                    p_rep_adj = min(1.0, p_rep_adj + REP_RAMP_FLOOR * tail_factor)
                    p_rep_adj = min(1.0, p_rep_adj * float(GRADE_P_MULT.get(grade_key, 1.0)))
                    p_rep_adj = float(np.clip(p_rep_adj, 1e-6, 1.0 - 1e-6))
                    logit_p = np.log(p_rep_adj / (1.0 - p_rep_adj))
                    logit_p += MSCI_REP_P_BETA * msci_z_eff
                    p_rep_adj = float(1.0 / (1.0 + np.exp(-logit_p)))

                    rep_event = (U["rep_event"][i] < p_rep_adj) and (NAV_prev > NAV_EPS)
                    if rep_event:
                        rep_ratio = lognormal_from_u(params["mu_rep"], params["sig_rep"], float(U["rep_size"][i]))
                        rep_ratio = float(np.clip(rep_ratio, 0.0, 1.0))
                        rep_ratio = min(rep_ratio * (1.0 + (REP_RAMP_SIZE * runoff_mult) * tail_factor), 1.0)
                        rep_ratio = min(rep_ratio * float(GRADE_SIZE_MULT.get(grade_key, 1.0)), 1.0)
                        size_mult = max(0.0, 1.0 + MSCI_REP_SIZE_BETA * msci_z_eff)
                        rep_ratio = min(rep_ratio * size_mult, 1.0)
                        rep_regular = rep_ratio * NAV_prev

                    # recallable
                    rc_added = 0.0
                    rc_event = (rep_regular > 0.0) and (U["rc_event"][i] < params["p_rc_given_rep"])
                    if rc_event:
                        rc_ratio = lognormal_from_u(params["mu_rc"], params["sig_rc"], float(U["rc_size"][i]))
                        rc_ratio = float(np.clip(rc_ratio, 0.0, 1.0))
                        rc_amt_raw = rc_ratio * rep_regular
                        rc_added = ledger.add_recallable(step, rc_amt_raw, enforce_cap=True)

                    # NAV update
                    available_nav = max(NAV_prev + float(draw_amt) - float(rep_regular), 0.0)
                    rep_terminal = 0.0
                    if USE_FORCED_TERMINAL_REPAY and RUNOFF_Q > 0 and q_left < RUNOFF_Q:
                        base_ratio = q_left / (q_left + 1.0)
                        target_nav = available_nav * (base_ratio ** max(runoff_mult, 0.0))
                        rep_terminal = max(0.0, available_nav - target_nav)

                    nav_after_flow = max(available_nav - rep_terminal, 0.0)
                    nav_after_val = nav_after_flow * (1.0 + float(omega))
                    if not np.isfinite(nav_after_val):
                        nav_after_val = 0.0
                    nav_after_val = max(float(nav_after_val), 0.0)

                    st["nav"] = nav_after_val

                    if q_left == 0 or nav_after_val <= NAV_STOP_EPS:
                        st["alive"] = False

                    # aggregates
                    bkey = fund_bucket[fid]
                    bidx = bucket_index[bkey]

                    sim_draw[s_idx, t_idx] += draw_amt
                    sim_rep[s_idx, t_idx] += (rep_regular + rep_terminal)
                    sim_nav[s_idx, t_idx] += nav_after_val

                    sim_draw_b[s_idx, t_idx, bidx] += draw_amt
                    sim_rep_b[s_idx, t_idx, bidx] += (rep_regular + rep_terminal)
                    sim_nav_b[s_idx, t_idx, bidx] += nav_after_val

        # build portfolio series table
        q_end = pd.Index(test_quarters, name="quarter_end")
        portfolio_series = pd.DataFrame({
            "quarter_end": q_end,
            "actual_draw": actual_draw,
            "actual_rep": actual_rep,
            "actual_nav": actual_nav,
            "sim_draw_mean": sim_draw.mean(axis=0),
            "sim_rep_mean": sim_rep.mean(axis=0),
            "sim_nav_mean": sim_nav.mean(axis=0),
            "sim_draw_p05": np.quantile(sim_draw, 0.05, axis=0),
            "sim_draw_p95": np.quantile(sim_draw, 0.95, axis=0),
            "sim_rep_p05": np.quantile(sim_rep, 0.05, axis=0),
            "sim_rep_p95": np.quantile(sim_rep, 0.95, axis=0),
            "sim_nav_p05": np.quantile(sim_nav, 0.05, axis=0),
            "sim_nav_p95": np.quantile(sim_nav, 0.95, axis=0),
        })

        # portfolio summary
        def _metrics(sim_mean, sim_p05, sim_p95, actual):
            err = sim_mean - actual
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            bias = float(np.mean(err))
            coverage = float(np.mean((actual >= sim_p05) & (actual <= sim_p95)))
            return rmse, mae, bias, coverage

        draw_rmse, draw_mae, draw_bias, draw_cov = _metrics(
            portfolio_series["sim_draw_mean"].to_numpy(),
            portfolio_series["sim_draw_p05"].to_numpy(),
            portfolio_series["sim_draw_p95"].to_numpy(),
            portfolio_series["actual_draw"].to_numpy(),
        )
        rep_rmse, rep_mae, rep_bias, rep_cov = _metrics(
            portfolio_series["sim_rep_mean"].to_numpy(),
            portfolio_series["sim_rep_p05"].to_numpy(),
            portfolio_series["sim_rep_p95"].to_numpy(),
            portfolio_series["actual_rep"].to_numpy(),
        )
        nav_rmse, nav_mae, nav_bias, nav_cov = _metrics(
            portfolio_series["sim_nav_mean"].to_numpy(),
            portfolio_series["sim_nav_p05"].to_numpy(),
            portfolio_series["sim_nav_p95"].to_numpy(),
            portfolio_series["actual_nav"].to_numpy(),
        )

        portfolio_summary = pd.DataFrame([
            {
                "scenario": scenario,
                "bucket": "PORTFOLIO",
                "n_funds": len(fund_ids),
                "draw_rmse": draw_rmse, "draw_mae": draw_mae, "draw_bias": draw_bias, "draw_cov_90": draw_cov,
                "rep_rmse": rep_rmse, "rep_mae": rep_mae, "rep_bias": rep_bias, "rep_cov_90": rep_cov,
                "nav_rmse": nav_rmse, "nav_mae": nav_mae, "nav_bias": nav_bias, "nav_cov_90": nav_cov,
            }
        ])

        # bucket summary
        bucket_rows = []
        for bkey, bidx in bucket_index.items():
            sim_draw_m = sim_draw_b[:, :, bidx].mean(axis=0)
            sim_rep_m = sim_rep_b[:, :, bidx].mean(axis=0)
            sim_nav_m = sim_nav_b[:, :, bidx].mean(axis=0)

            sim_draw_p05 = np.quantile(sim_draw_b[:, :, bidx], 0.05, axis=0)
            sim_draw_p95 = np.quantile(sim_draw_b[:, :, bidx], 0.95, axis=0)

            sim_rep_p05 = np.quantile(sim_rep_b[:, :, bidx], 0.05, axis=0)
            sim_rep_p95 = np.quantile(sim_rep_b[:, :, bidx], 0.95, axis=0)

            sim_nav_p05 = np.quantile(sim_nav_b[:, :, bidx], 0.05, axis=0)
            sim_nav_p95 = np.quantile(sim_nav_b[:, :, bidx], 0.95, axis=0)

            act_draw = actual_draw_b[:, bidx]
            act_rep = actual_rep_b[:, bidx]
            act_nav = actual_nav_b[:, bidx]

            d_rmse, d_mae, d_bias, d_cov = _metrics(sim_draw_m, sim_draw_p05, sim_draw_p95, act_draw)
            r_rmse, r_mae, r_bias, r_cov = _metrics(sim_rep_m, sim_rep_p05, sim_rep_p95, act_rep)
            n_rmse, n_mae, n_bias, n_cov = _metrics(sim_nav_m, sim_nav_p05, sim_nav_p95, act_nav)

            n_funds_bucket = sum(1 for f, b in fund_bucket.items() if b == bkey)

            bucket_rows.append({
                "scenario": scenario,
                "strategy": bkey[0],
                "grade": bkey[1],
                "age_bucket": bkey[2],
                "n_funds": n_funds_bucket,
                "draw_rmse": d_rmse, "draw_mae": d_mae, "draw_bias": d_bias, "draw_cov_90": d_cov,
                "rep_rmse": r_rmse, "rep_mae": r_mae, "rep_bias": r_bias, "rep_cov_90": r_cov,
                "nav_rmse": n_rmse, "nav_mae": n_mae, "nav_bias": n_bias, "nav_cov_90": n_cov,
            })

        bucket_summary = pd.DataFrame(bucket_rows)
        return portfolio_series, portfolio_summary, bucket_summary

    # Run conditional / unconditional as configured
    cond_series = cond_portfolio = cond_bucket = None
    uncond_series = uncond_portfolio = uncond_bucket = None

    if RUN_CONDITIONAL:
        print("Running conditional backtest (actual MSCI)...")
        cond_series, cond_portfolio, cond_bucket = run_backtest("conditional", conditional=True)

    if RUN_UNCONDITIONAL:
        print("Running unconditional backtest (simulated MSCI)...")
        uncond_series, uncond_portfolio, uncond_bucket = run_backtest(scenario_uncond, conditional=False)

    # =============================
    # Omega + fund coverage checks
    # =============================
    print("=== Backtest validation ===")
    try:
        # Fund coverage in test window
        test_funds_all = set(test["FundID"].unique().tolist())
        used_funds = set(fund_states.keys())
        print("Funds in test window:", len(test_funds_all))
        print("Funds used in backtest:", len(used_funds))
        if USE_NAV_PROJECTIONS:
            nav_funds = set(nav_start_by_fund.keys())
            omega_funds = set(omega_df["FundID"].unique()) if omega_df is not None else set()
            print("Funds with nav_start:", len(nav_funds))
            print("Funds with omega:", len(omega_funds))
            print("Overlap (test & nav & omega):", len(used_funds))
        # Funds that begin after train end (excluded from NAV projections)
        fc = data.groupby("FundID")["first_close_qe"].min()
        n_post_train = int((fc > train_end_qe).sum())
        print("Funds with first_close after train end:", n_post_train)
        if USE_NAV_PROJECTIONS and navstart is not None:
            ns = navstart.set_index("FundID").reindex(list(used_funds))
            nav_zero = int((ns["NAV_start"] <= NAV_EPS).sum())
            nav_total = int(ns["NAV_start"].notna().sum())
            print("NAV_start <= NAV_EPS among used funds:", nav_zero, "/", nav_total)
    except Exception as e:
        print("Fund coverage check failed:", repr(e))

    try:
        if omega_df is None:
            print("Omega validation skipped (omega_df missing).")
        else:
            # Actual omega (test window)
            df = data.sort_values(["FundID", "quarter_end"]).copy()
            df["nav_prev"] = df.groupby("FundID")["NAV Adjusted EUR"].shift(1)
            df["flow_net"] = df["Adj Drawdown EUR"] - df["Adj Repayment EUR"]
            m = df["nav_prev"].abs() > NAV_EPS
            df["omega_actual"] = np.nan
            df.loc[m, "omega_actual"] = (
                (df.loc[m, "NAV Adjusted EUR"] - df.loc[m, "nav_prev"]) - df.loc[m, "flow_net"]
            ) / df.loc[m, "nav_prev"]

            df_test = df[(df["quarter_end"] >= test_start_qe) &
                         (df["quarter_end"] <= test_end_qe) &
                         (df["FundID"].isin(used_funds))].copy()
            df_test = df_test.dropna(subset=["omega_actual"])

            # Projected omega (align to test window + used funds)
            om = omega_df.copy()
            om = om[(om["quarter_end"] >= test_start_qe) & (om["quarter_end"] <= test_end_qe)]
            om = om[om["FundID"].isin(used_funds)]
            # If multiple sim_id, average omega across sim_id for a mean path
            if "sim_id" in om.columns:
                om = (om.groupby(["FundID", "quarter_end"], as_index=False)
                        .agg({"omega": "mean", "Adj Strategy": "last", "Grade": "last"}))

            # Ensure strategy/grade columns
            if "Adj Strategy" not in om.columns or "Grade" not in om.columns:
                om = om.merge(
                    df_test[["FundID", "quarter_end", "Adj Strategy", "Grade"]],
                    on=["FundID", "quarter_end"],
                    how="left"
                )

            # Overall mean comparison
            act_mean = float(df_test["omega_actual"].mean()) if len(df_test) else float("nan")
            proj_mean = float(om["omega"].mean()) if len(om) else float("nan")
            print("Omega mean (actual):", round(act_mean, 6))
            print("Omega mean (projected):", round(proj_mean, 6))
            print("Omega mean diff (proj - actual):", round(proj_mean - act_mean, 6))

            # Strategy/grade comparison (top gaps)
            act_g = (df_test.groupby(["Adj Strategy", "Grade"])["omega_actual"]
                     .mean().reset_index().rename(columns={"omega_actual": "omega_actual_mean"}))
            proj_g = (om.groupby(["Adj Strategy", "Grade"])["omega"]
                      .mean().reset_index().rename(columns={"omega": "omega_proj_mean"}))
            comp = act_g.merge(proj_g, on=["Adj Strategy", "Grade"], how="outer")
            comp["omega_gap"] = comp["omega_proj_mean"] - comp["omega_actual_mean"]
            comp = comp.sort_values("omega_gap")
            print("Top omega gaps (proj - actual) by strategy/grade:")
            print(comp.head(8).to_string(index=False))
    except Exception as e:
        print("Omega validation failed:", repr(e))
    print("=========================")

    # Save
    def _yq(qe: pd.Timestamp) -> str:
        return f"{qe.year}_Q{qe.quarter}"

    test_end_tag = _yq(test_end_qe)
    out_cond_series = os.path.join(DATA_DIR, f"backtest_portfolio_series_conditional_{train_year}_{train_quarter}_to_{test_end_tag}.csv")
    out_uncond_series = os.path.join(DATA_DIR, f"backtest_portfolio_series_unconditional_{train_year}_{train_quarter}_to_{test_end_tag}.csv")
    out_cond_port = os.path.join(DATA_DIR, f"backtest_portfolio_summary_conditional_{train_year}_{train_quarter}_to_{test_end_tag}.csv")
    out_uncond_port = os.path.join(DATA_DIR, f"backtest_portfolio_summary_unconditional_{train_year}_{train_quarter}_to_{test_end_tag}.csv")
    out_cond_bucket = os.path.join(DATA_DIR, f"backtest_bucket_summary_conditional_{train_year}_{train_quarter}_to_{test_end_tag}.csv")
    out_uncond_bucket = os.path.join(DATA_DIR, f"backtest_bucket_summary_unconditional_{train_year}_{train_quarter}_to_{test_end_tag}.csv")

    print("Saved:")
    if RUN_CONDITIONAL and cond_series is not None:
        cond_series.to_csv(out_cond_series, index=False)
        cond_portfolio.to_csv(out_cond_port, index=False)
        cond_bucket.to_csv(out_cond_bucket, index=False)
        print(out_cond_series)
        print(out_cond_port)
        print(out_cond_bucket)
    if RUN_UNCONDITIONAL and uncond_series is not None:
        uncond_series.to_csv(out_uncond_series, index=False)
        uncond_portfolio.to_csv(out_uncond_port, index=False)
        uncond_bucket.to_csv(out_uncond_bucket, index=False)
        print(out_uncond_series)
        print(out_uncond_port)
        print(out_uncond_bucket)

    print("Runtime (seconds):", round(time.perf_counter() - t0, 2))


if __name__ == "__main__":
    main()
