from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from attack import TARGET_AD_ID, compute_budget_spent


# =====================================
# Config
# =====================================

INITIAL_BUDGET = 100.0
CPC = 2.0
DEFAULT_OUTPUT_DIR = Path("outputs_defense")

# Layer 1 thresholds / controls
L1_BUCKET_CAPACITY = 4
L1_BUCKET_REFILL_SECONDS = 15
L1_CHALLENGE_FAIL_THRESHOLD = 2

# Layer 2 thresholds
L2_WITHHOLD_THRESHOLD = 0.60
L2_REFUND_THRESHOLD = 0.85
RANDOM_STATE = 42

# Layer 3
L3_DYNAMIC_CAP_RATIO = 0.5


# =====================================
# Utilities
# =====================================

@dataclass
class BudgetState:
    advertiser_id: str
    ad_id: str
    initial_budget: float
    budget_left: float
    spent_amount: float = 0.0
    withheld_amount: float = 0.0
    refunded_amount: float = 0.0
    cap_active: bool = False
    cap_limit: Optional[float] = None
    capped_spent: float = 0.0


class IPState:
    TRUSTED = "TRUSTED"
    CHALLENGED = "CHALLENGED"
    BLOCKED = "BLOCKED"


class TokenBucket:
    def __init__(self, capacity: int, refill_rate_seconds: int):
        self.capacity = capacity
        self.refill_rate_seconds = refill_rate_seconds
        self.tokens = capacity
        self.last_refill_time = None

    def consume(self, current_time) -> bool:
        if self.last_refill_time is None:
            self.last_refill_time = current_time

        elapsed = (current_time - self.last_refill_time).total_seconds()
        tokens_to_add = int(elapsed / self.refill_rate_seconds)

        if tokens_to_add > 0:
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill_time += pd.Timedelta(seconds=tokens_to_add * self.refill_rate_seconds)

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


def ensure_output_dir(output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =====================================
# Click feature preparation
# =====================================

def prepare_click_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

    clicks = raw_df[raw_df["event_type"] == "click"].copy()
    clicks = clicks.sort_values(["device_id", "timestamp"]).reset_index(drop=True)

    device_click_counts = clicks.groupby("device_id").size().to_dict()
    clicks["click_frequency_device"] = clicks["device_id"].map(device_click_counts)

    ip_click_counts = clicks.groupby("ip").size().to_dict()
    clicks["click_frequency_ip"] = clicks["ip"].map(ip_click_counts)

    clicks["click_interval_sec"] = (
        clicks.groupby("device_id")["timestamp"].diff().dt.total_seconds()
    )

    def interval_label(x: float) -> str:
        if pd.isna(x):
            return "Irregular"
        return "Regular" if x <= 5 else "Irregular"

    clicks["interval_pattern"] = clicks["click_interval_sec"].apply(interval_label)

    dwell_records = []
    for device_id, group in raw_df.sort_values("timestamp").groupby("device_id"):
        click_times = group[group["event_type"] == "click"]["timestamp"].tolist()
        leave_times = group[group["event_type"] == "leave"]["timestamp"].tolist()

        for i, click_time in enumerate(click_times):
            dwell = None
            if i < len(leave_times):
                dwell = (leave_times[i] - click_time).total_seconds()
            dwell_records.append((device_id, click_time, dwell))

    dwell_df = pd.DataFrame(dwell_records, columns=["device_id", "timestamp", "avg_dwell_time_sec"])
    clicks = clicks.merge(dwell_df, on=["device_id", "timestamp"], how="left")

    same_ad_conc_map = {}
    for device_id, group in clicks.groupby("device_id"):
        ad_ratio = group["ad_id"].value_counts(normalize=True).to_dict()
        for idx, row in group.iterrows():
            same_ad_conc_map[idx] = ad_ratio.get(row["ad_id"], 0.0)

    clicks["same_ad_concentration"] = clicks.index.map(same_ad_conc_map)
    return clicks


# =====================================
# Layer 1: sequential identity / access filtering
# =====================================

def layer1_filter(click_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    [LATEST UPDATE]
    由原来的简单阈值 block，扩展为你同学版本里的 sequential Layer 1：
    Token Bucket + CHALLENGED/BLOCKED 状态机。
    同时接口保持 DataFrame in / DataFrame out，不依赖 CSV。
    """
    df = click_df.copy().sort_values("timestamp").reset_index(drop=True)

    ip_records: Dict[str, Dict[str, object]] = {}
    actions = []
    risk_levels = []
    time_since_last_clicks = []

    for _, row in df.iterrows():
        ip = row["ip"]
        current_time = row["timestamp"]

        if ip not in ip_records:
            ip_records[ip] = {
                "bucket": TokenBucket(capacity=L1_BUCKET_CAPACITY, refill_rate_seconds=L1_BUCKET_REFILL_SECONDS),
                "state": IPState.TRUSTED,
                "captcha_failures": 0,
                "last_click_time": current_time,
            }
            delta = 0.0
        else:
            delta = (current_time - ip_records[ip]["last_click_time"]).total_seconds()
            ip_records[ip]["last_click_time"] = current_time

        time_since_last_clicks.append(delta)
        record = ip_records[ip]

        if record["state"] == IPState.BLOCKED:
            actions.append("BLOCKED (Blacklisted)")
            risk_levels.append("High")
            continue

        if record["state"] == IPState.CHALLENGED:
            if delta < 5.0:
                record["captcha_failures"] += 1
                actions.append("BLOCKED (CAPTCHA Failed)")
                risk_levels.append("High")
                if record["captcha_failures"] >= L1_CHALLENGE_FAIL_THRESHOLD:
                    record["state"] = IPState.BLOCKED
            else:
                record["state"] = IPState.TRUSTED
                record["bucket"].tokens = record["bucket"].capacity
                record["captcha_failures"] = 0
                actions.append("ALLOWED (CAPTCHA Passed)")
                risk_levels.append("Medium")
            continue

        if record["bucket"].consume(current_time):
            actions.append("ALLOWED")
            risk_levels.append("Low")
        else:
            record["state"] = IPState.CHALLENGED
            actions.append("CHALLENGED (Rate Limit -> CAPTCHA Required)")
            risk_levels.append("Medium")

    df["layer1_action"] = actions
    df["ip_risk_level"] = risk_levels
    df["time_since_last_click_sec"] = time_since_last_clicks
    df["layer1_pass"] = ~df["layer1_action"].astype(str).str.contains("BLOCKED", na=False)

    passed_df = df[df["layer1_pass"]].copy()
    metrics = {
        "total_incoming": len(df),
        "blocked_or_dropped": int((~df["layer1_pass"]).sum()),
        "passed_to_layer2": len(passed_df),
        "challenged": int(df["layer1_action"].astype(str).str.contains("CHALLENGED", na=False).sum()),
    }
    return df, passed_df, metrics


# =====================================
# Layer 2: ML behavioural detection
# =====================================

def extract_ip_features_for_layer2(df: pd.DataFrame) -> pd.DataFrame:
    feature_rows = []

    for ip, group in df.groupby("ip"):
        group = group.sort_values("timestamp").copy()
        group["time_diff"] = group["timestamp"].diff().dt.total_seconds()
        intervals = group["time_diff"].dropna()

        click_count = float(len(group))
        duration = float((group["timestamp"].max() - group["timestamp"].min()).total_seconds()) if len(group) > 1 else 0.0
        mean_interval = float(intervals.mean()) if not intervals.empty else 0.0
        std_interval = float(intervals.std()) if not intervals.empty else 0.0
        min_interval = float(intervals.min()) if not intervals.empty else 0.0
        max_interval = float(intervals.max()) if not intervals.empty else 0.0
        click_rate = click_count / duration if duration > 0 else click_count

        avg_time_since_last_click = float(group["time_since_last_click_sec"].mean()) if "time_since_last_click_sec" in group.columns else 0.0
        challenge_count = float(group["layer1_action"].astype(str).str.contains("CHALLENGED", na=False).sum()) if "layer1_action" in group.columns else 0.0
        medium_risk_count = float((group["ip_risk_level"] == "Medium").sum()) if "ip_risk_level" in group.columns else 0.0

        avg_dwell_time = float(group["avg_dwell_time_sec"].fillna(0).mean()) if "avg_dwell_time_sec" in group.columns else 0.0
        same_ad_concentration = float(group["same_ad_concentration"].fillna(0).mean()) if "same_ad_concentration" in group.columns else 0.0
        device_click_frequency = float(group["click_frequency_device"].fillna(0).mean()) if "click_frequency_device" in group.columns else 0.0

        threat_ratio = float((group["traffic_type"] != "normal").mean()) if "traffic_type" in group.columns else 0.0
        label = 1 if threat_ratio >= 0.5 else 0

        feature_rows.append(
            {
                "ip": ip,
                "click_count": click_count,
                "mean_interval": mean_interval,
                "std_interval": 0.0 if pd.isna(std_interval) else std_interval,
                "min_interval": min_interval,
                "max_interval": max_interval,
                "duration": duration,
                "click_rate": float(click_rate),
                "avg_time_since_last_click": avg_time_since_last_click,
                "challenge_count": challenge_count,
                "medium_risk_count": medium_risk_count,
                "avg_dwell_time_sec": avg_dwell_time,
                "same_ad_concentration": same_ad_concentration,
                "device_click_frequency": device_click_frequency,
                "threat_ratio": threat_ratio,
                "label": label,
            }
        )

    return pd.DataFrame(feature_rows)


def compute_rule_based_fraud_score(row: pd.Series) -> float:
    score = 0.0

    if row["device_click_frequency"] >= 15:
        score += 0.25
    elif row["device_click_frequency"] >= 8:
        score += 0.15

    dwell = row.get("avg_dwell_time_sec", 0.0)
    if pd.notna(dwell):
        if dwell <= 1:
            score += 0.25
        elif dwell <= 3:
            score += 0.15

    mean_interval = row.get("mean_interval", 0.0)
    if mean_interval <= 5:
        score += 0.20

    conc = row.get("same_ad_concentration", 0.0)
    if conc >= 0.8:
        score += 0.20
    elif conc >= 0.5:
        score += 0.10

    challenge_count = row.get("challenge_count", 0.0)
    if challenge_count >= 2:
        score += 0.10

    return min(score, 1.0)


def assign_layer2_action(ip_feature_df: pd.DataFrame) -> pd.DataFrame:
    df = ip_feature_df.copy()
    high_threshold = df["fraud_score"].quantile(0.85) if not df.empty else 1.0
    medium_threshold = df["fraud_score"].quantile(0.60) if not df.empty else 0.0

    def get_risk_level(score: float) -> str:
        if score >= high_threshold:
            return "high"
        if score >= medium_threshold:
            return "medium"
        return "low"

    def get_action(risk: str) -> str:
        if risk == "high":
            return "block_and_withhold_billing"
        if risk == "medium":
            return "flag_for_review"
        return "allow"

    df["risk_level"] = df["fraud_score"].apply(get_risk_level)
    df["layer2_action"] = df["risk_level"].apply(get_action)
    df.loc[df["fraud_score"] >= L2_WITHHOLD_THRESHOLD, "layer2_action"] = "block_and_withhold_billing"
    return df


def run_layer2_df(click_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    [LATEST UPDATE]
    融合你同学 Layer 2 的 IP 级 ML 思路，同时保留你原先可解释的规则分数。
    最终输出是 IP 级结果，供 Layer 3 再映射回 click 级。
    """
    feature_df = extract_ip_features_for_layer2(click_df)
    if feature_df.empty:
        return feature_df.copy(), {"ip_rows": 0, "high_risk_ips": 0}

    model_features = feature_df.drop(columns=["ip", "label", "threat_ratio"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_features)

    iso_model = IsolationForest(contamination=0.2, random_state=RANDOM_STATE)
    iso_model.fit(X_scaled)
    iso_score = -iso_model.decision_function(X_scaled)

    feature_df["rule_fraud_score"] = feature_df.apply(compute_rule_based_fraud_score, axis=1)

    if np.max(iso_score) - np.min(iso_score) < 1e-9:
        feature_df["iso_fraud_score"] = 0.0
    else:
        feature_df["iso_fraud_score"] = (iso_score - np.min(iso_score)) / (np.max(iso_score) - np.min(iso_score))

    feature_df["fraud_score"] = 0.55 * feature_df["rule_fraud_score"] + 0.45 * feature_df["iso_fraud_score"]
    feature_df = assign_layer2_action(feature_df)

    metrics = {
        "ip_rows": len(feature_df),
        "high_risk_ips": int((feature_df["risk_level"] == "high").sum()),
        "medium_risk_ips": int((feature_df["risk_level"] == "medium").sum()),
        "mean_fraud_score": round(float(feature_df["fraud_score"].mean()), 4),
    }
    return feature_df, metrics


# =====================================
# Layer 3: Budget-aware Control
# =====================================

def initialize_budget_states(click_df: pd.DataFrame, initial_budget: float) -> Dict[str, BudgetState]:
    states = {}
    pairs = click_df[["advertiser_id", "ad_id"]].drop_duplicates()

    for _, row in pairs.iterrows():
        key = f"{row['advertiser_id']}::{row['ad_id']}"
        states[key] = BudgetState(
            advertiser_id=row["advertiser_id"],
            ad_id=row["ad_id"],
            initial_budget=initial_budget,
            budget_left=initial_budget,
        )
    return states


def attach_layer2_results(click_df: pd.DataFrame, layer2_ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    [LATEST UPDATE]
    解决 click 级 Layer 1 / Layer 3 与 IP 级 Layer 2 之间的接口断层。
    """
    df = click_df.copy()
    if layer2_ip_df.empty:
        df["fraud_score"] = 0.0
        df["risk_level"] = "low"
        df["layer2_action"] = "allow"
        return df

    cols = ["ip", "fraud_score", "risk_level", "layer2_action", "rule_fraud_score", "iso_fraud_score"]
    available_cols = [c for c in cols if c in layer2_ip_df.columns]
    return df.merge(layer2_ip_df[available_cols], on="ip", how="left")


def layer3_budget_control(
    click_df: pd.DataFrame,
    initial_budget: float = INITIAL_BUDGET,
    cpc: float = CPC,
    cap_ratio: float = L3_DYNAMIC_CAP_RATIO,
) -> pd.DataFrame:
    df = click_df.copy().sort_values("timestamp").reset_index(drop=True)

    df["billing_action"] = "charge"
    df["charged_amount"] = 0.0
    df["withheld_amount"] = 0.0
    df["refunded_amount"] = 0.0
    df["cap_active"] = False
    df["budget_left_after"] = None

    states = initialize_budget_states(df, initial_budget=initial_budget)
    suspicious_counter: Dict[str, int] = {}
    total_counter: Dict[str, int] = {}

    for idx, row in df.iterrows():
        key = f"{row['advertiser_id']}::{row['ad_id']}"
        state = states[key]

        total_counter[key] = total_counter.get(key, 0) + 1
        if row.get("layer2_action", "allow") == "block_and_withhold_billing":
            suspicious_counter[key] = suspicious_counter.get(key, 0) + 1

        suspicious_ratio = suspicious_counter.get(key, 0) / total_counter[key]

        if suspicious_ratio >= 0.5 and not state.cap_active:
            state.cap_active = True
            state.cap_limit = state.initial_budget * cap_ratio

        df.at[idx, "cap_active"] = state.cap_active

        if not row.get("layer1_pass", True):
            df.at[idx, "billing_action"] = "blocked"
            df.at[idx, "budget_left_after"] = state.budget_left
            continue

        if row.get("layer2_action", "allow") == "block_and_withhold_billing":
            df.at[idx, "billing_action"] = "withhold"
            df.at[idx, "withheld_amount"] = cpc
            state.withheld_amount += cpc
            df.at[idx, "budget_left_after"] = state.budget_left
            continue

        if state.cap_active and state.cap_limit is not None:
            if state.capped_spent + cpc > state.cap_limit:
                df.at[idx, "billing_action"] = "cap_block"
                df.at[idx, "budget_left_after"] = state.budget_left
                continue

        df.at[idx, "billing_action"] = "charge"
        df.at[idx, "charged_amount"] = cpc
        state.spent_amount += cpc
        state.budget_left = max(state.budget_left - cpc, 0.0)

        if state.cap_active:
            state.capped_spent += cpc

        if row.get("fraud_score", 0.0) >= L2_REFUND_THRESHOLD:
            df.at[idx, "billing_action"] = "refund"
            df.at[idx, "charged_amount"] = 0.0
            df.at[idx, "refunded_amount"] = cpc
            state.refunded_amount += cpc
            state.spent_amount = max(state.spent_amount - cpc, 0.0)
            state.budget_left = min(state.budget_left + cpc, state.initial_budget)

        df.at[idx, "budget_left_after"] = state.budget_left

    return df


# =====================================
# End-to-end defense pipeline
# =====================================

def run_defense_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    click_df = prepare_click_features(raw_df)
    layer1_full_df, layer1_passed_df, _ = layer1_filter(click_df)
    layer2_ip_df, _ = run_layer2_df(layer1_passed_df)
    click_with_l2 = attach_layer2_results(layer1_full_df, layer2_ip_df)
    defense_df = layer3_budget_control(click_with_l2)
    return defense_df


def run_defense_pipeline_detailed(raw_df: pd.DataFrame) -> Dict[str, object]:
    """
    [LATEST UPDATE]
    main.py 可直接拿到各层中间结果，不必依赖 CSV 中转。
    """
    click_df = prepare_click_features(raw_df)
    layer1_full_df, layer1_passed_df, layer1_metrics = layer1_filter(click_df)
    layer2_ip_df, layer2_metrics = run_layer2_df(layer1_passed_df)
    click_with_l2 = attach_layer2_results(layer1_full_df, layer2_ip_df)
    defense_df = layer3_budget_control(click_with_l2)

    return {
        "click_df": click_df,
        "layer1_full_df": layer1_full_df,
        "layer1_passed_df": layer1_passed_df,
        "layer1_metrics": layer1_metrics,
        "layer2_ip_df": layer2_ip_df,
        "layer2_metrics": layer2_metrics,
        "defense_df": defense_df,
    }


# =====================================
# Summaries
# =====================================

def summarize_defense_actions(defense_df: pd.DataFrame) -> pd.DataFrame:
    return (
        defense_df.groupby("billing_action")
        .agg(
            clicks=("billing_action", "count"),
            charged_amount=("charged_amount", "sum"),
            withheld_amount=("withheld_amount", "sum"),
            refunded_amount=("refunded_amount", "sum"),
        )
        .reset_index()
    )


def summarize_by_traffic_type(defense_df: pd.DataFrame) -> pd.DataFrame:
    return (
        defense_df.groupby(["traffic_type", "billing_action"])
        .agg(
            clicks=("billing_action", "count"),
            charged_amount=("charged_amount", "sum"),
            withheld_amount=("withheld_amount", "sum"),
            refunded_amount=("refunded_amount", "sum"),
        )
        .reset_index()
    )


def summarize_budget_outcome(defense_df: pd.DataFrame) -> pd.DataFrame:
    return (
        defense_df.groupby(["advertiser_id", "ad_id"])
        .agg(
            total_clicks=("ad_id", "count"),
            total_charged=("charged_amount", "sum"),
            total_withheld=("withheld_amount", "sum"),
            total_refunded=("refunded_amount", "sum"),
            final_budget_left=("budget_left_after", "last"),
        )
        .reset_index()
    )


# =====================================
# Export helpers  [LATEST UPDATE]
# =====================================

def export_defense_artifacts(
    results: Dict[str, object],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Dict[str, str]:
    output_dir = ensure_output_dir(output_dir)
    saved: Dict[str, str] = {}

    csv_targets = {
        "layer1_full_df": "layer1_full_audit.csv",
        "layer1_passed_df": "layer1_passed_to_layer2.csv",
        "layer2_ip_df": "layer2_ip_results.csv",
        "defense_df": "defense_final_results.csv",
    }

    for key, filename in csv_targets.items():
        df = results.get(key)
        if isinstance(df, pd.DataFrame):
            path = output_dir / filename
            df.to_csv(path, index=False)
            saved[key] = str(path)

    return saved


# =====================================
# Visualizations
# =====================================

def plot_defense_action_distribution(defense_df: pd.DataFrame, save_path: Optional[Path | str] = None, show: bool = True) -> None:
    summary = summarize_defense_actions(defense_df)

    plt.figure(figsize=(8, 4))
    plt.bar(summary["billing_action"], summary["clicks"])
    plt.title("Defense Actions Distribution")
    plt.xlabel("Action")
    plt.ylabel("Number of Clicks")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_budget_after_defense(defense_df: pd.DataFrame, save_path: Optional[Path | str] = None, show: bool = True) -> None:
    plt.figure(figsize=(10, 4))

    for traffic_type in defense_df["traffic_type"].unique():
        sub = defense_df[defense_df["traffic_type"] == traffic_type].sort_values("timestamp")
        plt.plot(sub["timestamp"], sub["budget_left_after"], marker="o", label=traffic_type)

    plt.title("Budget After Defense Control")
    plt.xlabel("Time")
    plt.ylabel("Budget Left")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_attack_vs_defense(raw_df: pd.DataFrame, defense_df: pd.DataFrame, save_path: Optional[Path | str] = None, show: bool = True) -> None:
    attack_budget = (
        raw_df.groupby("traffic_type")
        .size()
        .reset_index(name="click_count")
    )
    attack_budget["budget_spent"] = attack_budget["click_count"] * CPC

    # After defense
    defense_budget = (
        defense_df.groupby("traffic_type")["charged_amount"]
        .sum()
        .reset_index(name="budget_spent")
    )

    # Align traffic types
    traffic_types = sorted(set(attack_budget["traffic_type"]) | set(defense_budget["traffic_type"]))

    attack_map = dict(zip(attack_budget["traffic_type"], attack_budget["budget_spent"]))
    defense_map = dict(zip(defense_budget["traffic_type"], defense_budget["budget_spent"]))

    before_vals = [attack_map.get(t, 0) for t in traffic_types]
    after_vals = [defense_map.get(t, 0) for t in traffic_types]

    x = range(len(traffic_types))
    width = 0.35

    plt.figure(figsize=(7, 4.5))
    plt.bar([i - width / 2 for i in x], before_vals, width=width, label="Before Defense")
    plt.bar([i + width / 2 for i in x], after_vals, width=width, label="After Defense")
    plt.xticks(list(x), traffic_types)
    plt.xlabel("Traffic Type")
    plt.ylabel("Budget Spent")
    plt.title("Budget Consumption Before vs After Defense")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_layer2_fraud_scores(layer2_ip_df: pd.DataFrame, save_path: Optional[Path | str] = None, show: bool = True) -> None:
    if layer2_ip_df.empty:
        return

    plot_df = layer2_ip_df.sort_values("fraud_score", ascending=False)
    plt.figure(figsize=(12, 5))
    plt.bar(plot_df["ip"].astype(str), plot_df["fraud_score"])
    plt.xticks(rotation=90)
    plt.xlabel("IP Address")
    plt.ylabel("Fraud Score")
    plt.title("Layer 2 Fraud Score by IP")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def save_defense_plots(
    raw_df: pd.DataFrame,
    results: Dict[str, object],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> Dict[str, str]:
    output_dir = ensure_output_dir(output_dir)
    saved: Dict[str, str] = {}

    defense_df = results["defense_df"]
    layer2_ip_df = results["layer2_ip_df"]

    path1 = output_dir / "defense_action_distribution.png"
    plot_defense_action_distribution(defense_df, save_path=path1, show=show)
    saved["defense_action_distribution"] = str(path1)

    path2 = output_dir / "budget_after_defense.png"
    plot_budget_after_defense(defense_df, save_path=path2, show=show)
    saved["budget_after_defense"] = str(path2)

    path3 = output_dir / "attack_vs_defense_budget.png"
    plot_attack_vs_defense(raw_df, defense_df, save_path=path3, show=show)
    saved["attack_vs_defense"] = str(path3)

    path4 = output_dir / "layer2_fraud_scores.png"
    plot_layer2_fraud_scores(layer2_ip_df, save_path=path4, show=show)
    saved["layer2_fraud_scores"] = str(path4)

    return saved
