import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd


# =========================
# 1. Config
# =========================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TARGET_AD_ID = "A1"
TARGET_ADVERTISER_ID = "ADV1"

OTHER_ADS = [
    ("A2", "ADV2"),
    ("A3", "ADV3"),
    ("A4", "ADV4"),
]

START_TIME = datetime(2026, 4, 1, 10, 0, 0)
DEFAULT_OUTPUT_DIR = Path("outputs_attack")
DEFAULT_RAW_CSV = "attack_raw_events.csv"
DEFAULT_PROCESSED_CSV = "attack_processed_features.csv"
DEFAULT_COMPARISON_CSV = "attack_comparison_table.csv"
DEFAULT_BUDGET_CSV = "attack_budget_simulation.csv"


# =========================
# 2. Data model
# =========================

@dataclass
class Event:
    timestamp: datetime
    device_id: str
    ip: str
    event_type: str   # "click" or "leave"
    ad_id: str
    advertiser_id: str
    traffic_type: str  # "normal", "competitive_fraud", "bot_fraud"


# =========================
# 3. Helpers
# =========================

def random_ip() -> str:
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def make_device_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:03d}"


def choose_normal_ad() -> Tuple[str, str]:
    return random.choice([(TARGET_AD_ID, TARGET_ADVERTISER_ID)] + OTHER_ADS)


def add_click_and_leave(
    events: List[Event],
    click_time: datetime,
    device_id: str,
    ip: str,
    ad_id: str,
    advertiser_id: str,
    dwell_seconds: int,
    traffic_type: str,
) -> None:
    events.append(
        Event(
            timestamp=click_time,
            device_id=device_id,
            ip=ip,
            event_type="click",
            ad_id=ad_id,
            advertiser_id=advertiser_id,
            traffic_type=traffic_type,
        )
    )

    leave_time = click_time + timedelta(seconds=dwell_seconds)
    events.append(
        Event(
            timestamp=leave_time,
            device_id=device_id,
            ip=ip,
            event_type="leave",
            ad_id=ad_id,
            advertiser_id=advertiser_id,
            traffic_type=traffic_type,
        )
    )


# =========================
# 4. Generate raw logs
# =========================

def generate_normal_traffic(
    num_devices: int = 30,
    clicks_per_device_range: Tuple[int, int] = (2, 6),
) -> List[Event]:
    events: List[Event] = []

    for i in range(num_devices):
        device_id = make_device_id("N", i)
        ip = random_ip()
        current_time = START_TIME + timedelta(minutes=random.randint(0, 20))
        num_clicks = random.randint(*clicks_per_device_range)

        for _ in range(num_clicks):
            ad_id, advertiser_id = choose_normal_ad()
            gap_seconds = random.randint(30, 300)
            current_time += timedelta(seconds=gap_seconds)
            dwell_seconds = random.randint(8, 60)

            add_click_and_leave(
                events=events,
                click_time=current_time,
                device_id=device_id,
                ip=ip,
                ad_id=ad_id,
                advertiser_id=advertiser_id,
                dwell_seconds=dwell_seconds,
                traffic_type="normal",
            )

    return events


def generate_competitive_fraud(
    num_devices: int = 6,
    clicks_per_device_range: Tuple[int, int] = (4, 10),
) -> List[Event]:
    events: List[Event] = []

    for i in range(num_devices):
        device_id = make_device_id("CF", i)
        ip = random_ip()
        current_time = START_TIME + timedelta(minutes=random.randint(5, 25))
        num_clicks = random.randint(*clicks_per_device_range)

        for _ in range(num_clicks):
            gap_seconds = random.randint(10, 90)
            current_time += timedelta(seconds=gap_seconds)
            dwell_seconds = random.randint(1, 5)

            add_click_and_leave(
                events=events,
                click_time=current_time,
                device_id=device_id,
                ip=ip,
                ad_id=TARGET_AD_ID,
                advertiser_id=TARGET_ADVERTISER_ID,
                dwell_seconds=dwell_seconds,
                traffic_type="competitive_fraud",
            )

    return events


# =========================
# generate_bot_fraud_with_theft
# =========================

def generate_bot_fraud_with_theft(
    normal_events: List[Event],  # 传入正常流量池用于“窃取”
    num_devices: int = 3,
    clicks_per_device: int = 30,
    theft_rate: float = 0.35     # 35% 的概率复用真实 IP
) -> List[Event]:
   
    events: List[Event] = []
    
    
    stolen_ip_vault = list(set([e.ip for e in normal_events if e.traffic_type == "normal"]))

    for i in range(num_devices):
        device_id = make_device_id("BOT_ADVANCED", i)
        
        base_ip = random_ip()
        current_time = START_TIME + timedelta(minutes=random.randint(8, 15))

        for click_idx in range(clicks_per_device):
            # IP resuse logic:
            # if we have a vault of stolen IPs and the random check passes, we pick one from the vault; otherwise, we use a new random IP for this bot device.
            if stolen_ip_vault and random.random() < theft_rate:
                active_ip = random.choice(stolen_ip_vault)
            else:
                active_ip = base_ip
            # --------------------------

            gap_seconds = random.choice([2, 3, 4])
            current_time += timedelta(seconds=gap_seconds)
            dwell_seconds = random.choice([0, 1])

            add_click_and_leave(
                events=events,
                click_time=current_time,
                device_id=device_id,
                ip=active_ip, 
                ad_id=TARGET_AD_ID,
                advertiser_id=TARGET_ADVERTISER_ID,
                dwell_seconds=dwell_seconds,
                traffic_type="bot_fraud",
            )

    return events

# =========================
#  build_raw_dataframe 
# =========================

def build_raw_dataframe() -> pd.DataFrame:
    events: List[Event] = []
    
    # 1.generate normal traffic
    normal_traffic = generate_normal_traffic()
    events.extend(normal_traffic)
    
    # 2.generate_competitive_fraud
    events.extend(generate_competitive_fraud())
    
    #3.generate_bot_fraud_with_theft need normal_traffic as input for potential IP theft
    events.extend(generate_bot_fraud_with_theft(normal_traffic))

    df = pd.DataFrame([e.__dict__ for e in events])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# =========================
# 5. Feature engineering
# =========================

def compute_dwell_time(df: pd.DataFrame) -> pd.DataFrame:
    result_rows = []

    for _, group in df.sort_values("timestamp").groupby(
        ["device_id", "ip", "ad_id", "advertiser_id", "traffic_type"]
    ):
        click_times = group[group["event_type"] == "click"]["timestamp"].tolist()
        leave_times = group[group["event_type"] == "leave"]["timestamp"].tolist()

        for c, l in zip(click_times, leave_times):
            result_rows.append(
                {
                    "timestamp": c,
                    "device_id": group["device_id"].iloc[0],
                    "ip": group["ip"].iloc[0],
                    "ad_id": group["ad_id"].iloc[0],
                    "advertiser_id": group["advertiser_id"].iloc[0],
                    "traffic_type": group["traffic_type"].iloc[0],
                    "dwell_time_sec": (l - c).total_seconds(),
                }
            )

    return pd.DataFrame(result_rows)


def compute_processed_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    click_df = raw_df[raw_df["event_type"] == "click"].copy()
    click_df = click_df.sort_values(["device_id", "timestamp"])

    click_df["click_interval_sec"] = (
        click_df.groupby("device_id")["timestamp"].diff().dt.total_seconds()
    )

    dwell_df = compute_dwell_time(raw_df)

    agg = (
        click_df.groupby(["device_id", "ad_id", "advertiser_id", "traffic_type"])
        .agg(
            total_clicks=("timestamp", "count"),
            unique_ips=("ip", "nunique"),
            avg_click_interval_sec=("click_interval_sec", "mean"),
        )
        .reset_index()
    )

    dwell_agg = (
        dwell_df.groupby(["device_id", "ad_id", "advertiser_id", "traffic_type"])
        .agg(avg_dwell_time_sec=("dwell_time_sec", "mean"))
        .reset_index()
    )

    processed = agg.merge(
        dwell_agg,
        on=["device_id", "ad_id", "advertiser_id", "traffic_type"],
        how="left",
    )

    processed["click_frequency_label"] = processed["total_clicks"].apply(
        lambda x: "High" if x >= 10 else "Low"
    )

    def interval_pattern(x: float) -> str:
        if pd.isna(x):
            return "N/A"
        if x <= 5:
            return "Regular"
        return "Irregular"

    processed["interval_pattern"] = processed["avg_click_interval_sec"].apply(interval_pattern)
    return processed


# =========================
# 6. Evidence / comparison
# =========================

def build_comparison_table(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> pd.DataFrame:
    del processed_df  # kept for compatibility with older calling style

    click_df = raw_df[raw_df["event_type"] == "click"].copy()
    dwell_df = compute_dwell_time(raw_df)

    def summarize(label_group: List[str], display_name: str) -> Dict[str, object]:
        clicks = click_df[click_df["traffic_type"].isin(label_group)].copy()
        dwell = dwell_df[dwell_df["traffic_type"].isin(label_group)].copy()

        click_per_device = clicks.groupby("device_id").size()
        avg_click_freq = click_per_device.mean() if not click_per_device.empty else 0
        same_ad_ratio = (clicks["ad_id"] == TARGET_AD_ID).mean() if not clicks.empty else 0
        avg_dwell = dwell["dwell_time_sec"].mean() if not dwell.empty else 0

        tmp = clicks.sort_values(["device_id", "timestamp"]).copy()
        tmp["interval_sec"] = tmp.groupby("device_id")["timestamp"].diff().dt.total_seconds()
        interval_regular_ratio = (
            (tmp["interval_sec"] <= 5).mean() if tmp["interval_sec"].notna().any() else 0
        )

        return {
            "Traffic": display_name,
            "Avg Clicks per Device": round(avg_click_freq, 2),
            "Avg Dwell Time (s)": round(avg_dwell, 2),
            "Same-ad Concentration": round(same_ad_ratio, 2),
            "Regular Interval Ratio": round(interval_regular_ratio, 2),
        }

    return pd.DataFrame(
        [
            summarize(["normal"], "Normal"),
            summarize(["competitive_fraud", "bot_fraud"], "Fraud"),
        ]
    )


# =========================
# 7. Budget simulation
# =========================

def simulate_budget(raw_df: pd.DataFrame, target_budget: float = 100.0, cpc: float = 2.0) -> pd.DataFrame:
    clicks = raw_df[
        (raw_df["event_type"] == "click")
        & (raw_df["ad_id"] == TARGET_AD_ID)
        & (raw_df["advertiser_id"] == TARGET_ADVERTISER_ID)
    ].copy().sort_values("timestamp")

    budget_left = target_budget
    rows = []

    for _, row in clicks.iterrows():
        budget_left = max(budget_left - cpc, 0)
        rows.append(
            {
                "timestamp": row["timestamp"],
                "traffic_type": row["traffic_type"],
                "budget_left": budget_left,
            }
        )
        if budget_left <= 0:
            break

    return pd.DataFrame(rows)


def compute_budget_spent(raw_df: pd.DataFrame, cpc: float = 2.0) -> pd.DataFrame:
    clicks = raw_df[
        (raw_df["event_type"] == "click") &
        (raw_df["ad_id"] == TARGET_AD_ID)
    ]

    result = clicks.groupby("traffic_type").size().reset_index(name="click_count")
    result["budget_spent"] = result["click_count"] * cpc
    return result


# =========================
# 8. Export utilities  [LATEST UPDATE]
# =========================

def ensure_output_dir(output_dir: Union[Path, str]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export_attack_artifacts(
    raw_df: pd.DataFrame,
    processed_df: Optional[pd.DataFrame] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    budget_df: Optional[pd.DataFrame] = None,
    output_dir: Union[Path, str] = DEFAULT_OUTPUT_DIR,
) -> Dict[str, str]:
    """
    [LATEST UPDATE]
    保留历史代码里的“生成数据文件能力”，但导出现在是可选的，不作为接口依赖。
    """
    output_dir = ensure_output_dir(output_dir)
    saved: Dict[str, str] = {}

    raw_path = output_dir / DEFAULT_RAW_CSV
    raw_df.to_csv(raw_path, index=False)
    saved["raw_csv"] = str(raw_path)

    if processed_df is not None:
        processed_path = output_dir / DEFAULT_PROCESSED_CSV
        processed_df.to_csv(processed_path, index=False)
        saved["processed_csv"] = str(processed_path)

    if comparison_df is not None:
        comparison_path = output_dir / DEFAULT_COMPARISON_CSV
        comparison_df.to_csv(comparison_path, index=False)
        saved["comparison_csv"] = str(comparison_path)

    if budget_df is not None:
        budget_path = output_dir / DEFAULT_BUDGET_CSV
        budget_df.to_csv(budget_path, index=False)
        saved["budget_csv"] = str(budget_path)

    return saved


# =========================
# 9. Plotting
# =========================

def plot_click_frequency(raw_df: pd.DataFrame, save_path: Optional[Union[Path, str]] = None, show: bool = True) -> None:
    clicks = raw_df[raw_df["event_type"] == "click"].copy()
    clicks["minute"] = clicks["timestamp"].dt.floor("min")

    grouped = clicks.groupby(["minute", "traffic_type"]).size().reset_index(name="click_count")

    plt.figure(figsize=(10, 4))
    for t in grouped["traffic_type"].unique():
        sub = grouped[grouped["traffic_type"] == t]
        plt.plot(sub["minute"], sub["click_count"], marker="o", label=t)

    plt.title("Click Frequency Over Time")
    plt.xlabel("Time")
    plt.ylabel("Clicks per Minute")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_interval_distribution(raw_df: pd.DataFrame, save_path: Optional[Union[Path, str]] = None, show: bool = True) -> None:
    clicks = raw_df[raw_df["event_type"] == "click"].copy().sort_values(["device_id", "timestamp"])
    clicks["interval_sec"] = clicks.groupby("device_id")["timestamp"].diff().dt.total_seconds()

    plt.figure(figsize=(8, 4))
    for label in ["normal", "competitive_fraud", "bot_fraud"]:
        sub = clicks[clicks["traffic_type"] == label]["interval_sec"].dropna()
        if not sub.empty:
            plt.hist(sub, bins=20, alpha=0.5, label=label)

    plt.title("Click Interval Distribution")
    plt.xlabel("Interval (seconds)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_budget_depletion(budget_df: pd.DataFrame, save_path: Optional[Union[Path, str]] = None, show: bool = True) -> None:
    if budget_df.empty:
        print("No target ad clicks found.")
        return

    plt.figure(figsize=(10, 4))
    for t in budget_df["traffic_type"].unique():
        sub = budget_df[budget_df["traffic_type"] == t]
        plt.plot(sub["timestamp"], sub["budget_left"], marker="o", label=t)

    plt.title("Budget Depletion Over Time")
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


def plot_budget_bar(raw_df: pd.DataFrame, save_path: Optional[Union[Path, str]] = None, show: bool = True) -> None:
    df = compute_budget_spent(raw_df)

    plt.figure(figsize=(6, 4))
    plt.bar(df["traffic_type"], df["budget_spent"])
    plt.title("Budget Consumption by Traffic Type")
    plt.xlabel("Traffic Type")
    plt.ylabel("Budget Spent")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def save_attack_plots(
    raw_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    output_dir: Union[Path, str] = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> Dict[str, str]:
    """
    [LATEST UPDATE]
    保留历史代码里的“生成图能力”，并支持 main.py 统一导出。
    """
    output_dir = ensure_output_dir(output_dir)
    saved = {}

    path1 = output_dir / "attack_click_frequency.png"
    plot_click_frequency(raw_df, save_path=path1, show=show)
    saved["click_frequency_plot"] = str(path1)

    path2 = output_dir / "attack_interval_distribution.png"
    plot_interval_distribution(raw_df, save_path=path2, show=show)
    saved["interval_distribution_plot"] = str(path2)

    path3 = output_dir / "attack_budget_depletion.png"
    plot_budget_depletion(budget_df, save_path=path3, show=show)
    saved["budget_depletion_plot"] = str(path3)

    path4 = output_dir / "attack_budget_bar.png"
    plot_budget_bar(raw_df, save_path=path4, show=show)
    saved["budget_bar_plot"] = str(path4)

    return saved


# =========================
# 10. Convenience runner  [LATEST UPDATE]
# =========================

def run_attack_pipeline(
    export_artifacts: bool = False,
    output_dir: Union[Path, str] = DEFAULT_OUTPUT_DIR,
    save_plots: bool = False,
    show_plots: bool = False,
) -> Dict[str, object]:
    raw_df = build_raw_dataframe()
    processed_df = compute_processed_features(raw_df)
    comparison_df = build_comparison_table(raw_df, processed_df)
    budget_df = simulate_budget(raw_df)

    artifacts: Dict[str, str] = {}
    if export_artifacts:
        artifacts.update(
            export_attack_artifacts(
                raw_df=raw_df,
                processed_df=processed_df,
                comparison_df=comparison_df,
                budget_df=budget_df,
                output_dir=output_dir,
            )
        )
    if save_plots:
        artifacts.update(
            save_attack_plots(
                raw_df=raw_df,
                budget_df=budget_df,
                output_dir=output_dir,
                show=show_plots,
            )
        )

    return {
        "raw_df": raw_df,
        "processed_df": processed_df,
        "comparison_df": comparison_df,
        "budget_df": budget_df,
        "artifacts": artifacts,
    }


# =========================
# 11. Main
# =========================

# =========================
# Debug / Standalone test
# =========================
if __name__ == "__main__":
    print("Running attack module standalone (debug mode)...")

    raw_df = build_raw_dataframe()
    processed_df = compute_processed_features(raw_df)

    plot_click_frequency(raw_df)
    plot_interval_distribution(raw_df)
    plot_budget_bar(raw_df)
