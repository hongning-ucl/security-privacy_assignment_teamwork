from pathlib import Path

from attack import (
    run_attack_pipeline,
    save_attack_plots,
)
from defense import (
    export_defense_artifacts,
    run_defense_pipeline_detailed,
    save_defense_plots,
    summarize_budget_outcome,
    summarize_by_traffic_type,
    summarize_defense_actions,
)


OUTPUT_DIR = Path("outputs_full_pipeline")


# =====================
# Main pipeline runner  [LATEST UPDATE]
# =====================

def run_main_pipeline(
    export_artifacts: bool = True,
    save_plots: bool = True,
    show_plots: bool = False,
    output_dir: Path | str = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Attack simulation
    attack_result = run_attack_pipeline(
        export_artifacts=export_artifacts,
        output_dir=output_dir,
        save_plots=False,
        show_plots=show_plots,
    )
    raw_df = attack_result["raw_df"]

    # 2. Defense pipeline
    defense_result = run_defense_pipeline_detailed(raw_df)
    defense_df = defense_result["defense_df"]

    # 3. Summaries
    defense_action_summary = summarize_defense_actions(defense_df)
    by_traffic_summary = summarize_by_traffic_type(defense_df)
    budget_outcome_summary = summarize_budget_outcome(defense_df)

    artifacts = {}

    # 4. Optional artifact export
    if export_artifacts:
        artifacts.update(attack_result["artifacts"])
        artifacts.update(export_defense_artifacts(defense_result, output_dir=output_dir))

        defense_action_summary.to_csv(output_dir / "summary_defense_actions.csv", index=False)
        by_traffic_summary.to_csv(output_dir / "summary_by_traffic_type.csv", index=False)
        budget_outcome_summary.to_csv(output_dir / "summary_budget_outcome.csv", index=False)

        artifacts["summary_defense_actions"] = str(output_dir / "summary_defense_actions.csv")
        artifacts["summary_by_traffic_type"] = str(output_dir / "summary_by_traffic_type.csv")
        artifacts["summary_budget_outcome"] = str(output_dir / "summary_budget_outcome.csv")

    # 5. Optional plots
    if save_plots:
        artifacts.update(save_attack_plots(raw_df, attack_result["budget_df"], output_dir=output_dir, show=show_plots))
        artifacts.update(save_defense_plots(raw_df, defense_result, output_dir=output_dir, show=show_plots))

    return {
        "raw_df": raw_df,
        "processed_df": attack_result["processed_df"],
        "comparison_df": attack_result["comparison_df"],
        "budget_df": attack_result["budget_df"],
        "layer1_full_df": defense_result["layer1_full_df"],
        "layer1_passed_df": defense_result["layer1_passed_df"],
        "layer2_ip_df": defense_result["layer2_ip_df"],
        "defense_df": defense_df,
        "layer1_metrics": defense_result["layer1_metrics"],
        "layer2_metrics": defense_result["layer2_metrics"],
        "defense_action_summary": defense_action_summary,
        "by_traffic_summary": by_traffic_summary,
        "budget_outcome_summary": budget_outcome_summary,
        "artifacts": artifacts,
    }


if __name__ == "__main__":
    result = run_main_pipeline(export_artifacts=True, save_plots=True, show_plots=False)

    print("\n=== RAW DATA SAMPLE ===")
    print(result["raw_df"].head())

    print("\n=== LAYER 1 METRICS ===")
    print(result["layer1_metrics"])

    print("\n=== LAYER 2 METRICS ===")
    print(result["layer2_metrics"])

    print("\n=== DEFENSE ACTION SUMMARY ===")
    print(result["defense_action_summary"])

    print("\n=== BY TRAFFIC TYPE ===")
    print(result["by_traffic_summary"])

    print("\n=== BUDGET OUTCOME ===")
    print(result["budget_outcome_summary"])

    print("\n=== ARTIFACTS ===")
    for k, v in result["artifacts"].items():
        print(f"{k}: {v}")
