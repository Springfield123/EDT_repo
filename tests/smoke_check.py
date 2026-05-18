from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def expected_directories_exist():
    for rel in [
        "data/raw",
        "data/processed",
        "data/figure_source_data",
        "code/analysis",
        "code/plotting",
        "code/utilities",
        "outputs/reproduced_figures",
        "outputs/reproduced_tables",
        "docs",
    ]:
        if not (ROOT / rel).exists():
            raise SystemExit(f"Missing expected directory: {rel}")


def core_source_data_files_exist():
    for rel in [
        "data/figure_source_data/Fig1_source_data.csv",
        "data/figure_source_data/Fig2_panel_b_source_data.csv",
        "data/figure_source_data/Fig5_source_data.csv",
    ]:
        if not (ROOT / rel).exists():
            raise SystemExit(f"Missing expected file: {rel}")


if __name__ == "__main__":
    expected_directories_exist()
    core_source_data_files_exist()
    print("Smoke checks passed.")
