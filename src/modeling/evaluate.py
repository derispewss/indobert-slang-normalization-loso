import os
os.environ["TORCH_LOAD_RESTRICTED"] = "0"
# ============================================================
# src/modeling/evaluate.py
# Skrip Evaluasi Cross-Platform (LOSO) dan In-Domain
# Mengukur performa generalisasi IndoBERT dengan/tanpa Normalisasi Slang
# Output: tabel log + 5 visualisasi PNG di folder results/
# ============================================================

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import ID2LABEL, INDOBERT_BATCH_SIZE, INDOBERT_MAX_LENGTH, LABEL2ID, LOG_DIR
from src.modeling.dataset import ReviewDataset
from src.scraping.scraper_utils import setup_logger

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

# ── Konstanta Visual ──────────────────────────────────────────
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = ["Negatif", "Netral", "Positif"]
COLOR_BASE   = "#4C72B0"   # biru (Baseline)
COLOR_PROP   = "#DD8452"   # oranye (Proposed)
COLOR_INDOMAIN   = "#2ecc71"
COLOR_CROSS      = "#e74c3c"

STYLE = {
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
}
plt.rcParams.update(STYLE)


# ── Helpers Data ──────────────────────────────────────────────

def get_test_path(platform: str, condition: str) -> Path:
    return ROOT / "data" / "processed" / "experiment_splits" / f"test_{platform}_{condition}.csv"

def get_model_path(platform: str, condition: str) -> Path:
    return ROOT / "model" / f"indobert_{platform}_{condition}"

def load_test_data(csv_path: Path) -> tuple[list[str], list[int]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    texts  = df["review_text_clean"].astype(str).tolist()
    labels = df["sentiment_label"].map(LABEL2ID).tolist()
    return texts, labels

def evaluate_model_on_dataset(model_path: Path, test_texts: list, test_labels: list) -> dict | None:
    if not model_path.exists():
        logger.warning(f"Model tidak ditemukan: {model_path}")
        return None

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path / "tokenizer"))
    model     = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    dataset = ReviewDataset(test_texts, test_labels, tokenizer, INDOBERT_MAX_LENGTH)
    loader  = DataLoader(dataset, batch_size=INDOBERT_BATCH_SIZE, shuffle=False, num_workers=2)

    all_preds = []
    start = time.time()

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

    elapsed_ms = (time.time() - start) / len(test_texts) * 1000

    f1m  = f1_score(test_labels, all_preds, average="macro",  zero_division=0)
    f1_per_class = f1_score(test_labels, all_preds, average=None, zero_division=0)
    acc  = accuracy_score(test_labels, all_preds)
    cm   = confusion_matrix(test_labels, all_preds, labels=[0, 1, 2])
    report = classification_report(
        test_labels, all_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "accuracy":      acc,
        "f1_macro":      f1m,
        "f1_per_class":  f1_per_class.tolist(),
        "confusion_matrix": cm,
        "report":        report,
        "inference_ms":  elapsed_ms,
        "all_preds":     all_preds,
        "all_labels":    test_labels,
    }


# ── Logging Tabel ─────────────────────────────────────────────

def print_result_table(results: list, title: str):
    logger.info(f"\n{'='*100}")
    logger.info(f"TABEL EVALUASI: {title}")
    logger.info(f"{'='*100}")
    header = (
        f"{'Model (Dilatih di)':<20} | {'Diuji di':<12} | {'Kondisi':<10} | "
        f"{'F1-Macro':<9} | {'Accuracy':<9} | "
        f"{'F1-Negatif':<11} | {'F1-Netral':<10} | {'F1-Positif':<10} | {'ms/sample'}"
    )
    logger.info(header)
    logger.info("-" * 100)
    for r in results:
        f1c = r.get("f1_per_class", [0, 0, 0])
        row = (
            f"{r['train_src']:<20} | {r['test_src']:<12} | {r['condition']:<10} | "
            f"{r['f1_macro']:.4f}    | {r['accuracy']:.4f}    | "
            f"{f1c[0]:.4f}      | {f1c[1]:.4f}     | {f1c[2]:.4f}     | {r['inference_ms']:.2f}"
        )
        logger.info(row)
    logger.info("=" * 100)


# ══════════════════════════════════════════════════════════════
# VISUALISASI 1 — Confusion Matrix (4 panel, per kondisi)
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrices(all_results: list):
    """4 confusion matrix: Toped-Base, Toped-Prop, Shopee-Base, Shopee-Prop (In-Domain)."""
    in_domain = [r for r in all_results if r["train_src"].lower() == r["test_src"].lower()]
    if not in_domain:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        "Confusion Matrix — In-Domain Evaluation\n(Model diuji pada platform yang sama)",
        fontsize=14, fontweight="bold", y=0.98
    )

    for ax, r in zip(axes.flat, in_domain):
        cm_norm = r["confusion_matrix"].astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm_norm, row_sums, where=row_sums != 0) * 100

        sns.heatmap(
            cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5, cbar_kws={"label": "% per baris"},
            vmin=0, vmax=100,
        )
        # Tambahkan angka absolut di bawah persentase
        for i in range(3):
            for j in range(3):
                ax.text(
                    j + 0.5, i + 0.72,
                    f"(n={r['confusion_matrix'][i, j]})",
                    ha="center", va="center", fontsize=7.5, color="dimgray"
                )

        title = f"{r['train_src']} — {r['condition']}\nF1-Macro: {r['f1_macro']:.4f} | Acc: {r['accuracy']:.4f}"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Prediksi", fontsize=9)
        ax.set_ylabel("Aktual", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS_DIR / "01_confusion_matrix_indomain.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[VIZ] Gambar disimpan: {out}")


# ══════════════════════════════════════════════════════════════
# VISUALISASI 2 — F1-Macro Bar Chart: Baseline vs Proposed
# ══════════════════════════════════════════════════════════════

def plot_f1_comparison(all_results: list):
    """Grouped bar chart F1-Macro Baseline vs Proposed untuk semua 8 skenario."""
    scenarios = [
        ("Tokopedia", "Tokopedia", "In-Domain"),
        ("Shopee",    "Shopee",    "In-Domain"),
        ("Tokopedia", "Shopee",    "Cross-Platform"),
        ("Shopee",    "Tokopedia", "Cross-Platform"),
    ]

    labels, f1_base, f1_prop, types = [], [], [], []
    for train, test, stype in scenarios:
        base = next((r for r in all_results
                     if r["train_src"] == train and r["test_src"] == test and r["condition"] == "Baseline"), None)
        prop = next((r for r in all_results
                     if r["train_src"] == train and r["test_src"] == test and r["condition"] == "Proposed"), None)
        if base and prop:
            labels.append(f"{train[:4]}→{test[:4]}\n({stype})")
            f1_base.append(base["f1_macro"])
            f1_prop.append(prop["f1_macro"])
            types.append(stype)

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_b = ax.bar(x - width/2, f1_base, width, label="Baseline (Tanpa Normalisasi)",
                    color=COLOR_BASE, alpha=0.85, edgecolor="white", linewidth=0.8)
    bars_p = ax.bar(x + width/2, f1_prop, width, label="Proposed (Dengan Normalisasi)",
                    color=COLOR_PROP, alpha=0.85, edgecolor="white", linewidth=0.8)

    # Annotasi nilai di atas bar
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8.5, color=COLOR_BASE, fontweight="bold")
    for bar in bars_p:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8.5, color=COLOR_PROP, fontweight="bold")

    # Annotasi delta
    for i, (b, p) in enumerate(zip(f1_base, f1_prop)):
        delta = p - b
        sign  = "+" if delta >= 0 else ""
        color = "#27ae60" if delta >= 0 else "#c0392b"
        ax.text(i, max(b, p) + 0.018, f"Δ{sign}{delta:.4f}",
                ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

    # Separator garis antar kelompok
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(0.95, 0.97, "In-Domain", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="gray", style="italic")
    ax.text(0.97, 0.97, "Cross-Platform", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="gray", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("F1-Macro", fontsize=11)
    ax.set_ylim(0.45, 0.78)
    ax.set_title("Perbandingan F1-Macro: Baseline vs Proposed\n(Semua Skenario Evaluasi LOSO)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    plt.tight_layout()
    out = RESULTS_DIR / "02_f1_comparison_baseline_vs_proposed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[VIZ] Gambar disimpan: {out}")


# ══════════════════════════════════════════════════════════════
# VISUALISASI 3 — Domain Shift Heatmap (F1-Macro Matrix)
# ══════════════════════════════════════════════════════════════

def plot_domain_shift_heatmap(all_results: list):
    """Heatmap 2×2 menampilkan F1-Macro untuk setiap kombinasi Train→Test per kondisi."""
    platforms = ["Tokopedia", "Shopee"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Domain Shift Heatmap — F1-Macro (Train Platform → Test Platform)",
                 fontsize=13, fontweight="bold")

    for ax, condition in zip(axes, ["Baseline", "Proposed"]):
        matrix = np.zeros((2, 2))
        annot  = np.empty((2, 2), dtype=object)

        for i, train_p in enumerate(platforms):
            for j, test_p in enumerate(platforms):
                r = next((x for x in all_results
                          if x["train_src"] == train_p
                          and x["test_src"]  == test_p
                          and x["condition"] == condition), None)
                if r:
                    matrix[i, j] = r["f1_macro"]
                    tag = "In-Domain" if train_p == test_p else "Cross-Platform"
                    annot[i, j]  = f"{r['f1_macro']:.4f}\n({tag})"
                else:
                    annot[i, j] = "N/A"

        mask_diag  = np.eye(2, dtype=bool)
        mask_cross = ~mask_diag

        # Plot diagonal (In-Domain) dengan warna berbeda
        sns.heatmap(
            np.where(mask_diag,  matrix, np.nan),
            ax=ax, cmap="Greens", vmin=0.55, vmax=0.75,
            annot=np.where(mask_diag, annot, ""),
            fmt="", linewidths=1.5, linecolor="white",
            cbar=False, xticklabels=platforms, yticklabels=platforms,
        )
        sns.heatmap(
            np.where(mask_cross, matrix, np.nan),
            ax=ax, cmap="Reds_r",  vmin=0.50, vmax=0.70,
            annot=np.where(mask_cross, annot, ""),
            fmt="", linewidths=1.5, linecolor="white",
            cbar=False, xticklabels=platforms, yticklabels=platforms,
        )

        ax.set_title(f"Kondisi: {condition}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Diuji di (Test Platform)", fontsize=10)
        ax.set_ylabel("Dilatih di (Train Platform)", fontsize=10)
        ax.tick_params(labelsize=10)

        # Legend manual
        patches = [
            mpatches.Patch(color="#2ecc71", alpha=0.7, label="In-Domain (diagonal)"),
            mpatches.Patch(color="#e74c3c", alpha=0.7, label="Cross-Platform"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    out = RESULTS_DIR / "03_domain_shift_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[VIZ] Gambar disimpan: {out}")


# ══════════════════════════════════════════════════════════════
# VISUALISASI 4 — Per-Class F1 Breakdown (In-Domain)
# ══════════════════════════════════════════════════════════════

def plot_per_class_f1(all_results: list):
    """Grouped bar chart F1 per kelas (Negatif/Netral/Positif) untuk 4 model In-Domain."""
    in_domain = [r for r in all_results if r["train_src"].lower() == r["test_src"].lower()]
    if not in_domain:
        return

    colors_class = ["#e74c3c", "#f39c12", "#2ecc71"]
    x      = np.arange(len(in_domain))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    for ci, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors_class)):
        f1_vals = [r["f1_per_class"][ci] for r in in_domain]
        bars = ax.bar(
            x + (ci - 1) * width, f1_vals, width,
            label=cls_name, color=color, alpha=0.82,
            edgecolor="white", linewidth=0.8
        )
        for bar, val in zip(bars, f1_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=color
            )

    xlabels = [f"{r['train_src']}\n{r['condition']}" for r in in_domain]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("F1-Score per Kelas", fontsize=11)
    ax.set_ylim(0, 0.90)
    ax.set_title("F1-Score per Kelas Sentimen — In-Domain Evaluation\n(Negatif · Netral · Positif)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    plt.tight_layout()
    out = RESULTS_DIR / "04_per_class_f1_indomain.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[VIZ] Gambar disimpan: {out}")


# ══════════════════════════════════════════════════════════════
# VISUALISASI 5 — Radar Chart: Profil Model (per skenario)
# ══════════════════════════════════════════════════════════════

def plot_radar_model_profile(all_results: list):
    """Radar chart membandingkan profil metrik 4 model In-Domain secara holistik."""
    in_domain = [r for r in all_results if r["train_src"].lower() == r["test_src"].lower()]
    if len(in_domain) < 2:
        return

    metrics_labels = ["F1-Macro", "Accuracy", "F1-Negatif", "F1-Netral", "F1-Positif"]
    num_metrics    = len(metrics_labels)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # tutup lingkaran

    colors = [COLOR_BASE, COLOR_PROP, "#9b59b6", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#f8f9fa")

    for idx, (r, color) in enumerate(zip(in_domain, colors)):
        values = [
            r["f1_macro"],
            r["accuracy"],
            r["f1_per_class"][0],
            r["f1_per_class"][1],
            r["f1_per_class"][2],
        ]
        values += values[:1]
        label = f"{r['train_src']} — {r['condition']}"
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title("Profil Metrik Model — In-Domain Evaluation\n(Radar Chart)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / "05_radar_model_profile.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[VIZ] Gambar disimpan: {out}")


# ══════════════════════════════════════════════════════════════
# VISUALISASI 6 — Accuracy vs F1-Macro Scatter
# ══════════════════════════════════════════════════════════════

def plot_accuracy_vs_f1(all_results: list):
    """Scatter plot Accuracy vs F1-Macro untuk seluruh 8 skenario + label annotasi."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for r in all_results:
        is_indomain = r["train_src"].lower() == r["test_src"].lower()
        is_proposed = r["condition"] == "Proposed"

        marker = "o" if is_indomain else "^"
        color  = COLOR_PROP if is_proposed else COLOR_BASE
        ax.scatter(r["accuracy"], r["f1_macro"],
                   c=color, marker=marker, s=120, alpha=0.85,
                   edgecolors="white", linewidth=0.8, zorder=3)

        label_text = f"{r['train_src'][:4]}→{r['test_src'][:4]}\n{r['condition'][:4]}"
        ax.annotate(
            label_text,
            xy=(r["accuracy"], r["f1_macro"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=7.5, color="dimgray",
        )

    # Garis acuan (diagonal sempurna)
    lims = [0.50, 0.75]
    ax.plot(lims, lims, "--", color="gray", alpha=0.4, linewidth=1, label="Diagonal (ideal)")

    # Legend manual
    legend_handles = [
        mpatches.Patch(color=COLOR_BASE, alpha=0.85, label="Baseline"),
        mpatches.Patch(color=COLOR_PROP, alpha=0.85, label="Proposed"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=9, label="In-Domain"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                   markersize=9, label="Cross-Platform"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_ylabel("F1-Macro", fontsize=11)
    ax.set_title("Accuracy vs F1-Macro — Semua Skenario Evaluasi\n(Bulat = In-Domain · Segitiga = Cross-Platform)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0.57, 0.74)
    ax.set_ylim(0.52, 0.71)

    plt.tight_layout()
    out = RESULTS_DIR / "06_accuracy_vs_f1_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[VIZ] Gambar disimpan: {out}")


# ══════════════════════════════════════════════════════════════
# MAIN — Orchestrator
# ══════════════════════════════════════════════════════════════

def evaluate():
    setup_logger(LOG_DIR, "evaluate_cross_platform")
    logger.info("=" * 60)
    logger.info("STEP 7: CROSS-PLATFORM & IN-DOMAIN EVALUATION (INDOBERT)")
    logger.info("=" * 60)

    platforms  = ["tokopedia", "shopee"]
    conditions = ["baseline", "proposed"]
    all_results = []

    for condition in conditions:
        logger.info(f"\n{'#'*40} KONDISI: {condition.upper()} {'#'*40}")

        for train_plat in platforms:
            model_path = get_model_path(train_plat, condition)
            if not model_path.exists():
                logger.warning(f"Lewati: Model {train_plat}-{condition} belum ada.")
                continue

            for test_plat in platforms:
                test_path = get_test_path(test_plat, condition)
                if not test_path.exists():
                    logger.warning(f"Lewati: Test set {test_plat}-{condition} tidak ditemukan.")
                    continue

                logger.info(f"Menguji [{train_plat.upper()}] → [{test_plat.upper()}] ...")
                texts, labels = load_test_data(test_path)
                metrics = evaluate_model_on_dataset(model_path, texts, labels)

                if metrics:
                    all_results.append({
                        "train_src":      train_plat.title(),
                        "test_src":       test_plat.title(),
                        "condition":      condition.title(),
                        "f1_macro":       metrics["f1_macro"],
                        "accuracy":       metrics["accuracy"],
                        "f1_per_class":   metrics["f1_per_class"],
                        "confusion_matrix": metrics["confusion_matrix"],
                        "inference_ms":   metrics["inference_ms"],
                        "report":         metrics["report"],
                    })
                    logger.info(
                        f"  F1-Macro={metrics['f1_macro']:.4f} | "
                        f"Acc={metrics['accuracy']:.4f} | "
                        f"ms/sample={metrics['inference_ms']:.2f}"
                    )

    if not all_results:
        logger.error("Tidak ada hasil evaluasi. Pastikan model dan test set tersedia.")
        return

    in_domain_results  = [r for r in all_results if r["train_src"].lower() == r["test_src"].lower()]
    cross_domain_results = [r for r in all_results if r["train_src"].lower() != r["test_src"].lower()]

    # ── Tabel log ─────────────────────────────────────────────
    if in_domain_results:
        print_result_table(in_domain_results,  "IN-DOMAIN EVALUATION (Kontrol / Batas Atas)")
    if cross_domain_results:
        print_result_table(cross_domain_results, "CROSS-PLATFORM (LEAVE-ONE-SITE-OUT) EVALUATION")

    # ── Analisis delta normalisasi ─────────────────────────────
    if len(cross_domain_results) >= 4:
        logger.info("\n--- ANALISIS DELTA NORMALISASI SLANG (CROSS-PLATFORM) ---")
        for train_p in ["Tokopedia", "Shopee"]:
            base = next((r for r in cross_domain_results if r["train_src"] == train_p and r["condition"] == "Baseline"), None)
            prop = next((r for r in cross_domain_results if r["train_src"] == train_p and r["condition"] == "Proposed"), None)
            if base and prop:
                delta = prop["f1_macro"] - base["f1_macro"]
                sign  = "MENINGKAT" if delta >= 0 else "MENURUN (ANOMALI)"
                logger.info(
                    f"  {train_p} → {base['test_src']}: "
                    f"Baseline={base['f1_macro']:.4f} | Proposed={prop['f1_macro']:.4f} | "
                    f"Delta={delta:+.4f} [{sign}]"
                )

    # ── Classification Report per model ───────────────────────
    logger.info("\n--- CLASSIFICATION REPORT (IN-DOMAIN) ---")
    for r in in_domain_results:
        logger.info(f"\n>> {r['train_src']} — {r['condition']}")
        report_df = pd.DataFrame(r["report"]).T
        logger.info(f"\n{report_df.round(4).to_string()}")

    # ── Generate semua visualisasi ─────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("GENERATING VISUALISASI ...")
    logger.info(f"{'='*60}")

    plot_confusion_matrices(all_results)
    plot_f1_comparison(all_results)
    plot_domain_shift_heatmap(all_results)
    plot_per_class_f1(all_results)
    plot_radar_model_profile(all_results)
    plot_accuracy_vs_f1(all_results)

    logger.info(f"\n[DONE] Semua visualisasi tersimpan di: {RESULTS_DIR}")
    logger.info("File yang dihasilkan:")
    for f in sorted(RESULTS_DIR.glob("*.png")):
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    evaluate()
