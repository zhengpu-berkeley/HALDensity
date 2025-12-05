"""
Compare 5-fold CV vs Leave-One-Out CV for norm constraint selection.

This script investigates how 5-fold CV and LOOCV compare in selecting
the optimal norm_constraint for a smaller sample size (n=50).

Usage:
    uv run python compare_cv_loocv.py --n_seeds 50 --n_workers 10
    uv run python compare_cv_loocv.py --n_seeds 20 --n_workers 4 --output_prefix test_cv_comparison
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, LeaveOneOut
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import warnings

from haldensity.utils import TruncatedGMM
from haldensity.censoring import (
    KaplanMeier,
    compute_ipcw_weights,
    RightCensoredIPCWEstimator,
    incomplete_loglik,
    kl_divergence,
)

# Suppress warnings in worker processes
warnings.filterwarnings("ignore")


def generate_censored_data(seed: int, n_samples: int = 50) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate right-censored data from truncated GMM."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    sampler = TruncatedGMM(
        components=[
            {"mean": 0.2, "std": 0.05, "lower": 0, "upper": 1},
            {"mean": 0.5, "std": 0.05, "lower": 0, "upper": 1},
            {"mean": 0.8, "std": 0.05, "lower": 0, "upper": 1}
        ],
        weights=[0.33, 0.34, 0.33]
    )
    
    X_true = sampler.generate_samples(n_samples)
    C = rng.uniform(0.0, 1.0, size=n_samples)
    
    T = np.minimum(X_true, C)
    Delta = (X_true <= C).astype(int)
    
    data = pd.DataFrame({"T": T, "Delta": Delta})
    return data, X_true


def compute_cv_score_kfold(
    data: pd.DataFrame,
    norm_constraint: float,
    basis_order: int = 0,
    cv_folds: int = 5,
    random_state: int = 42,
    n_grid_points: int = 200,
) -> tuple[float, list[float]]:
    """
    Compute K-fold CV score for a given norm_constraint.
    
    Returns the mean CV log-likelihood and per-fold scores.
    """
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = []
    
    for train_idx, val_idx in kfold.split(data):
        train_df = data.iloc[train_idx].reset_index(drop=True)
        val_df = data.iloc[val_idx].reset_index(drop=True)
        
        try:
            # Fit KM on train
            km = KaplanMeier().fit(train_df, time_col="T", delta_col="Delta")
            
            # Compute IPCW weights
            T_vals = np.asarray(train_df["T"].values, dtype=float)
            Delta_vals = np.asarray(train_df["Delta"].values, dtype=int)
            w = compute_ipcw_weights(
                T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t))
            )
            
            # Fit on uncensored only
            unc_mask = Delta_vals == 1
            df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
            w_unc = w[unc_mask]
            
            # Skip if too few uncensored observations
            if len(df_unc) < 5:
                scores.append(float("-inf"))
                continue
            
            est = RightCensoredIPCWEstimator(
                tol=1e-6,
                norm_constraint=norm_constraint,
                n_grid_points=n_grid_points,
                basis_order=basis_order,
                solver="ECOS",
                use_secondary_solver=True,
            ).fit(df_unc, sample_weights=w_unc)
            
            # Evaluate on validation
            score = incomplete_loglik(est, val_df, time_col="T", delta_col="Delta")
            scores.append(score)
        except Exception as e:
            scores.append(float("-inf"))
    
    mean_score = float(np.mean(scores))
    return mean_score, scores


def compute_cv_score_loocv(
    data: pd.DataFrame,
    norm_constraint: float,
    basis_order: int = 0,
    n_grid_points: int = 200,
) -> tuple[float, list[float]]:
    """
    Compute Leave-One-Out CV score for a given norm_constraint.
    
    Returns the mean CV log-likelihood and per-sample scores.
    """
    loo = LeaveOneOut()
    scores = []
    
    for train_idx, val_idx in loo.split(data):
        train_df = data.iloc[train_idx].reset_index(drop=True)
        val_df = data.iloc[val_idx].reset_index(drop=True)
        
        try:
            # Fit KM on train
            km = KaplanMeier().fit(train_df, time_col="T", delta_col="Delta")
            
            # Compute IPCW weights
            T_vals = np.asarray(train_df["T"].values, dtype=float)
            Delta_vals = np.asarray(train_df["Delta"].values, dtype=int)
            w = compute_ipcw_weights(
                T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t))
            )
            
            # Fit on uncensored only
            unc_mask = Delta_vals == 1
            df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
            w_unc = w[unc_mask]
            
            # Skip if too few uncensored observations
            if len(df_unc) < 5:
                scores.append(float("-inf"))
                continue
            
            est = RightCensoredIPCWEstimator(
                tol=1e-6,
                norm_constraint=norm_constraint,
                n_grid_points=n_grid_points,
                basis_order=basis_order,
                solver="ECOS",
                use_secondary_solver=True,
            ).fit(df_unc, sample_weights=w_unc)
            
            # Evaluate on validation (single sample)
            score = incomplete_loglik(est, val_df, time_col="T", delta_col="Delta")
            scores.append(score)
        except Exception as e:
            scores.append(float("-inf"))
    
    # Filter out -inf scores before computing mean
    valid_scores = [s for s in scores if s > float("-inf")]
    if valid_scores:
        mean_score = float(np.mean(valid_scores))
    else:
        mean_score = float("-inf")
    return mean_score, scores


def compute_full_data_metrics(
    data: pd.DataFrame,
    norm_constraint: float,
    true_pdf_fn,
    basis_order: int = 0,
    n_grid_points: int = 200,
) -> dict:
    """
    Fit model on full data and compute metrics.
    
    Returns dict with log-likelihood and KL divergence.
    """
    # Fit KM
    km = KaplanMeier().fit(data, time_col="T", delta_col="Delta")
    
    T_vals = np.asarray(data["T"].values, dtype=float)
    Delta_vals = np.asarray(data["Delta"].values, dtype=int)
    w = compute_ipcw_weights(
        T_vals, Delta_vals, lambda t: np.atleast_1d(km.predict(t))
    )
    
    unc_mask = Delta_vals == 1
    df_unc = pd.DataFrame({"W1": T_vals[unc_mask]})
    w_unc = w[unc_mask]
    
    try:
        est = RightCensoredIPCWEstimator(
            tol=1e-6,
            norm_constraint=norm_constraint,
            n_grid_points=n_grid_points,
            basis_order=basis_order,
            solver="ECOS",
            use_secondary_solver=True,
        ).fit(df_unc, sample_weights=w_unc)
        
        eval_grid = np.linspace(0, 1, 500)
        density = est.get_density_at_points(eval_grid)
        
        ll = incomplete_loglik(est, data, time_col="T", delta_col="Delta")
        kl = kl_divergence(true_pdf_fn=true_pdf_fn, grid=eval_grid, est_density=density)
        
        return {"ll": ll, "kl": kl, "density": density, "grid": eval_grid}
    except Exception as e:
        eval_grid = np.linspace(0, 1, 500)
        return {"ll": float("nan"), "kl": float("nan"), "density": np.full_like(eval_grid, np.nan), "grid": eval_grid}


def process_single_seed(args: tuple) -> dict:
    """
    Process a single seed - designed for parallel execution.
    
    Args:
        args: Tuple of (seed, norm_constraints, basis_order, cv_random_state, n_samples)
    
    Returns:
        Dict with results for this seed
    """
    seed, norm_constraints, basis_order, cv_random_state, n_samples = args
    
    # Recreate true_pdf_fn in worker process
    sampler = TruncatedGMM(
        components=[
            {"mean": 0.2, "std": 0.05, "lower": 0, "upper": 1},
            {"mean": 0.5, "std": 0.05, "lower": 0, "upper": 1},
            {"mean": 0.8, "std": 0.05, "lower": 0, "upper": 1}
        ],
        weights=[0.33, 0.34, 0.33]
    )
    true_pdf_fn = sampler.compute_density
    
    # Generate data
    data, X_true = generate_censored_data(seed, n_samples)
    censoring_rate = (1 - data["Delta"].mean())
    
    # Storage for this seed
    cv_5fold_scores = []
    cv_loocv_scores = []
    full_ll = []
    full_kl = []
    
    # Iterate over norm constraints
    for nc in norm_constraints:
        # 5-fold CV score
        mean_cv_5fold, _ = compute_cv_score_kfold(
            data=data,
            norm_constraint=nc,
            basis_order=basis_order,
            cv_folds=5,
            random_state=cv_random_state,
        )
        cv_5fold_scores.append(mean_cv_5fold)
        
        # LOOCV score
        mean_cv_loocv, _ = compute_cv_score_loocv(
            data=data,
            norm_constraint=nc,
            basis_order=basis_order,
        )
        cv_loocv_scores.append(mean_cv_loocv)
        
        # Full data metrics
        metrics = compute_full_data_metrics(
            data=data,
            norm_constraint=nc,
            true_pdf_fn=true_pdf_fn,
            basis_order=basis_order,
        )
        full_ll.append(metrics["ll"])
        full_kl.append(metrics["kl"])
    
    return {
        "seed": seed,
        "norm_constraints": norm_constraints,
        "cv_5fold_scores": np.array(cv_5fold_scores),
        "cv_loocv_scores": np.array(cv_loocv_scores),
        "full_ll": np.array(full_ll),
        "full_kl": np.array(full_kl),
        "censoring_rate": censoring_rate,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare 5-fold CV vs LOOCV for norm constraint selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--n_seeds", type=int, default=50,
        help="Number of random seeds to test"
    )
    parser.add_argument(
        "--n_workers", type=int, default=10,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--basis_order", type=int, default=0,
        help="Basis order for HAL estimator"
    )
    parser.add_argument(
        "--nc_min", type=float, default=1.0,
        help="Minimum norm constraint"
    )
    parser.add_argument(
        "--nc_max", type=float, default=25.0,
        help="Maximum norm constraint"
    )
    parser.add_argument(
        "--nc_step", type=float, default=0.5,
        help="Norm constraint step size"
    )
    parser.add_argument(
        "--n_samples", type=int, default=50,
        help="Sample size per dataset"
    )
    parser.add_argument(
        "--output_prefix", type=str, default=None,
        help="Output file prefix. If not provided, uses timestamp"
    )
    parser.add_argument(
        "--seed_base", type=int, default=2024,
        help="Base seed for reproducible seed generation"
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Configuration
    n_seeds = args.n_seeds
    n_workers = args.n_workers
    n_samples = args.n_samples
    np.random.seed(args.seed_base)
    seeds = np.random.randint(1, 100000, size=n_seeds).tolist()
    
    norm_constraints = np.arange(args.nc_min, args.nc_max, args.nc_step)
    basis_order = args.basis_order
    cv_random_state = 42
    
    # Generate output prefix with timestamp if not provided
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"cv_loocv_comparison_{timestamp}"
    
    print("=" * 70)
    print("5-Fold CV vs LOOCV Comparison")
    print("=" * 70)
    print(f"Sample size: {n_samples}")
    print(f"Number of seeds: {n_seeds}")
    print(f"Number of workers: {n_workers}")
    print(f"Norm constraints: {norm_constraints[0]} to {norm_constraints[-1]} (step {args.nc_step})")
    print(f"Number of norm constraints: {len(norm_constraints)}")
    print(f"Basis order: {basis_order}")
    print(f"Output prefix: {output_prefix}")
    print("=" * 70)
    
    # Store results
    all_results = {}
    
    # Prepare arguments for parallel processing
    task_args = [
        (seed, norm_constraints, basis_order, cv_random_state, n_samples)
        for seed in seeds
    ]
    
    # Run in parallel
    print(f"\nProcessing {n_seeds} seeds with {n_workers} workers...")
    print("Note: LOOCV is slower than 5-fold CV due to n_samples iterations per norm constraint")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_seed, arg): arg[0] for arg in task_args}
        
        with tqdm(total=n_seeds, desc="Seeds") as pbar:
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    result = future.result()
                    all_results[result["seed"]] = result
                    
                    # Brief summary
                    cv_5fold = result["cv_5fold_scores"]
                    cv_loocv = result["cv_loocv_scores"]
                    kl_arr = result["full_kl"]
                    
                    if not np.all(np.isnan(cv_5fold)) and not np.all(np.isnan(kl_arr)):
                        best_5fold_idx = np.nanargmax(cv_5fold)
                        best_loocv_idx = np.nanargmax(cv_loocv)
                        best_kl_idx = np.nanargmin(kl_arr)
                        gap_5fold = norm_constraints[best_5fold_idx] - norm_constraints[best_kl_idx]
                        gap_loocv = norm_constraints[best_loocv_idx] - norm_constraints[best_kl_idx]
                        pbar.set_postfix({
                            "seed": seed, 
                            "gap_5f": f"{gap_5fold:.1f}",
                            "gap_loo": f"{gap_loocv:.1f}",
                        })
                except Exception as e:
                    print(f"\nError processing seed {seed}: {e}")
                
                pbar.update(1)
    
    # Reorder results by seed order
    all_results = {seed: all_results[seed] for seed in seeds if seed in all_results}
    seeds = list(all_results.keys())
    n_seeds = len(seeds)
    
    print(f"\nSuccessfully processed {n_seeds} seeds")
    
    # Compute summary statistics
    best_5fold_ncs = []
    best_loocv_ncs = []
    best_kl_ncs = []
    gaps_5fold = []
    gaps_loocv = []
    
    for seed in seeds:
        res = all_results[seed]
        cv_5fold = res["cv_5fold_scores"]
        cv_loocv = res["cv_loocv_scores"]
        kl_scores = res["full_kl"]
        
        # Best 5-fold CV
        if not np.all(np.isnan(cv_5fold)) and not np.all(cv_5fold == float("-inf")):
            valid_5fold = np.where(cv_5fold > float("-inf"))[0]
            if len(valid_5fold) > 0:
                best_5fold_idx = valid_5fold[np.nanargmax(cv_5fold[valid_5fold])]
                best_5fold_nc = res["norm_constraints"][best_5fold_idx]
            else:
                best_5fold_nc = float("nan")
        else:
            best_5fold_nc = float("nan")
        
        # Best LOOCV
        if not np.all(np.isnan(cv_loocv)) and not np.all(cv_loocv == float("-inf")):
            valid_loocv = np.where(cv_loocv > float("-inf"))[0]
            if len(valid_loocv) > 0:
                best_loocv_idx = valid_loocv[np.nanargmax(cv_loocv[valid_loocv])]
                best_loocv_nc = res["norm_constraints"][best_loocv_idx]
            else:
                best_loocv_nc = float("nan")
        else:
            best_loocv_nc = float("nan")
        
        # Best KL
        if not np.all(np.isnan(kl_scores)):
            best_kl_idx = np.nanargmin(kl_scores)
            best_kl_nc = res["norm_constraints"][best_kl_idx]
        else:
            best_kl_nc = float("nan")
        
        best_5fold_ncs.append(best_5fold_nc)
        best_loocv_ncs.append(best_loocv_nc)
        best_kl_ncs.append(best_kl_nc)
        
        if not np.isnan(best_5fold_nc) and not np.isnan(best_kl_nc):
            gaps_5fold.append(best_5fold_nc - best_kl_nc)
        else:
            gaps_5fold.append(float("nan"))
        
        if not np.isnan(best_loocv_nc) and not np.isnan(best_kl_nc):
            gaps_loocv.append(best_loocv_nc - best_kl_nc)
        else:
            gaps_loocv.append(float("nan"))
    
    best_5fold_ncs = np.array(best_5fold_ncs)
    best_loocv_ncs = np.array(best_loocv_ncs)
    best_kl_ncs = np.array(best_kl_ncs)
    gaps_5fold = np.array(gaps_5fold)
    gaps_loocv = np.array(gaps_loocv)
    
    # ==================== PLOTTING ====================
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)
    
    # Figure 1: Side-by-side comparison of CV curves for each seed
    n_cols = 5
    n_rows = (n_seeds + n_cols - 1) // n_cols
    subplot_width = 4
    subplot_height = 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * subplot_width, n_rows * subplot_height))
    if n_seeds == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, seed in enumerate(seeds):
        ax = axes[i]
        res = all_results[seed]
        
        # Plot both CV methods
        ax.plot(res["norm_constraints"], res["cv_5fold_scores"], "b-", linewidth=1.5, label="5-Fold CV")
        ax.plot(res["norm_constraints"], res["cv_loocv_scores"], "g-", linewidth=1.5, label="LOOCV")
        
        # Mark best for each method
        cv_5fold = res["cv_5fold_scores"]
        cv_loocv = res["cv_loocv_scores"]
        kl_scores = res["full_kl"]
        
        if not np.all(np.isnan(cv_5fold)) and not np.all(cv_5fold == float("-inf")):
            valid_5fold = np.where(cv_5fold > float("-inf"))[0]
            if len(valid_5fold) > 0:
                best_5fold_idx = valid_5fold[np.nanargmax(cv_5fold[valid_5fold])]
                ax.axvline(res["norm_constraints"][best_5fold_idx], color="b", linestyle="--", alpha=0.7, linewidth=1.2)
        
        if not np.all(np.isnan(cv_loocv)) and not np.all(cv_loocv == float("-inf")):
            valid_loocv = np.where(cv_loocv > float("-inf"))[0]
            if len(valid_loocv) > 0:
                best_loocv_idx = valid_loocv[np.nanargmax(cv_loocv[valid_loocv])]
                ax.axvline(res["norm_constraints"][best_loocv_idx], color="g", linestyle="--", alpha=0.7, linewidth=1.2)
        
        if not np.all(np.isnan(kl_scores)):
            best_kl_idx = np.nanargmin(kl_scores)
            ax.axvline(res["norm_constraints"][best_kl_idx], color="r", linestyle="--", alpha=0.7, linewidth=1.2)
        
        ax.set_title(f"Seed {seed}", fontsize=10, pad=2)
        ax.tick_params(axis='both', which='both', labelsize=8, pad=2)
        ax.grid(alpha=0.3, linewidth=0.5)
        
        if i == 0:
            ax.legend(fontsize=8, loc="lower right")
        
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Norm Constraint", fontsize=9)
        if i % n_cols == 0:
            ax.set_ylabel("CV LL", fontsize=9)
    
    for i in range(n_seeds, n_rows * n_cols):
        axes[i].axis("off")
    
    plt.suptitle(f"5-Fold CV (Blue) vs LOOCV (Green) | Red=Best KL\n(n={n_samples}, {n_seeds} Seeds)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_prefix}_curves.png")
    
    # Figure 2: Aggregated comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Normalized CV landscape comparison
    ax = axes[0, 0]
    for seed in seeds:
        res = all_results[seed]
        cv_5fold = res["cv_5fold_scores"]
        mean_val = np.nanmean(cv_5fold[cv_5fold > float("-inf")])
        std_val = np.nanstd(cv_5fold[cv_5fold > float("-inf")])
        if std_val > 0:
            cv_norm = (cv_5fold - mean_val) / std_val
        else:
            cv_norm = cv_5fold - mean_val
        ax.plot(res["norm_constraints"], cv_norm, "b-", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Norm Constraint")
    ax.set_ylabel("Normalized CV LL (z-score)")
    ax.set_title(f"5-Fold CV Landscape ({n_seeds} Seeds)")
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    for seed in seeds:
        res = all_results[seed]
        cv_loocv = res["cv_loocv_scores"]
        mean_val = np.nanmean(cv_loocv[cv_loocv > float("-inf")])
        std_val = np.nanstd(cv_loocv[cv_loocv > float("-inf")])
        if std_val > 0:
            cv_norm = (cv_loocv - mean_val) / std_val
        else:
            cv_norm = cv_loocv - mean_val
        ax.plot(res["norm_constraints"], cv_norm, "g-", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Norm Constraint")
    ax.set_ylabel("Normalized CV LL (z-score)")
    ax.set_title(f"LOOCV Landscape ({n_seeds} Seeds)")
    ax.grid(alpha=0.3)
    
    # KL divergence comparison
    ax = axes[0, 2]
    for seed in seeds:
        res = all_results[seed]
        ax.plot(res["norm_constraints"], res["full_kl"], "r-", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Norm Constraint")
    ax.set_ylabel("KL Divergence")
    ax.set_title(f"KL Divergence Landscape ({n_seeds} Seeds)")
    ax.grid(alpha=0.3)
    
    # Distribution of best norm constraints
    ax = axes[1, 0]
    valid_mask = ~np.isnan(best_5fold_ncs) & ~np.isnan(best_loocv_ncs) & ~np.isnan(best_kl_ncs)
    ax.hist(best_5fold_ncs[valid_mask], bins=20, alpha=0.5, label=f"5-Fold (mean={np.nanmean(best_5fold_ncs):.2f})", color="blue")
    ax.hist(best_loocv_ncs[valid_mask], bins=20, alpha=0.5, label=f"LOOCV (mean={np.nanmean(best_loocv_ncs):.2f})", color="green")
    ax.hist(best_kl_ncs[valid_mask], bins=20, alpha=0.5, label=f"Best KL (mean={np.nanmean(best_kl_ncs):.2f})", color="red")
    ax.axvline(np.nanmean(best_5fold_ncs), color="blue", linestyle="--", linewidth=2)
    ax.axvline(np.nanmean(best_loocv_ncs), color="green", linestyle="--", linewidth=2)
    ax.axvline(np.nanmean(best_kl_ncs), color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Norm Constraint")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Best Norm Constraints")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Gap distribution comparison
    ax = axes[1, 1]
    valid_gaps_5fold = gaps_5fold[~np.isnan(gaps_5fold)]
    valid_gaps_loocv = gaps_loocv[~np.isnan(gaps_loocv)]
    ax.hist(valid_gaps_5fold, bins=20, alpha=0.6, label=f"5-Fold Gap (mean={np.mean(valid_gaps_5fold):.2f})", color="blue", edgecolor="black")
    ax.hist(valid_gaps_loocv, bins=20, alpha=0.6, label=f"LOOCV Gap (mean={np.mean(valid_gaps_loocv):.2f})", color="green", edgecolor="black")
    ax.axvline(0, color="black", linestyle="-", linewidth=2)
    ax.axvline(np.mean(valid_gaps_5fold), color="blue", linestyle="--", linewidth=2)
    ax.axvline(np.mean(valid_gaps_loocv), color="green", linestyle="--", linewidth=2)
    ax.set_xlabel("Gap (Selected NC - Best KL NC)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Gaps | 5-Fold: {np.mean(valid_gaps_5fold):.2f}±{np.std(valid_gaps_5fold):.2f}, LOOCV: {np.mean(valid_gaps_loocv):.2f}±{np.std(valid_gaps_loocv):.2f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Scatter: 5-fold selection vs LOOCV selection
    ax = axes[1, 2]
    valid_both = ~np.isnan(best_5fold_ncs) & ~np.isnan(best_loocv_ncs)
    ax.scatter(best_5fold_ncs[valid_both], best_loocv_ncs[valid_both], alpha=0.6, s=30)
    min_val = min(np.nanmin(best_5fold_ncs), np.nanmin(best_loocv_ncs))
    max_val = max(np.nanmax(best_5fold_ncs), np.nanmax(best_loocv_ncs))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="y=x")
    ax.set_xlabel("5-Fold CV Selected NC")
    ax.set_ylabel("LOOCV Selected NC")
    ax.set_title("5-Fold vs LOOCV Selection")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Compute correlation
    corr_5fold_loocv = np.corrcoef(best_5fold_ncs[valid_both], best_loocv_ncs[valid_both])[0, 1]
    ax.text(0.05, 0.95, f"Corr: {corr_5fold_loocv:.3f}", transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"5-Fold CV vs LOOCV Comparison (n={n_samples}, {n_seeds} Seeds)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_aggregated.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_prefix}_aggregated.png")
    
    # Figure 3: KL performance comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # KL at selected NC for each method
    kl_at_5fold = []
    kl_at_loocv = []
    kl_at_best = []
    
    for seed in seeds:
        res = all_results[seed]
        cv_5fold = res["cv_5fold_scores"]
        cv_loocv = res["cv_loocv_scores"]
        kl_scores = res["full_kl"]
        
        # 5-fold selected KL
        if not np.all(np.isnan(cv_5fold)) and not np.all(cv_5fold == float("-inf")):
            valid_5fold = np.where(cv_5fold > float("-inf"))[0]
            if len(valid_5fold) > 0:
                best_5fold_idx = valid_5fold[np.nanargmax(cv_5fold[valid_5fold])]
                kl_at_5fold.append(kl_scores[best_5fold_idx])
            else:
                kl_at_5fold.append(float("nan"))
        else:
            kl_at_5fold.append(float("nan"))
        
        # LOOCV selected KL
        if not np.all(np.isnan(cv_loocv)) and not np.all(cv_loocv == float("-inf")):
            valid_loocv = np.where(cv_loocv > float("-inf"))[0]
            if len(valid_loocv) > 0:
                best_loocv_idx = valid_loocv[np.nanargmax(cv_loocv[valid_loocv])]
                kl_at_loocv.append(kl_scores[best_loocv_idx])
            else:
                kl_at_loocv.append(float("nan"))
        else:
            kl_at_loocv.append(float("nan"))
        
        # Best achievable KL
        if not np.all(np.isnan(kl_scores)):
            kl_at_best.append(np.nanmin(kl_scores))
        else:
            kl_at_best.append(float("nan"))
    
    kl_at_5fold = np.array(kl_at_5fold)
    kl_at_loocv = np.array(kl_at_loocv)
    kl_at_best = np.array(kl_at_best)
    
    # Scatter: KL at 5-fold vs KL at best
    ax = axes[0]
    valid_kl_mask = ~np.isnan(kl_at_5fold) & ~np.isnan(kl_at_best)
    ax.scatter(kl_at_best[valid_kl_mask], kl_at_5fold[valid_kl_mask], alpha=0.6, s=30, color="blue")
    max_kl = max(np.nanmax(kl_at_5fold), np.nanmax(kl_at_best))
    ax.plot([0, max_kl], [0, max_kl], 'r--', linewidth=2, label="y=x")
    ax.set_xlabel("Best Achievable KL")
    ax.set_ylabel("KL at 5-Fold Selected NC")
    ax.set_title(f"5-Fold CV: Mean excess KL = {np.nanmean(kl_at_5fold - kl_at_best):.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Scatter: KL at LOOCV vs KL at best
    ax = axes[1]
    valid_kl_mask = ~np.isnan(kl_at_loocv) & ~np.isnan(kl_at_best)
    ax.scatter(kl_at_best[valid_kl_mask], kl_at_loocv[valid_kl_mask], alpha=0.6, s=30, color="green")
    ax.plot([0, max_kl], [0, max_kl], 'r--', linewidth=2, label="y=x")
    ax.set_xlabel("Best Achievable KL")
    ax.set_ylabel("KL at LOOCV Selected NC")
    ax.set_title(f"LOOCV: Mean excess KL = {np.nanmean(kl_at_loocv - kl_at_best):.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Direct comparison: 5-fold vs LOOCV KL
    ax = axes[2]
    valid_both_kl = ~np.isnan(kl_at_5fold) & ~np.isnan(kl_at_loocv)
    ax.scatter(kl_at_5fold[valid_both_kl], kl_at_loocv[valid_both_kl], alpha=0.6, s=30, color="purple")
    max_kl_both = max(np.nanmax(kl_at_5fold), np.nanmax(kl_at_loocv))
    ax.plot([0, max_kl_both], [0, max_kl_both], 'r--', linewidth=2, label="y=x")
    ax.set_xlabel("KL at 5-Fold Selected NC")
    ax.set_ylabel("KL at LOOCV Selected NC")
    loocv_better = np.sum((kl_at_loocv < kl_at_5fold)[valid_both_kl])
    fold5_better = np.sum((kl_at_5fold < kl_at_loocv)[valid_both_kl])
    ax.set_title(f"5-Fold vs LOOCV | LOOCV better: {loocv_better}/{np.sum(valid_both_kl)}, 5-Fold better: {fold5_better}/{np.sum(valid_both_kl)}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle(f"KL Divergence Performance Comparison (n={n_samples}, {n_seeds} Seeds)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_kl_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_prefix}_kl_comparison.png")
    
    # Figure 4: Heatmaps comparing both methods
    fig, axes = plt.subplots(1, 3, figsize=(20, max(8, n_seeds * 0.12)))
    
    cv_5fold_matrix = np.array([all_results[seed]["cv_5fold_scores"] for seed in seeds])
    cv_loocv_matrix = np.array([all_results[seed]["cv_loocv_scores"] for seed in seeds])
    kl_matrix = np.array([all_results[seed]["full_kl"] for seed in seeds])
    
    # 5-fold CV Heatmap
    ax = axes[0]
    im1 = ax.imshow(cv_5fold_matrix, aspect="auto", cmap="viridis", 
                    extent=[norm_constraints[0], norm_constraints[-1], len(seeds)-0.5, -0.5])
    ax.set_xlabel("Norm Constraint")
    ax.set_ylabel("Seed Index")
    ax.set_title("5-Fold CV Log-Likelihood")
    plt.colorbar(im1, ax=ax, label="CV LL")
    
    # LOOCV Heatmap
    ax = axes[1]
    im2 = ax.imshow(cv_loocv_matrix, aspect="auto", cmap="viridis", 
                    extent=[norm_constraints[0], norm_constraints[-1], len(seeds)-0.5, -0.5])
    ax.set_xlabel("Norm Constraint")
    ax.set_ylabel("Seed Index")
    ax.set_title("LOOCV Log-Likelihood")
    plt.colorbar(im2, ax=ax, label="CV LL")
    
    # KL Heatmap
    ax = axes[2]
    im3 = ax.imshow(kl_matrix, aspect="auto", cmap="hot_r", 
                    extent=[norm_constraints[0], norm_constraints[-1], len(seeds)-0.5, -0.5])
    ax.set_xlabel("Norm Constraint")
    ax.set_ylabel("Seed Index")
    ax.set_title("KL Divergence")
    plt.colorbar(im3, ax=ax, label="KL")
    
    plt.suptitle(f"Heatmaps: 5-Fold CV vs LOOCV vs KL (n={n_samples}, {n_seeds} Seeds)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_prefix}_heatmaps.png")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Sample size: {n_samples}")
    print(f"Number of seeds: {n_seeds}")
    print(f"Valid seeds: {np.sum(valid_mask)}")
    
    print(f"\n--- 5-Fold CV ---")
    print(f"Best NC: Mean={np.nanmean(best_5fold_ncs):.2f}, Std={np.nanstd(best_5fold_ncs):.2f}")
    print(f"Gap (vs KL): Mean={np.nanmean(gaps_5fold):.2f}, Std={np.nanstd(gaps_5fold):.2f}")
    print(f"% positive gap (over-selects): {100 * np.nanmean(gaps_5fold > 0):.1f}%")
    print(f"Mean excess KL: {np.nanmean(kl_at_5fold - kl_at_best):.4f}")
    
    print(f"\n--- LOOCV ---")
    print(f"Best NC: Mean={np.nanmean(best_loocv_ncs):.2f}, Std={np.nanstd(best_loocv_ncs):.2f}")
    print(f"Gap (vs KL): Mean={np.nanmean(gaps_loocv):.2f}, Std={np.nanstd(gaps_loocv):.2f}")
    print(f"% positive gap (over-selects): {100 * np.nanmean(gaps_loocv > 0):.1f}%")
    print(f"Mean excess KL: {np.nanmean(kl_at_loocv - kl_at_best):.4f}")
    
    print(f"\n--- Best KL (Oracle) ---")
    print(f"Best NC: Mean={np.nanmean(best_kl_ncs):.2f}, Std={np.nanstd(best_kl_ncs):.2f}")
    
    print(f"\n--- Comparison ---")
    print(f"Correlation (5-Fold vs LOOCV): {corr_5fold_loocv:.3f}")
    print(f"LOOCV better than 5-Fold: {loocv_better}/{np.sum(valid_both_kl)} ({100*loocv_better/np.sum(valid_both_kl):.1f}%)")
    print(f"5-Fold better than LOOCV: {fold5_better}/{np.sum(valid_both_kl)} ({100*fold5_better/np.sum(valid_both_kl):.1f}%)")
    
    # Correlation with true optimal
    valid_5fold_kl = ~np.isnan(best_5fold_ncs) & ~np.isnan(best_kl_ncs)
    valid_loocv_kl = ~np.isnan(best_loocv_ncs) & ~np.isnan(best_kl_ncs)
    corr_5fold_kl = np.corrcoef(best_5fold_ncs[valid_5fold_kl], best_kl_ncs[valid_5fold_kl])[0, 1]
    corr_loocv_kl = np.corrcoef(best_loocv_ncs[valid_loocv_kl], best_kl_ncs[valid_loocv_kl])[0, 1]
    print(f"\nCorrelation with true optimal (Best KL NC):")
    print(f"  5-Fold: {corr_5fold_kl:.3f}")
    print(f"  LOOCV:  {corr_loocv_kl:.3f}")
    
    print("\n" + "=" * 70)
    print("Done! All plots saved.")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    results = main()

