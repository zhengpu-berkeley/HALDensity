"""
Explore CV metric landscape for different norm constraints.

This script investigates how the cross-validation log-likelihood metric
varies across a fine grid of norm_constraint values (10 to 35 with 0.25 step)
for different random seeds.

Usage:
    uv run python explore_cv_landscape.py --n_seeds 200 --n_workers 10
    uv run python explore_cv_landscape.py --n_seeds 50 --n_workers 4 --output_prefix test
    uv run python explore_cv_landscape.py --dgp sinusoidal --n_seeds 50 --n_workers 4
    uv run python explore_cv_landscape.py --dgp step --n_seeds 50
    uv run python explore_cv_landscape.py --dgp truncnorm --n_seeds 50
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import warnings

from haldensity.utils import TruncatedGMM, TruncatedNormal, Sinusoidal, StepFunction
from haldensity.censoring import (
    KaplanMeier,
    compute_ipcw_weights,
    RightCensoredIPCWEstimator,
    incomplete_loglik,
    kl_divergence,
)

# Suppress warnings in worker processes
warnings.filterwarnings("ignore")

# Available DGP choices
DGP_CHOICES = ["gmm", "truncnorm", "sinusoidal", "step"]


def create_dgp(dgp_name: str):
    """
    Create a DGP sampler based on the name.
    
    Args:
        dgp_name: One of 'gmm', 'truncnorm', 'sinusoidal', 'step'
    
    Returns:
        DGP sampler object with generate_samples() and compute_density() methods
    """
    if dgp_name == "gmm":
        return TruncatedGMM(
            components=[
                {"mean": 0.2, "std": 0.05, "lower": 0, "upper": 1},
                {"mean": 0.5, "std": 0.05, "lower": 0, "upper": 1},
                {"mean": 0.8, "std": 0.05, "lower": 0, "upper": 1}
            ],
            weights=[0.33, 0.34, 0.33]
        )
    elif dgp_name == "truncnorm":
        return TruncatedNormal(mean=0.5, std=0.15, lower=0, upper=1)
    elif dgp_name == "sinusoidal":
        return Sinusoidal()
    elif dgp_name == "step":
        return StepFunction(level1=1.5, level2=0.5, breakpoint=0.4)
    else:
        raise ValueError(f"Unknown DGP: {dgp_name}. Choose from {DGP_CHOICES}")


def generate_censored_data(seed: int, n_samples: int = 1000, dgp_name: str = "gmm") -> tuple[pd.DataFrame, np.ndarray]:
    """Generate right-censored data from specified DGP."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    sampler = create_dgp(dgp_name)
    
    X_true = sampler.generate_samples(n_samples)
    C = rng.uniform(0.0, 1.0, size=n_samples)
    
    T = np.minimum(X_true, C)
    Delta = (X_true <= C).astype(int)
    
    data = pd.DataFrame({"T": T, "Delta": Delta})
    return data, X_true


def compute_cv_score(
    data: pd.DataFrame,
    norm_constraint: float,
    basis_order: int = 0,
    cv_folds: int = 5,
    random_state: int = 42,
    n_grid_points: int = 200,
) -> tuple[float, list[float]]:
    """
    Compute CV score for a given norm_constraint.
    
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
            
            est = RightCensoredIPCWEstimator(
                tol=1e-6,
                norm_constraint=norm_constraint,
                n_grid_points=n_grid_points,
                basis_order=basis_order,
                solver="ECOS",
                use_secondary_solver=True,  # Enable fallback solver
            ).fit(df_unc, sample_weights=w_unc)
            
            # Evaluate on validation
            score = incomplete_loglik(est, val_df, time_col="T", delta_col="Delta")
            scores.append(score)
        except Exception as e:
            print(f"    Warning: fold failed with {e}")
            scores.append(float("-inf"))
    
    mean_score = float(np.mean(scores))
    return mean_score, scores


def process_single_seed(args: tuple) -> dict:
    """
    Process a single seed - designed for parallel execution.
    
    Args:
        args: Tuple of (seed, norm_constraints, basis_order, cv_folds, cv_random_state, n_samples, dgp_name)
    
    Returns:
        Dict with results for this seed
    """
    seed, norm_constraints, basis_order, cv_folds, cv_random_state, n_samples, dgp_name = args
    
    # Recreate true_pdf_fn in worker process
    sampler = create_dgp(dgp_name)
    true_pdf_fn = sampler.compute_density
    
    # Generate data
    data, X_true = generate_censored_data(seed, n_samples, dgp_name)
    censoring_rate = (1 - data["Delta"].mean())
    
    # Storage for this seed
    cv_scores = []
    cv_sds = []  # Standard deviation of fold scores for each lambda
    full_ll = []
    full_kl = []
    
    # Iterate over norm constraints
    for nc in norm_constraints:
        # CV score
        mean_cv, fold_scores = compute_cv_score(
            data=data,
            norm_constraint=nc,
            basis_order=basis_order,
            cv_folds=cv_folds,
            random_state=cv_random_state,
        )
        cv_scores.append(mean_cv)
        # Compute SD of fold scores (excluding -inf values from failed folds)
        valid_fold_scores = [s for s in fold_scores if s != float("-inf")]
        if len(valid_fold_scores) >= 2:
            cv_sds.append(np.std(valid_fold_scores))
        else:
            cv_sds.append(float("nan"))
        
        # Full data metrics
        metrics = compute_full_data_metrics(
            data=data,
            norm_constraint=nc,
            true_pdf_fn=true_pdf_fn,
            basis_order=basis_order,
        )
        full_ll.append(metrics["ll"])
        full_kl.append(metrics["kl"])
    
    # Compute conservative lambda selections using k*SD rule (Option A)
    # threshold = max(CV_LL) - k * SD_at_argmax
    # Select smallest lambda where CV_LL >= threshold
    cv_scores_arr = np.array(cv_scores)
    cv_sds_arr = np.array(cv_sds)
    
    conservative_selections = {}
    if not np.all(np.isnan(cv_scores_arr)):
        best_cv_idx = np.nanargmax(cv_scores_arr)
        best_cv_ll = cv_scores_arr[best_cv_idx]
        sd_at_best = cv_sds_arr[best_cv_idx]
        
        # Different k values: 1%, 5%, 10%
        for k in [0.01, 0.05, 0.1]:
            threshold = best_cv_ll - k * sd_at_best
            # Find smallest lambda (first index) where CV_LL >= threshold
            valid_indices = np.where(cv_scores_arr >= threshold)[0]
            if len(valid_indices) > 0:
                conservative_idx = valid_indices[0]  # Smallest lambda meeting threshold
                conservative_selections[f"k_{k}"] = {
                    "idx": int(conservative_idx),
                    "nc": float(norm_constraints[conservative_idx]),
                    "cv_ll": float(cv_scores_arr[conservative_idx]),
                }
            else:
                conservative_selections[f"k_{k}"] = None
    else:
        for k in [0.01, 0.05, 0.1]:
            conservative_selections[f"k_{k}"] = None
    
    return {
        "seed": seed,
        "norm_constraints": norm_constraints,
        "cv_scores": cv_scores_arr,
        "cv_sds": cv_sds_arr,  # SD of CV scores per lambda
        "full_ll": np.array(full_ll),
        "full_kl": np.array(full_kl),
        "censoring_rate": censoring_rate,
        "conservative_selections": conservative_selections,  # k*SD rule selections
    }


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
            use_secondary_solver=True,  # Enable fallback solver
        ).fit(df_unc, sample_weights=w_unc)
        
        eval_grid = np.linspace(0, 1, 500)
        density = est.get_density_at_points(eval_grid)
        
        ll = incomplete_loglik(est, data, time_col="T", delta_col="Delta")
        kl = kl_divergence(true_pdf_fn=true_pdf_fn, grid=eval_grid, est_density=density)
        
        return {"ll": ll, "kl": kl, "density": density, "grid": eval_grid}
    except Exception as e:
        # Return NaN for failed fits
        eval_grid = np.linspace(0, 1, 500)
        return {"ll": float("nan"), "kl": float("nan"), "density": np.full_like(eval_grid, np.nan), "grid": eval_grid}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Explore CV metric landscape for different norm constraints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dgp", type=str, default="gmm", choices=DGP_CHOICES,
        help="Data generating process: gmm (3-component GMM), truncnorm (truncated normal), "
             "sinusoidal (sin-based), step (step function)"
    )
    parser.add_argument(
        "--n_seeds", type=int, default=200,
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
        "--nc_min", type=float, default= 0.25,
        help="Minimum norm constraint"
    )
    parser.add_argument(
        "--nc_max", type=float, default= 25.0,
        help="Maximum norm constraint"
    )
    parser.add_argument(
        "--nc_step", type=float, default= 0.25,
        help="Norm constraint step size"
    )
    parser.add_argument(
        "--output_prefix", type=str, default=None,
        help="Output file prefix. If not provided, uses dgp_timestamp"
    )
    parser.add_argument(
        "--seed_base", type=int, default=2024,
        help="Base seed for reproducible seed generation"
    )
    parser.add_argument(
        "--n_samples", type=int, default=1000,
        help="Number of samples per dataset"
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Configuration
    dgp_name = args.dgp
    n_seeds = args.n_seeds
    n_workers = args.n_workers
    np.random.seed(args.seed_base)  # For reproducibility of seed generation
    seeds = np.random.randint(1, 100000, size=n_seeds).tolist()
    
    norm_constraints = np.arange(args.nc_min, args.nc_max, args.nc_step)
    basis_order = args.basis_order
    cv_folds = 5
    cv_random_state = 42
    n_samples = args.n_samples
    
    # Generate output prefix with timestamp if not provided
    if args.output_prefix:
        output_prefix = f"{args.output_prefix}_{dgp_name}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"cv_landscape_{dgp_name}_{timestamp}"
    
    # Grid layout - compute based on n_seeds
    n_cols = 10
    n_rows = (n_seeds + n_cols - 1) // n_cols  # Ceiling division
    
    # Get DGP description
    dgp_descriptions = {
        "gmm": "3-Component GMM (peaks at 0.2, 0.5, 0.8)",
        "truncnorm": "Truncated Normal (μ=0.5, σ=0.15)",
        "sinusoidal": "Sinusoidal (sin(πx) + 1.1)",
        "step": "Step Function (1.5 → 0.5 at x=0.4)",
    }
    
    print("=" * 70)
    print("CV Metric Landscape Exploration (Large Scale - Parallel)")
    print("=" * 70)
    print(f"DGP: {dgp_name} - {dgp_descriptions[dgp_name]}")
    print(f"Number of samples: {n_samples}")
    print(f"Number of seeds: {n_seeds}")
    print(f"Number of workers: {n_workers}")
    print(f"Norm constraints: {norm_constraints[0]} to {norm_constraints[-1]} (step {args.nc_step})")
    print(f"Number of norm constraints: {len(norm_constraints)}")
    print(f"Basis order: {basis_order}")
    print(f"CV folds: {cv_folds}")
    print(f"Output prefix: {output_prefix}")
    print("=" * 70)
    
    # Store results
    all_results = {}
    
    # Prepare arguments for parallel processing
    task_args = [
        (seed, norm_constraints, basis_order, cv_folds, cv_random_state, n_samples, dgp_name)
        for seed in seeds
    ]
    
    # Run in parallel
    print(f"\nProcessing {n_seeds} seeds with {n_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_seed, arg): arg[0] for arg in task_args}
        
        # Process results as they complete
        with tqdm(total=n_seeds, desc="Seeds") as pbar:
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    result = future.result()
                    all_results[result["seed"]] = result
                    
                    # Brief summary
                    cv_arr = result["cv_scores"]
                    kl_arr = result["full_kl"]
                    if not np.all(np.isnan(cv_arr)) and not np.all(np.isnan(kl_arr)):
                        best_cv_idx = np.nanargmax(cv_arr)
                        best_kl_idx = np.nanargmin(kl_arr)
                        gap = norm_constraints[best_cv_idx] - norm_constraints[best_kl_idx]
                        pbar.set_postfix({
                            "seed": seed, 
                            "gap": f"{gap:.1f}",
                            "censor": f"{result['censoring_rate']:.0%}"
                        })
                except Exception as e:
                    print(f"\nError processing seed {seed}: {e}")
                
                pbar.update(1)
    
    # Reorder results by seed order
    all_results = {seed: all_results[seed] for seed in seeds if seed in all_results}
    seeds = list(all_results.keys())  # Update seeds list to only include successful ones
    n_seeds = len(seeds)
    
    # Recalculate grid layout based on actual number of successful seeds
    n_cols = 10
    n_rows = (n_seeds + n_cols - 1) // n_cols  # Ceiling division
    
    print(f"\nSuccessfully processed {n_seeds} seeds")
    
    # Plotting
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)
    
    # Compute summary statistics for all seeds
    best_cv_ncs = []
    best_kl_ncs = []
    gaps = []
    for seed in seeds:
        res = all_results[seed]
        cv_scores = res["cv_scores"]
        kl_scores = res["full_kl"]
        
        if not np.all(np.isnan(cv_scores)):
            best_cv_idx = np.nanargmax(cv_scores)
            best_cv_nc = res["norm_constraints"][best_cv_idx]
        else:
            best_cv_nc = float("nan")
        
        if not np.all(np.isnan(kl_scores)):
            best_kl_idx = np.nanargmin(kl_scores)
            best_kl_nc = res["norm_constraints"][best_kl_idx]
        else:
            best_kl_nc = float("nan")
        
        best_cv_ncs.append(best_cv_nc)
        best_kl_ncs.append(best_kl_nc)
        if not np.isnan(best_cv_nc) and not np.isnan(best_kl_nc):
            gaps.append(best_cv_nc - best_kl_nc)
        else:
            gaps.append(float("nan"))
    
    best_cv_ncs = np.array(best_cv_ncs)
    best_kl_ncs = np.array(best_kl_ncs)
    gaps = np.array(gaps)
    
    # Figure 1: CV score landscape for all seeds - scale with n_seeds
    # Each subplot gets 3x2.5 inches for good readability
    subplot_width = 3
    subplot_height = 2.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * subplot_width, n_rows * subplot_height))
    if n_seeds == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, seed in enumerate(seeds):
        ax = axes[i]
        res = all_results[seed]
        
        # Plot CV score with error bands (± 1 SD)
        cv_scores = res["cv_scores"]
        cv_sds = res["cv_sds"]
        nc = res["norm_constraints"]
        
        ax.plot(nc, cv_scores, "b-", linewidth=1.2)
        
        # Add shaded error band for ± 1 SD
        ax.fill_between(
            nc, 
            cv_scores - cv_sds, 
            cv_scores + cv_sds, 
            alpha=0.2, 
            color="blue",
            label="±1 SD"
        )
        
        # Handle NaN values
        kl_scores = res["full_kl"]
        
        if not np.all(np.isnan(cv_scores)):
            best_cv_idx = np.nanargmax(cv_scores)
            ax.axvline(nc[best_cv_idx], color="b", linestyle="--", alpha=0.7, linewidth=1.2)
            
            # Show threshold line for 1% SD rule
            sd_at_best = cv_sds[best_cv_idx]
            threshold_1pct = cv_scores[best_cv_idx] - 0.01 * sd_at_best
            ax.axhline(threshold_1pct, color="green", linestyle=":", alpha=0.5, linewidth=1)
        
        if not np.all(np.isnan(kl_scores)):
            best_kl_idx = np.nanargmin(kl_scores)
            ax.axvline(nc[best_kl_idx], color="r", linestyle="--", alpha=0.7, linewidth=1.2)
        
        # Mark conservative selection (1% SD rule) with green
        cons_sel = res["conservative_selections"].get("k_0.01")
        if cons_sel is not None:
            ax.axvline(cons_sel["nc"], color="green", linestyle="-", alpha=0.7, linewidth=1.5)
        
        ax.set_title(f"Seed {seed}", fontsize=9, pad=2)
        ax.tick_params(axis='both', which='both', labelsize=7, pad=2)
        ax.grid(alpha=0.3, linewidth=0.5)
        
        # Only add labels on edge subplots
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Norm Constraint", fontsize=8)
        if i % n_cols == 0:
            ax.set_ylabel("CV LL", fontsize=8)
    
    # Hide unused subplots
    for i in range(n_seeds, n_rows * n_cols):
        axes[i].axis("off")
    
    plt.suptitle(f"CV Log-Likelihood Landscape ({n_seeds} Seeds, DGP={dgp_name}) | Blue=Best CV, Red=Best KL, Green=1% SD Rule", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_seeds.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_prefix}_seeds.png")
    
    # Figure 2: CV score vs KL divergence comparison - scale with n_seeds
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * subplot_width, n_rows * subplot_height))
    if n_seeds == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, seed in enumerate(seeds):
        ax = axes[i]
        res = all_results[seed]
        
        # Twin axis for KL
        ax2 = ax.twinx()
        
        ax.plot(res["norm_constraints"], res["cv_scores"], "b-", linewidth=1.2)
        ax2.plot(res["norm_constraints"], res["full_kl"], "r-", linewidth=1.2)
        
        ax.tick_params(axis="y", labelcolor="b", labelsize=7, pad=2)
        ax2.tick_params(axis="y", labelcolor="r", labelsize=7, pad=2)
        ax.tick_params(axis="x", labelsize=7, pad=2)
        ax.set_title(f"Seed {seed}", fontsize=9, pad=2)
        ax.grid(alpha=0.3, linewidth=0.5)
        
        # Only add labels on edge subplots
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Norm Constraint", fontsize=8)
        if i % n_cols == 0:
            ax.set_ylabel("CV LL", fontsize=8, color="b")
        if (i + 1) % n_cols == 0:
            ax2.set_ylabel("KL Div", fontsize=8, color="r")
    
    # Hide unused subplots
    for i in range(n_seeds, n_rows * n_cols):
        axes[i].axis("off")
    
    plt.suptitle(f"CV LL (Blue) vs KL Divergence (Red) ({n_seeds} Seeds, DGP={dgp_name})", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_cv_vs_kl.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_prefix}_cv_vs_kl.png")
    
    # Figure 3: Aggregated view - all seeds overlaid (now 3x2 for SD analysis)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    # Normalize CV scores for comparison (z-score per seed)
    ax = axes[0, 0]
    for seed in seeds:
        res = all_results[seed]
        cv_scores = res["cv_scores"]
        # Normalize (handle NaN)
        mean_val = np.nanmean(cv_scores)
        std_val = np.nanstd(cv_scores)
        if std_val > 0:
            cv_norm = (cv_scores - mean_val) / std_val
        else:
            cv_norm = cv_scores - mean_val
        ax.plot(res["norm_constraints"], cv_norm, linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Norm Constraint")
    ax.set_ylabel("Normalized CV LL (z-score)")
    ax.set_title(f"Normalized CV Landscape ({n_seeds} Seeds, {dgp_name})")
    ax.grid(alpha=0.3)
    
    # KL divergence comparison
    ax = axes[0, 1]
    for seed in seeds:
        res = all_results[seed]
        ax.plot(res["norm_constraints"], res["full_kl"], linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Norm Constraint")
    ax.set_ylabel("KL Divergence")
    ax.set_title(f"KL Divergence Landscape ({n_seeds} Seeds, {dgp_name})")
    ax.grid(alpha=0.3)
    
    # NEW: CV SD landscape - individual seeds
    ax = axes[1, 0]
    for seed in seeds:
        res = all_results[seed]
        ax.plot(res["norm_constraints"], res["cv_sds"], linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Norm Constraint")
    ax.set_ylabel("CV LL Standard Deviation")
    ax.set_title(f"CV SD Landscape ({n_seeds} Seeds, {dgp_name})")
    ax.grid(alpha=0.3)
    
    # NEW: Mean CV SD with confidence band across seeds
    ax = axes[1, 1]
    # Stack all CV SDs and compute mean/std across seeds for each norm constraint
    cv_sd_matrix = np.array([all_results[seed]["cv_sds"] for seed in seeds])
    mean_cv_sd = np.nanmean(cv_sd_matrix, axis=0)
    std_cv_sd = np.nanstd(cv_sd_matrix, axis=0)
    
    ax.plot(norm_constraints, mean_cv_sd, "b-", linewidth=2, label="Mean CV SD")
    ax.fill_between(
        norm_constraints,
        mean_cv_sd - std_cv_sd,
        mean_cv_sd + std_cv_sd,
        alpha=0.3,
        color="blue",
        label="±1 SD across seeds"
    )
    ax.set_xlabel("Norm Constraint")
    ax.set_ylabel("CV LL Standard Deviation")
    ax.set_title(f"Mean CV SD vs Norm Constraint ({n_seeds} Seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Collect conservative selection NCs
    cons_ncs_1pct = []
    cons_ncs_5pct = []
    cons_ncs_10pct = []
    for seed in seeds:
        res = all_results[seed]
        cons_1 = res["conservative_selections"].get("k_0.01")
        cons_5 = res["conservative_selections"].get("k_0.05")
        cons_10 = res["conservative_selections"].get("k_0.1")
        cons_ncs_1pct.append(cons_1["nc"] if cons_1 else float("nan"))
        cons_ncs_5pct.append(cons_5["nc"] if cons_5 else float("nan"))
        cons_ncs_10pct.append(cons_10["nc"] if cons_10 else float("nan"))
    cons_ncs_1pct = np.array(cons_ncs_1pct)
    cons_ncs_5pct = np.array(cons_ncs_5pct)
    cons_ncs_10pct = np.array(cons_ncs_10pct)
    
    # Distribution of best CV vs conservative vs best KL
    ax = axes[2, 0]
    valid_mask = ~np.isnan(best_cv_ncs) & ~np.isnan(best_kl_ncs)
    ax.hist(best_cv_ncs[valid_mask], bins=30, alpha=0.5, label=f"Best CV (mean={np.nanmean(best_cv_ncs):.2f})", color="blue")
    ax.hist(cons_ncs_1pct[~np.isnan(cons_ncs_1pct)], bins=30, alpha=0.5, label=f"1% SD Rule (mean={np.nanmean(cons_ncs_1pct):.2f})", color="green")
    ax.hist(best_kl_ncs[valid_mask], bins=30, alpha=0.5, label=f"Best KL (mean={np.nanmean(best_kl_ncs):.2f})", color="red")
    ax.axvline(np.nanmean(best_cv_ncs), color="blue", linestyle="--", linewidth=2)
    ax.axvline(np.nanmean(cons_ncs_1pct), color="green", linestyle="--", linewidth=2)
    ax.axvline(np.nanmean(best_kl_ncs), color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Norm Constraint")
    ax.set_ylabel("Count")
    ax.set_title("Distribution: Best CV vs 1% SD Rule vs Best KL")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Compare gaps for different k values
    ax = axes[2, 1]
    gaps_cv = gaps  # Already computed: best_cv - best_kl
    gaps_1pct = cons_ncs_1pct - best_kl_ncs
    gaps_5pct = cons_ncs_5pct - best_kl_ncs
    
    valid_cv = ~np.isnan(gaps_cv)
    valid_1pct = ~np.isnan(gaps_1pct)
    valid_5pct = ~np.isnan(gaps_5pct)
    
    ax.hist(gaps_cv[valid_cv], bins=30, alpha=0.5, label=f"CV gap (mean={np.nanmean(gaps_cv):.2f})", color="blue")
    ax.hist(gaps_1pct[valid_1pct], bins=30, alpha=0.5, label=f"1% SD gap (mean={np.nanmean(gaps_1pct):.2f})", color="green")
    ax.hist(gaps_5pct[valid_5pct], bins=30, alpha=0.5, label=f"5% SD gap (mean={np.nanmean(gaps_5pct):.2f})", color="orange")
    ax.axvline(0, color="black", linestyle="-", linewidth=2, label="Zero (optimal)")
    ax.set_xlabel("Gap (Selected NC - Best KL NC)")
    ax.set_ylabel("Count")
    ax.set_title("Gap Comparison: CV vs Conservative Rules")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.suptitle(f"CV Landscape Analysis Summary ({n_seeds} Seeds, DGP={dgp_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_aggregated.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_prefix}_aggregated.png")
    
    # Figure 4: Summary scatter plot and statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Scatter plot: Best CV NC vs Best KL NC
    ax = axes[0]
    valid_mask = ~np.isnan(best_cv_ncs) & ~np.isnan(best_kl_ncs)
    ax.scatter(best_kl_ncs[valid_mask], best_cv_ncs[valid_mask], alpha=0.5, s=20)
    min_val = min(np.nanmin(best_kl_ncs), np.nanmin(best_cv_ncs))
    max_val = max(np.nanmax(best_kl_ncs), np.nanmax(best_cv_ncs))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="y=x (perfect)")
    ax.set_xlabel("Best KL Norm Constraint (True Optimal)")
    ax.set_ylabel("Best CV Norm Constraint (Selected)")
    ax.set_title("CV Selection vs True Optimal")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Correlation
    corr = np.corrcoef(best_kl_ncs[valid_mask], best_cv_ncs[valid_mask])[0, 1]
    ax.text(0.05, 0.95, f"Corr: {corr:.3f}", transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Gap vs censoring rate
    ax = axes[1]
    censoring_rates = np.array([all_results[seed]["censoring_rate"] for seed in seeds])
    ax.scatter(censoring_rates[valid_mask], gaps[valid_mask], alpha=0.5, s=20)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel("Censoring Rate")
    ax.set_ylabel("Gap (Best CV NC - Best KL NC)")
    ax.set_title("Gap vs Censoring Rate")
    ax.grid(alpha=0.3)
    
    # KL at best CV vs KL at best KL
    ax = axes[2]
    kl_at_cv = []
    kl_at_kl = []
    for seed in seeds:
        res = all_results[seed]
        cv_scores = res["cv_scores"]
        kl_scores = res["full_kl"]
        
        if not np.all(np.isnan(cv_scores)) and not np.all(np.isnan(kl_scores)):
            best_cv_idx = np.nanargmax(cv_scores)
            best_kl_idx = np.nanargmin(kl_scores)
            kl_at_cv.append(kl_scores[best_cv_idx])
            kl_at_kl.append(kl_scores[best_kl_idx])
        else:
            kl_at_cv.append(float("nan"))
            kl_at_kl.append(float("nan"))
    
    kl_at_cv = np.array(kl_at_cv)
    kl_at_kl = np.array(kl_at_kl)
    valid_kl_mask = ~np.isnan(kl_at_cv) & ~np.isnan(kl_at_kl)
    
    ax.scatter(kl_at_kl[valid_kl_mask], kl_at_cv[valid_kl_mask], alpha=0.5, s=20)
    max_kl = max(np.nanmax(kl_at_cv), np.nanmax(kl_at_kl))
    ax.plot([0, max_kl], [0, max_kl], 'r--', linewidth=2, label="y=x")
    ax.set_xlabel("KL at Best KL NC (Achievable)")
    ax.set_ylabel("KL at Best CV NC (Actual)")
    ax.set_title("Actual KL vs Achievable KL")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add statistics
    mean_excess_kl = np.nanmean(kl_at_cv - kl_at_kl)
    ax.text(0.05, 0.95, f"Mean excess KL: {mean_excess_kl:.4f}", transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"CV Selection Analysis ({n_seeds} Seeds, DGP={dgp_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_prefix}_summary.png")
    
    # Collect conservative selection data for summary
    cons_ncs_1pct = []
    cons_ncs_5pct = []
    for seed in seeds:
        res = all_results[seed]
        cons_1 = res["conservative_selections"].get("k_0.01")
        cons_5 = res["conservative_selections"].get("k_0.05")
        cons_ncs_1pct.append(cons_1["nc"] if cons_1 else float("nan"))
        cons_ncs_5pct.append(cons_5["nc"] if cons_5 else float("nan"))
    cons_ncs_1pct = np.array(cons_ncs_1pct)
    cons_ncs_5pct = np.array(cons_ncs_5pct)
    
    # Compute KL at conservative selections
    kl_at_1pct = []
    kl_at_5pct = []
    for seed in seeds:
        res = all_results[seed]
        kl_scores = res["full_kl"]
        cons_1 = res["conservative_selections"].get("k_0.01")
        cons_5 = res["conservative_selections"].get("k_0.05")
        if cons_1 is not None and not np.all(np.isnan(kl_scores)):
            kl_at_1pct.append(kl_scores[cons_1["idx"]])
        else:
            kl_at_1pct.append(float("nan"))
        if cons_5 is not None and not np.all(np.isnan(kl_scores)):
            kl_at_5pct.append(kl_scores[cons_5["idx"]])
        else:
            kl_at_5pct.append(float("nan"))
    kl_at_1pct = np.array(kl_at_1pct)
    kl_at_5pct = np.array(kl_at_5pct)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"DGP: {dgp_name} - {dgp_descriptions[dgp_name]}")
    print(f"Number of samples: {n_samples}")
    print(f"Number of seeds: {n_seeds}")
    print(f"Valid seeds: {np.sum(valid_mask)}")
    print(f"\nBest CV Norm Constraint:")
    print(f"  Mean: {np.nanmean(best_cv_ncs):.2f}")
    print(f"  Std:  {np.nanstd(best_cv_ncs):.2f}")
    print(f"  Min:  {np.nanmin(best_cv_ncs):.2f}")
    print(f"  Max:  {np.nanmax(best_cv_ncs):.2f}")
    print(f"\n1% SD Rule Norm Constraint:")
    print(f"  Mean: {np.nanmean(cons_ncs_1pct):.2f}")
    print(f"  Std:  {np.nanstd(cons_ncs_1pct):.2f}")
    print(f"  Mean reduction from CV: {np.nanmean(best_cv_ncs - cons_ncs_1pct):.2f}")
    print(f"\n5% SD Rule Norm Constraint:")
    print(f"  Mean: {np.nanmean(cons_ncs_5pct):.2f}")
    print(f"  Std:  {np.nanstd(cons_ncs_5pct):.2f}")
    print(f"  Mean reduction from CV: {np.nanmean(best_cv_ncs - cons_ncs_5pct):.2f}")
    print(f"\nBest KL Norm Constraint (True Optimal):")
    print(f"  Mean: {np.nanmean(best_kl_ncs):.2f}")
    print(f"  Std:  {np.nanstd(best_kl_ncs):.2f}")
    print(f"  Min:  {np.nanmin(best_kl_ncs):.2f}")
    print(f"  Max:  {np.nanmax(best_kl_ncs):.2f}")
    print(f"\nGap (Selected NC - Best KL NC):")
    print(f"  CV:       Mean={np.nanmean(gaps):.2f}, Std={np.nanstd(gaps):.2f}")
    print(f"  1% SD:    Mean={np.nanmean(cons_ncs_1pct - best_kl_ncs):.2f}, Std={np.nanstd(cons_ncs_1pct - best_kl_ncs):.2f}")
    print(f"  5% SD:    Mean={np.nanmean(cons_ncs_5pct - best_kl_ncs):.2f}, Std={np.nanstd(cons_ncs_5pct - best_kl_ncs):.2f}")
    print(f"\nCorrelation (Best CV NC vs Best KL NC): {corr:.3f}")
    print(f"\nMean KL Divergence:")
    print(f"  At Best CV:   {np.nanmean(kl_at_cv):.4f}")
    print(f"  At 1% SD:     {np.nanmean(kl_at_1pct):.4f}")
    print(f"  At 5% SD:     {np.nanmean(kl_at_5pct):.4f}")
    print(f"  At Best KL:   {np.nanmean(kl_at_kl):.4f} (achievable)")
    print(f"\nExcess KL (relative to best KL):")
    print(f"  CV selection: {np.nanmean(kl_at_cv - kl_at_kl):.4f}")
    print(f"  1% SD rule:   {np.nanmean(kl_at_1pct - kl_at_kl):.4f}")
    print(f"  5% SD rule:   {np.nanmean(kl_at_5pct - kl_at_kl):.4f}")
    
    print("\n" + "=" * 70)
    print("Done! All plots saved.")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    results = main()

