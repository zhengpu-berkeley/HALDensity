import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Optional, Type, Any
from haldensity.estimation.base_estimator import BaseEstimator
from enum import Enum


class CVSelectorMetric(Enum):
    SLL = "sll"
    BIC = "bic"


class KFold_CV_LambdaSelector:
    """
    A class to select the best lambda value for regularization in a model.
    """

    def __init__(
            self,
            method: Type[BaseEstimator],  # Changed to Type[BaseEstimator]
            full_data: pd.DataFrame,
            lambdas: list[float],
            folds: int = 5,
            tolerance: float = 1e-3,
            max_iter: int = 10_000,
            random_state: int = 42,
            metric: CVSelectorMetric = CVSelectorMetric.SLL,
            do_warm_start: bool = False,
            **estimator_kwargs: Any  # Added to capture other estimator params
        ):
        """
        Initializes the LambdaSelector with a list of lambda values.

        :param method: The estimator class to use
        :param full_data: The full dataset for cross-validation
        :param lambdas: list of lambda values to consider
        :param folds: Number of CV folds
        :param tolerance: Tolerance for convergence
        :param max_iter: Maximum iterations
        :param random_state: Random state for reproducibility
        :param metric: Metric to use for lambda selection (SLL or BIC)
        :param estimator_kwargs: Additional estimator parameters
        """
        self.method_class = method # Store the class
        self.full_data = full_data
        if len(lambdas) == 0:
            raise ValueError("The list of lambdas cannot be empty.")
        # Limit the number of lambdas to avoid excessive computation time
        if len(lambdas) > 20: # Example limit
            print(f"Warning: Number of lambdas ({len(lambdas)}) is high, CV might be slow.")
        # sort lambdas to ensure consistent order larger to smaller
        self.lambdas = sorted(lambdas, reverse=True)  # Sort from largest
        self.folds = folds
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_state = random_state
        self.metric = metric
        self.estimator_kwargs = estimator_kwargs # Store additional args
        self.best_lambda: Optional[float] = None
        self.best_metric_value: float = -np.inf if metric == CVSelectorMetric.SLL else np.inf
        # Store both metrics for each lambda and fold
        self.cv_results: dict[float, dict[str, list[float]]] = {
            l: {"sll": [], "bic": [], "n_selected_knots": []} for l in lambdas
        }
        # Store fitted theta for warm start if needed
        self.do_warm_start: bool = do_warm_start
        self.warm_start_fitted_theta: Optional[np.ndarray] = None

    def select_lambda(self, do_plot_fited: bool = False) -> Optional[float]:
        """
        Select the best lambda value based on cross-validation using the specified metric.
        Iterates through each lambda, performs K-fold CV, and records both SLL and BIC
        for each lambda on validation sets. The lambda with the best metric value is chosen.
        """
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=self.random_state)
        avg_metrics_per_lambda: dict[float, float] = {}

        for lam_idx, lambda_val in enumerate(self.lambdas):
            print(f"\n\nEvaluating lambda: {lambda_val} (Index {lam_idx + 1}/{len(self.lambdas)})\n\n")
            current_lambda_fold_slls: list[float] = []
            current_lambda_fold_bics: list[float] = []
            # This will hold the coefficients from the last successful fold for this lambda
            last_successful_theta_for_lambda: Optional[np.ndarray] = None

            for fold_idx, (train_index, val_index) in enumerate(kf.split(self.full_data)):
                print(f"Processing fold {fold_idx + 1}/{self.folds} for lambda={lambda_val}")
                train_data = self.full_data.iloc[train_index]
                val_data = self.full_data.iloc[val_index]

                try:
                    # Instantiate the estimator for the current lambda and fold
                    estimator = self.method_class(
                        lam=lambda_val,
                        n_iterations=self.max_iter,
                        tol=self.tolerance,
                        **self.estimator_kwargs # Pass stored additional args
                    )

                    # If warm start is enabled, use the fitted theta from the previous lambda
                    if self.do_warm_start and self.warm_start_fitted_theta is not None:
                        print(f"Using warm start with fitted theta from previous lambda for lambda={lambda_val}, fold={fold_idx+1}.")
                        estimator.fit(train_data, warm_start_coefficients=self.warm_start_fitted_theta)
                    else:
                        estimator.fit(train_data)

                    # Plot fitted density if requested
                    if do_plot_fited:
                        
                        fig = estimator.plot_estimator_results(
                            data=train_data,
                            title=f"{estimator.__class__.__name__} Fitted Density for Lambda={lambda_val}, Fold={fold_idx+1}",
                        )
                    
                    if 'W1' not in val_data.columns:
                        print(f"Warning: 'W1' column not found in validation data for lambda={lambda_val}, fold={fold_idx+1}.")
                        sum_ll = -np.inf
                        bic_val = np.inf
                    elif val_data.empty:
                        print(f"Warning: Validation data is empty for lambda={lambda_val}, fold={fold_idx+1}.")
                        sum_ll = -np.inf
                        bic_val = np.inf
                    else:
                        # Compute SLL
                        sum_ll = estimator.get_avg_log_likelihood_for_points(np.asarray(val_data['W1'].values))
                        
                        # Compute BIC
                        try:
                            bic_val = estimator.compute_bic(val_data)
                        except Exception as bic_error:
                            print(f"Warning: BIC computation failed for lambda={lambda_val}, fold={fold_idx+1}: {bic_error}")
                            bic_val = np.inf
                        
                        print(f"Fold {fold_idx+1} - SLL: {sum_ll:.4f}, BIC: {bic_val:.4f}")
                    
                    current_lambda_fold_slls.append(sum_ll)
                    current_lambda_fold_bics.append(bic_val)

                    fitted_results = estimator.get_results()
                    fitted_n_knot = fitted_results.get('n_selected_knots', np.inf)  

                    # fitted theta (to be used for warm start in next iterations)
                    fitted_theta_hat = fitted_results.get('theta_hat', None)
                    if fitted_theta_hat is not None and self.do_warm_start:
                        # Store the successful theta. This will be the last one from this lambda's folds.
                        last_successful_theta_for_lambda = fitted_theta_hat
                    
                    # Store both metrics
                    self.cv_results[lambda_val]["sll"].append(sum_ll)
                    self.cv_results[lambda_val]["bic"].append(bic_val)
                    self.cv_results[lambda_val]["n_selected_knots"].append(fitted_n_knot)

                except Exception as e:
                    print(f"Error during CV for lambda={lambda_val}, fold={fold_idx+1}: {e}")
                    current_lambda_fold_slls.append(-np.inf)
                    current_lambda_fold_bics.append(np.inf)
                    self.cv_results[lambda_val]["sll"].append(-np.inf)
                    self.cv_results[lambda_val]["bic"].append(np.inf)
                    self.cv_results[lambda_val]["n_selected_knots"].append(np.inf)
            
            # After all folds for a lambda are done, update the warm_start_theta for the next lambda
            # This happens only if at least one fold was successful for the current lambda.
            if self.do_warm_start and last_successful_theta_for_lambda is not None:
                self.warm_start_fitted_theta = last_successful_theta_for_lambda
            
            # Calculate average metric based on selected metric type
            if self.metric == CVSelectorMetric.SLL:
                # For SLL, higher is better, filter out -inf
                valid_values = [val for val in current_lambda_fold_slls if val > -np.inf and not np.isnan(val)]
                if valid_values:
                    avg_metrics_per_lambda[lambda_val] = np.mean(valid_values)
                else:
                    avg_metrics_per_lambda[lambda_val] = -np.inf
            else:  # BIC
                # For BIC, lower is better, filter out inf
                valid_values = [val for val in current_lambda_fold_bics if val < np.inf and not np.isnan(val)]
                if valid_values:
                    avg_metrics_per_lambda[lambda_val] = np.mean(valid_values)
                else:
                    avg_metrics_per_lambda[lambda_val] = np.inf
        
        # Select best lambda based on metric
        if self.metric == CVSelectorMetric.SLL:
            # For SLL, higher is better
            valid_avg_metrics = {
                l: val for l, val in avg_metrics_per_lambda.items() if val > -np.inf and not np.isnan(val)
            }
            if not valid_avg_metrics:
                print("Warning: Cross-validation did not yield any valid SLL values.")
                self.best_lambda = None
                self.best_metric_value = -np.inf
            else:
                self.best_lambda = max(valid_avg_metrics, key=valid_avg_metrics.get)
                self.best_metric_value = valid_avg_metrics[self.best_lambda]
                print(f"\nBest lambda: {self.best_lambda} with average SLL: {self.best_metric_value:.4f}")
        else:  # BIC
            # For BIC, lower is better
            valid_avg_metrics = {
                l: val for l, val in avg_metrics_per_lambda.items() if val < np.inf and not np.isnan(val)
            }
            if not valid_avg_metrics:
                print("Warning: Cross-validation did not yield any valid BIC values.")
                self.best_lambda = None
                self.best_metric_value = np.inf
            else:
                self.best_lambda = min(valid_avg_metrics, key=valid_avg_metrics.get)
                self.best_metric_value = valid_avg_metrics[self.best_lambda]
                print(f"\nBest lambda: {self.best_lambda} with average BIC: {self.best_metric_value:.4f}")
        
        return self.best_lambda

    def get_cv_results_summary(self) -> dict[float, dict[str, float]]:
        """
        Get a summary of CV results with average SLL and BIC for each lambda.
        
        Returns:
            dictionary with lambda values as keys and average metrics as values
        """
        summary = {}
        for lambda_val in self.lambdas:
            sll_values = [val for val in self.cv_results[lambda_val]["sll"] if val > -np.inf and not np.isnan(val)]
            bic_values = [val for val in self.cv_results[lambda_val]["bic"] if val < np.inf and not np.isnan(val)]
            n_knot_values = [val for val in self.cv_results[lambda_val]["n_selected_knots"] if val != np.inf]
            
            summary[float(lambda_val)] = {
                "avg_sll": float(np.mean(sll_values)) if sll_values else float('-inf'),
                "avg_bic": float(np.mean(bic_values)) if bic_values else float('inf'),
                "avg_n_selected_knots": float(np.mean(n_knot_values)) if n_knot_values else float('inf'),
                "n_valid_folds_sll": int(len(sll_values)),
                "n_valid_folds_bic": int(len(bic_values))
            }
        
        return summary
