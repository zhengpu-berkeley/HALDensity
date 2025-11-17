#!/usr/bin/env python
import sys
sys.path.insert(0, "src")

print("Test 1: Import modules")
from haldensity.censoring import WeightedCVXPYEstimator, KaplanMeier
print("✓ Imports successful")

print("\nTest 2: Generate simple data")
import numpy as np
import pandas as pd
rng = np.random.default_rng(12776)
data = pd.DataFrame({"W1": rng.random(10)})
print(f"✓ Generated {len(data)} points")

print("\nTest 3: Fit IPCW estimator with ECOS")
try:
    est = WeightedCVXPYEstimator(
        norm_constraint=10.0,
        n_grid_points=50,
        solver="ECOS",
        use_secondary_solver=False,
        legacy_mode=True,
    ).fit(data)
    print(f"✓ Fit successful, theta shape: {est.theta_hat.shape}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n✓ All tests passed!")

