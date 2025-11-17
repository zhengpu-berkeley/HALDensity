# HALDensity Cleanup Checklist

Use this guide after you are satisfied with the new censored-data implementation and want to trim the repo back down to production-only assets.

## 1. Remove Temporary/Debug Scripts
Delete the following files from the project root unless you plan to keep them for demos:
- `test_censored_complete.py`
- `censor_data_comp_tests.py`
- `test_ipcw_only.py`
- `test_em_only.py`
- `test_minimal.py`
- `test_ultra_minimal.py`
- `debug_compare_mstep.py`
- `censored_data_workflow.md` (if you no longer need the scratch notes)

## 2. Notebook + Archived Artifacts
If the historical reference materials are no longer required (they live safely in Git history), remove the PyTorch driver script in the repo root together with the two censored-data notebooks:
- `1O_CV_EM_IPCW_HAL_MLE.ipynb`
- `ZO_CV_EM_IPCW_HAL_MLE.ipynb`

Keep these files only if you plan to continue cross-checking against the original experiments; otherwise remove them to prevent drift.

## 3. Dependency Hygiene
- Ensure `pyproject.toml` / `requirements` include `cvxpy`, `lifelines`, etc., if you run the tests in CI. Remove any ad‑hoc dependencies that were introduced only for debugging.
- Run `uv pip check` (or your preferred tool) once more after the cleanup to confirm there are no missing wheels.

## 4. Testing Before Commit
After removing the files, re-run the two official tests:
1. `uv run test_censored_complete.py` (if you keep it) or integrate its assertions into your formal test suite.
2. `gtimeout 60s uv run censor_data_comp_tests.py`

If you delete those scripts, make sure the remaining test harness (e.g. `pytest` or `tox`) still covers the censored-data pipeline or port the assertions into your main suites.

## 5. Documentation
- Keep `HALDENSITY_FIX_HISTORY.md` (this file’s companion) if you want the historical context committed; otherwise, archive it off-repo.
- Update the main `README.md` to point to production documentation (`src/haldensity/censoring/README.md`) and remove references to temporary scripts.

## 6. Git Hygiene
Finally, check for untracked files with `git status -sb` and make sure no temporary artifacts (`*.log`, saved plots, etc.) remain. Commit only the files that belong to the final implementation.

