"""Phase 9: the one-fold reproduction check runs exactly the folds of the full protocol."""
import numpy as np

from experiments.protocol import outer_splits


def test_only_filter_keeps_identical_folds():
    rng = np.random.default_rng(0)
    strata = rng.integers(0, 4, 500)
    full = {(r, f): (tr, te) for r, f, _, tr, te in outer_splits(strata, 2, 5, [0, 1])}
    sub = list(outer_splits(strata, 2, 5, [0, 1], only=[[0, 0], [1, 3]]))
    assert [(r, f) for r, f, *_ in sub] == [(0, 0), (1, 3)]
    for r, f, seed, tr, te in sub:
        assert seed == [0, 1][r]
        assert np.array_equal(tr, full[(r, f)][0]) and np.array_equal(te, full[(r, f)][1])
    assert len(list(outer_splits(strata, 2, 5, [0, 1], only=None))) == 10


def test_compare_treats_missing_text_as_equal_and_finds_real_differences(tmp_path, monkeypatch):
    import importlib.util
    import pandas as pd
    spec = importlib.util.spec_from_file_location("check_reproduction", "scripts/check_reproduction.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    monkeypatch.setattr(cr, "REPO_ROOT", tmp_path)
    old = pd.DataFrame({"repeat": [0, 0, 1], "fold": [0, 0, 0], "mitigated": ["race", None, "sex"],
                        "acc": [0.5, np.nan, 0.7], "seconds": [1.0, 2.0, 3.0]})
    old.to_csv(tmp_path / "old.csv", index=False)
    new = old[old.repeat == 0].assign(seconds=[9.0, 9.0])          # time columns are not compared
    new.to_csv(tmp_path / "new.csv", index=False)
    row = cr.compare(tmp_path / "new.csv", tmp_path / "old.csv")
    assert row["identical"] and row["rows_compared"] == 2 and row["columns_skipped"] == "seconds"
    new.assign(mitigated=["race", "sex"]).to_csv(tmp_path / "new.csv", index=False)
    row = cr.compare(tmp_path / "new.csv", tmp_path / "old.csv")
    assert not row["identical"] and "mitigated" in row["note"]
    outside = tmp_path.parent / "elsewhere.csv"                     # scratch files outside the repository
    old.to_csv(outside, index=False)
    assert cr.compare(tmp_path / "new.csv", outside)["file"] == str(outside.resolve())
