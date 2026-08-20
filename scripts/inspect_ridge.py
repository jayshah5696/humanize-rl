"""Inspect the canonical ridge pickle to decide TS port shape.

Prints everything needed to plan the JSON export and pick sparse vs dense.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import click
import numpy as np


@click.command()
@click.option(
    "--path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("models/track_a_10k/ridge.pkl"),
    show_default=True,
)
def main(path: Path) -> None:
    raw = pickle.load(path.open("rb"))
    obj = raw
    cls = obj.__class__
    print(f"== {path} ==")
    print(f"class: {cls.__module__}.{cls.__name__}")
    print(f"top-level attrs: {sorted(a for a in vars(obj).keys() if not a.startswith('_'))}")
    print(f"file size: {path.stat().st_size:,} bytes")
    print()

    vec = getattr(obj, "vectorizer", None)
    clf = getattr(obj, "classifier", None)
    regs = getattr(obj, "regressors", None)

    if vec is None or clf is None or regs is None:
        print("!! Missing expected attrs (vectorizer / classifier / regressors).")
        print("   Full attrs:", vars(obj).keys())
        sys.exit(1)

    # ---- TfidfVectorizer ----
    print("== TfidfVectorizer ==")
    print(f"class: {vec.__class__.__name__}")
    print(f"ngram_range: {vec.ngram_range}")
    print(f"max_features: {vec.max_features}")
    print(f"vocabulary size: {len(vec.vocabulary_):,}")
    print(f"lowercase: {vec.lowercase}")
    print(f"token_pattern: {vec.token_pattern!r}")
    print(f"analyzer: {vec.analyzer}")
    print(f"norm: {vec.norm}")
    print(f"sublinear_tf: {vec.sublinear_tf}")
    print(f"smooth_idf: {vec.smooth_idf}")
    print(f"use_idf: {vec.use_idf}")
    print(f"stop_words: {vec.stop_words}")
    print(f"strip_accents: {vec.strip_accents}")
    print(f"min_df: {vec.min_df}, max_df: {vec.max_df}")
    print(f"binary: {vec.binary}")
    print(f"dtype: {vec.dtype}")
    idf = np.asarray(vec.idf_)
    print(f"idf_ shape: {idf.shape}  dtype: {idf.dtype}")
    print(f"idf_ min/mean/max: {idf.min():.4f} / {idf.mean():.4f} / {idf.max():.4f}")
    # sample vocab to gauge term length
    sample_terms = list(vec.vocabulary_.keys())[:10]
    print(f"sample vocab: {sample_terms}")
    # ngram distribution
    by_n: dict[int, int] = {}
    for term in vec.vocabulary_:
        n = term.count(" ") + 1
        by_n[n] = by_n.get(n, 0) + 1
    print(f"vocab by ngram n: {dict(sorted(by_n.items()))}")
    avg_len = sum(len(t) for t in vec.vocabulary_) / len(vec.vocabulary_)
    max_len = max(len(t) for t in vec.vocabulary_)
    print(f"vocab term len avg/max: {avg_len:.1f} / {max_len}")
    print()

    # ---- LogisticRegression (binary head) ----
    print("== LogisticRegression (binary AI/human) ==")
    print(f"class: {clf.__class__.__name__}")
    coef = np.asarray(clf.coef_)
    intercept = np.asarray(clf.intercept_)
    print(f"coef_ shape: {coef.shape}  dtype: {coef.dtype}")
    print(f"intercept_: {intercept.tolist()}")
    print(f"classes_: {getattr(clf, 'classes_', None)}")
    print(f"coef abs min/mean/max: {np.abs(coef).min():.6f} / {np.abs(coef).mean():.6f} / {np.abs(coef).max():.6f}")
    for thr in [1e-6, 1e-4, 1e-3, 1e-2]:
        nnz = int((np.abs(coef) > thr).sum())
        print(f"  coef nnz > {thr:>.0e}: {nnz:>6,} / {coef.size:,} ({100 * nnz / coef.size:.1f}%)")
    print()

    # ---- 8 Ridge heads ----
    print("== 8 Ridge regressors (rubric dims) ==")
    print(f"len(regressors): {len(regs)}")
    print(f"first class: {regs[0].__class__.__name__}")
    all_coefs = np.stack([np.asarray(r.coef_) for r in regs], axis=0)
    all_intercepts = np.asarray([float(r.intercept_) for r in regs])
    print(f"stacked coef shape: {all_coefs.shape}  dtype: {all_coefs.dtype}")
    print(f"intercepts: {all_intercepts.tolist()}")
    print(f"|coef| min/mean/max: {np.abs(all_coefs).min():.6f} / {np.abs(all_coefs).mean():.6f} / {np.abs(all_coefs).max():.6f}")
    for thr in [1e-6, 1e-4, 1e-3, 1e-2]:
        nnz = int((np.abs(all_coefs) > thr).sum())
        print(f"  coef nnz > {thr:>.0e}: {nnz:>9,} / {all_coefs.size:,} ({100 * nnz / all_coefs.size:.1f}%)")
    print()

    # ---- Bundle size estimate ----
    print("== Bundle size estimate ==")
    vocab_bytes = sum(len(t) + 12 for t in vec.vocabulary_)  # JSON: "term":N,
    idf_bytes = idf.size * 8  # JSON float ~ 8 chars
    lr_bytes = coef.size * 8 + 8
    ridge_bytes = all_coefs.size * 8 + all_intercepts.size * 8
    total = vocab_bytes + idf_bytes + lr_bytes + ridge_bytes
    print(f"vocab JSON:      ~{vocab_bytes / 1024:>7.1f} KB")
    print(f"idf JSON:        ~{idf_bytes / 1024:>7.1f} KB")
    print(f"LR JSON:         ~{lr_bytes / 1024:>7.1f} KB")
    print(f"8x Ridge JSON:   ~{ridge_bytes / 1024:>7.1f} KB")
    print(f"TOTAL JSON est:  ~{total / 1024:>7.1f} KB  ({total / (1024 * 1024):.2f} MB)")
    print(f"  gzipped (~30%): ~{total * 0.30 / 1024:>5.1f} KB")
    print()

    # ---- Sparse projection at a workable threshold ----
    print("== Sparse-export projection (zero out |coef| < 1e-4) ==")
    lr_keep = int((np.abs(coef) > 1e-4).sum())
    ridge_keep = int((np.abs(all_coefs) > 1e-4).sum())
    sparse_lr_bytes = lr_keep * 16  # idx (~6) + value (~10) per nnz in JSON
    sparse_ridge_bytes = ridge_keep * 16
    sparse_total = vocab_bytes + idf_bytes + sparse_lr_bytes + sparse_ridge_bytes
    print(f"LR nnz kept:     {lr_keep:,} / {coef.size:,}")
    print(f"Ridge nnz kept:  {ridge_keep:,} / {all_coefs.size:,}")
    print(f"TOTAL (sparse):  ~{sparse_total / 1024:.1f} KB  ({sparse_total / (1024 * 1024):.2f} MB)")
    print()

    # ---- Smoke prediction ----
    print("== Smoke predictions on sample text ==")
    sample = "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix."
    features = vec.transform([sample])
    print(f"transform shape: {features.shape}  nnz: {features.nnz}")
    p_ai = float(clf.predict_proba(features)[0, 1])
    rubric = np.clip(
        np.stack([r.predict(features) for r in regs], axis=1)[0], 0.0, 1.0
    )
    print(f"P(AI):    {p_ai:.6f}")
    print(f"P(human): {1 - p_ai:.6f}")
    print(f"rubric (8): {[f'{v:.4f}' for v in rubric.tolist()]}")


if __name__ == "__main__":
    main()
