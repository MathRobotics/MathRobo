#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

import mathrobo as mr


@dataclass
class BenchCase:
    name: str
    fn: Callable[[], object]
    number: int


def bench_us(case: BenchCase, repeat: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        for _ in range(case.number):
            case.fn()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(case.number):
            case.fn()
        dt = (time.perf_counter() - start) * 1e6 / case.number
        samples.append(dt)

    return float(np.mean(samples)), float(np.std(samples))


def _print_table(title: str, rows: list[tuple[str, float, float, int]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'name':28s} {'mean[us]':>12s} {'std[us]':>10s} {'loops':>8s}")
    for name, mean_us, std_us, loops in rows:
        print(f"{name:28s} {mean_us:12.3f} {std_us:10.3f} {loops:8d}")


def _make_vecs(order: int, dof: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if order <= 1:
        return np.zeros((0, dof), dtype=np.float64)
    return rng.standard_normal((order - 1, dof), dtype=np.float64)


def run_for_group(
    group_cls,
    group_name: str,
    order: int,
    repeat: int,
    warmup: int,
    seed: int,
) -> None:
    vecs = _make_vecs(order, group_cls.dof(), seed=seed)
    elem = group_cls.eye()

    c = mr.CMTM[group_cls](elem, vecs)

    hot_cases = [
        BenchCase("ctor", lambda: mr.CMTM[group_cls](elem, vecs), 5000),
        BenchCase("mat", c.mat, 5000),
        BenchCase("mat_adj", c.mat_adj, 5000),
        BenchCase("mat_inv", c.mat_inv, 5000),
        BenchCase("mat_inv_adj", c.mat_inv_adj, 5000),
        BenchCase("tangent_mat", c.tangent_mat, 3000),
        BenchCase("tangent_mat_cm", c.tangent_mat_cm, 3000),
        BenchCase("matmul(c@c)", lambda: c @ c, 2000),
    ]

    # Warm cache once so hot-path timings are stable.
    c.mat()
    c.mat_adj()
    c.mat_inv()
    c.mat_inv_adj()
    c.tangent_mat()
    c.tangent_mat_cm()

    hot_rows = []
    for case in hot_cases:
        mean_us, std_us = bench_us(case, repeat=repeat, warmup=warmup)
        hot_rows.append((case.name, mean_us, std_us, case.number))

    cold_cases = [
        BenchCase("new+mat", lambda: mr.CMTM[group_cls](elem, vecs).mat(), 3000),
        BenchCase("new+mat_adj", lambda: mr.CMTM[group_cls](elem, vecs).mat_adj(), 3000),
        BenchCase("new+mat_inv", lambda: mr.CMTM[group_cls](elem, vecs).mat_inv(), 3000),
        BenchCase("new+tangent_mat", lambda: mr.CMTM[group_cls](elem, vecs).tangent_mat(), 1500),
        BenchCase(
            "new+tangent_mat_cm",
            lambda: mr.CMTM[group_cls](elem, vecs).tangent_mat_cm(),
            1500,
        ),
    ]

    cold_rows = []
    for case in cold_cases:
        mean_us, std_us = bench_us(case, repeat=repeat, warmup=warmup)
        cold_rows.append((case.name, mean_us, std_us, case.number))

    print(f"\n=== {group_name} / order={order} ===")
    _print_table("hot path (same CMTM instance)", hot_rows)
    _print_table("cold path (new CMTM per call)", cold_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CMTM calculation times (SO3/SE3) for selected orders."
        )
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[5],
        help="Target CMTM orders (default: 5). Example: --orders 3 5 8",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=7,
        help="Number of repeated measurements per case (default: 7).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup rounds before measurement (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for input vectors (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Benchmark: CMTM timing comparison")
    print("Note: set OPENBLAS_NUM_THREADS=1 for more stable small-matrix timings.")
    for order in args.orders:
        run_for_group(
            mr.SO3,
            "SO3",
            order=order,
            repeat=args.repeat,
            warmup=args.warmup,
            seed=args.seed + order * 10 + 1,
        )
        run_for_group(
            mr.SE3,
            "SE3",
            order=order,
            repeat=args.repeat,
            warmup=args.warmup,
            seed=args.seed + order * 10 + 2,
        )


if __name__ == "__main__":
    main()
