#!/usr/bin/env python3
"""
Fourier Series Analyzer from CSV

Reads CSV (time, value), computes Fourier series coefficients,
prints the Fourier series formula, and plots the reconstructed waveform.

Usage:
    python fourier_series_formula.py data.csv --n-harmonics 10
"""

import argparse
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import os

def export_c_function(coeffs, n_terms=None, func_name="f_series"):
    """Generate a C function string that evaluates the Fourier series."""
    a0, a, b, T = coeffs['a0'], coeffs['a'], coeffs['b'], coeffs['T']
    if n_terms is None:
        n_terms = len(a)-1

    lines = []
    lines.append("#include <math.h>")
    lines.append("")
    lines.append(f"double {func_name}(double t) {{")
    lines.append(f"    double result = {a0/2:.12f};  // a0/2")
    for n in range(1, n_terms+1):
        if abs(a[n]) > 1e-12:
            lines.append(f"    result += {a[n]:+.12f} * cos(2*M_PI*{n}*t/{T:.12f});")
        if abs(b[n]) > 1e-12:
            lines.append(f"    result += {b[n]:+.12f} * sin(2*M_PI*{n}*t/{T:.12f});")
    lines.append("    return result;")
    lines.append("}")
    return "\n".join(lines)


def load_csv(path, time_col=None, value_col=None):
    df = pd.read_csv(path)
    if time_col is None or value_col is None:
        time_col = df.columns[0] if time_col is None else time_col
        value_col = df.columns[1] if value_col is None else value_col
    t = df[time_col].values.astype(float)
    y = df[value_col].values.astype(float)
    return t, y

def resample_to_uniform(t, y, num=1024):
    dt = np.diff(np.sort(t))
    if len(dt) == 0:
        raise ValueError("Not enough time data")
    median_dt = np.median(dt)
    if np.allclose(dt, median_dt, rtol=1e-2, atol=1e-6):
        sort_idx = np.argsort(t)
        return t[sort_idx], y[sort_idx]
    t0, t1 = np.min(t), np.max(t)
    tu = np.linspace(t0, t1, num=num)
    f = interpolate.interp1d(t, y, kind='linear', fill_value='extrapolate')
    yu = f(tu)
    return tu, yu

def compute_fourier_series_coeffs(t, y, T=None, n_harmonics=10):
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    y = y[sort_idx]
    t0, t1 = t[0], t[-1]
    if T is None:
        T = t1 - t0
    tau = (t - t0) * (T / (t1 - t0)) if (t1 - t0) != T else (t - t0)

    def integral(fvals):
        return np.trapz(fvals, tau)

    a = np.zeros(n_harmonics+1)
    b = np.zeros(n_harmonics+1)
    a0 = (2.0 / T) * integral(y)
    a[0] = a0
    for n in range(1, n_harmonics+1):
        cosn = np.cos(2*np.pi*n*tau/T)
        sinn = np.sin(2*np.pi*n*tau/T)
        a[n] = (2.0 / T) * integral(y * cosn)
        b[n] = (2.0 / T) * integral(y * sinn)
    return {'a0': a0, 'a': a, 'b': b, 'T': T, 't0': t0}

def reconstruct_from_coeffs(coeffs, t_eval, n_terms=None):
    a0, a, b, T, t0 = coeffs['a0'], coeffs['a'], coeffs['b'], coeffs['T'], coeffs['t0']
    if n_terms is None:
        n_terms = len(a)-1
    tau = (t_eval - t0)
    y = a0/2.0 * np.ones_like(tau)
    for n in range(1, n_terms+1):
        y += a[n] * np.cos(2*np.pi*n*tau/T) + b[n] * np.sin(2*np.pi*n*tau/T)
    return y

def fourier_series_formula(coeffs, n_terms=None):
    """Return a string of Fourier series formula."""
    a0, a, b, T = coeffs['a0'], coeffs['a'], coeffs['b'], coeffs['T']
    if n_terms is None:
        n_terms = len(a)-1
    terms = [f"{a0/2:.3f}"]
    for n in range(1, n_terms+1):
        if abs(a[n]) > 1e-6:
            terms.append(f"{a[n]:+.3f} cos(2π*{n}t/{T:.3f})")
        if abs(b[n]) > 1e-6:
            terms.append(f"{b[n]:+.3f} sin(2π*{n}t/{T:.3f})")
    return "f(t) = " + " ".join(terms)

def plot_results(t, y, coeffs, n_terms=None, out_prefix=None, show=True):
    t_uni = np.linspace(np.min(t), np.max(t), 1000)
    y_recon = reconstruct_from_coeffs(coeffs, t_uni, n_terms=n_terms)
    formula = fourier_series_formula(coeffs, n_terms=n_terms)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(t, y, 'o', label="Original data", alpha=0.6)
    ax.plot(t_uni, y_recon, '-', label=f"Reconstructed (n={n_terms})")
    ax.set_title("Fourier Series Approximation")
    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    ax.grid(True)
    ax.legend()

    # Add formula as text
    ax.text(0.02, -0.15, formula, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace')

    if out_prefix:
        plt.savefig(out_prefix + "_fourier.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description="CSV -> Fourier series analyzer with formula output")
    p.add_argument("csv", help="Input CSV file (time, value)")
    p.add_argument("--time-col", help="Time column name (default: first column)")
    p.add_argument("--value-col", help="Value column name (default: second column)")
    p.add_argument("--n-harmonics", type=int, default=10, help="Number of harmonics (default=10)")
    args = p.parse_args()

    t, y = load_csv(args.csv, args.time_col, args.value_col)
    t_uni, y_uni = resample_to_uniform(t, y)
    coeffs = compute_fourier_series_coeffs(t_uni, y_uni, n_harmonics=args.n_harmonics)

    formula = fourier_series_formula(coeffs, n_terms=args.n_harmonics)
    print("Fourier series formula:")
    print(formula)

    print("\n\nEquivalent C function:")
    print(export_c_function(coeffs, n_terms=args.n_harmonics))

    out_prefix = os.path.splitext(os.path.basename(args.csv))[0]
    plot_results(t, y, coeffs, n_terms=args.n_harmonics, out_prefix=out_prefix, show=True)

if __name__ == "__main__":
    main()
