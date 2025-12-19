#!/usr/bin/env python3
import os
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from causallearn.utils.cit import fisherz
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.PermutationBased.GRaSP import grasp


# --------------------------- Config ---------------------------

REGIMES = {
    "0%":  [0, 1, 2],
    "7%":  [3, 4, 5, 6, 7],
    "10%": [8, 9],
    "13%": [10, 11],
    "20%": [12, 13],
}

DEFAULTS = dict(
    n_seeds=10,
    n_boot=100,
    alpha=0.1,
    depth=-1,
    seed_boot_base=12345,
)

OUTDIR = "../results"
os.makedirs(OUTDIR, exist_ok=True)


# --------------------------- Data prep ---------------------------

def entry_to_df(entry):
    df = pd.DataFrame(dict(
        Alkali_Cations = np.ravel(entry['I1'] + entry['I2'] + entry['I3'] + entry['I4']),
        Metal_Cations  = np.ravel(entry['I5']),
        Lattice_a      = np.sqrt(entry['a'][0]**2  + entry['a'][1]**2),
        Lattice_b      = np.sqrt(entry['b'][0]**2  + entry['b'][1]**2),
        ab             = np.sqrt(entry['ab'][0]**2 + entry['ab'][1]**2),
        Angle          = np.ravel(entry['alpha']),
        Volume         = np.ravel(entry['Vol']),
        Px             = np.ravel(entry['Pxy'][0]),
        Py             = np.ravel(entry['Pxy'][1]),
    ))
    return df


def fill_nan_with_local_mean(df):
    df = df.copy()
    up = df.shift(1)
    dn = df.shift(-1)
    local_mean = (up + dn) / 2.0
    return df.where(~df.isna(), local_mean)


def clean_df(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = fill_nan_with_local_mean(df)
    df = df.ffill().bfill()

    if not np.isfinite(df.values).all():
        bad = np.where(~np.isfinite(df.values))
        raise ValueError(f"Non-finite values remain after cleaning at indices: {bad[:2]}")
    return df


def build_regime_df(SmBFOdata, indices):
    dfs = []
    for i in indices:
        df = entry_to_df(SmBFOdata[i])
        df = clean_df(df)
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)


def standardize_within_regime(df):
    X = df.values.astype(np.float64)
    X = StandardScaler().fit_transform(X)
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values after standardization.")
    return X, df.columns.tolist()


# --------------------------- Edge extractors ---------------------------

def graph_directed_edges(graph, var_names):
    if not hasattr(graph, "get_graph_edges"):
        raise TypeError(f"Expected causallearn graph with get_graph_edges(); got {type(graph)}")

    nodes = graph.get_nodes()
    node_to_name = {nodes[i]: var_names[i] for i in range(len(nodes))}

    edges = set()
    for e in graph.get_graph_edges():
        u = node_to_name[e.get_node1()]
        v = node_to_name[e.get_node2()]
        ep1 = e.get_endpoint1().name
        ep2 = e.get_endpoint2().name

        if ep1 == "TAIL" and ep2 == "ARROW":
            edges.add((u, v, "-->"))
        elif ep1 == "ARROW" and ep2 == "TAIL":
            edges.add((v, u, "-->"))
    return edges


def graph_skeleton_edges(graph, var_names):
    if not hasattr(graph, "get_graph_edges"):
        raise TypeError(f"Expected causallearn graph with get_graph_edges(); got {type(graph)}")

    nodes = graph.get_nodes()
    node_to_name = {nodes[i]: var_names[i] for i in range(len(nodes))}

    skel = set()
    for e in graph.get_graph_edges():
        u = node_to_name[e.get_node1()]
        v = node_to_name[e.get_node2()]
        if u == v:
            continue
        a, b = sorted([u, v])
        skel.add((a, b, "---"))
    return skel


def grasp_result_to_edges(result, var_names):
    if isinstance(result, np.ndarray) and result.ndim == 2:
        edges = set()
        p = result.shape[0]
        for i in range(p):
            for j in range(p):
                if abs(result[i, j]) > 1e-12:
                    edges.add((var_names[i], var_names[j], "-->"))
        return edges

    if hasattr(result, "get_graph_edges"):
        return graph_directed_edges(result, var_names)

    if hasattr(result, "adjacency_matrix_"):
        return grasp_result_to_edges(result.adjacency_matrix_, var_names)

    raise TypeError(f"Unsupported GRaSP return type: {type(result)}")


# --------------------------- Consensus runner ---------------------------

def consensus_seed_bootstrap(
    X,
    var_names,
    method="grasp",      # 'grasp' | 'pc' | 'fci'
    n_seeds=5,
    n_boot=100,
    alpha=0.1,
    depth=-1,
    seed_boot_base=12345,
    mode="directed",     # for fci: 'directed'|'skeleton'
):
    seeds = list(range(n_seeds))
    counts_per_seed = {s: defaultdict(int) for s in seeds}

    for s in tqdm(seeds, desc=f"{method.upper()} seeds", leave=True):
        rng = np.random.default_rng(seed_boot_base + 100000 * s)

        for _ in tqdm(range(n_boot), desc=f"boot(seed={s})", leave=False):
            idx = rng.choice(X.shape[0], size=X.shape[0], replace=True)
            Xb = X[idx]

            if method == "grasp":
                try:
                    res = grasp(Xb, random_state=s)
                except TypeError:
                    np.random.seed(s)
                    res = grasp(Xb)
                edges = grasp_result_to_edges(res, var_names)

            elif method == "pc":
                cg = pc(
                    Xb,
                    alpha=alpha,
                    indep_test=fisherz,
                    stable=True,
                    verbose=False,
                    show_progress=False
                )
                edges = graph_directed_edges(cg.G, var_names)

            elif method == "fci":
                cg, _ = fci(
                    Xb,
                    independence_test_method=fisherz,
                    alpha=alpha,
                    depth=depth,
                    verbose=False
                )
                edges = graph_skeleton_edges(cg, var_names) if mode == "skeleton" else graph_directed_edges(cg, var_names)

            else:
                raise ValueError(f"Unknown method={method}")

            for e in edges:
                counts_per_seed[s][e] += 1

    # aggregate
    all_edges = set()
    for s in seeds:
        all_edges |= set(counts_per_seed[s].keys())

    rows = []
    for e in sorted(all_edges):
        freqs = [counts_per_seed[s].get(e, 0) / n_boot for s in seeds]
        mean_f = float(np.mean(freqs))
        std_f  = float(np.std(freqs, ddof=0))
        min_f  = float(np.min(freqs))
        max_f  = float(np.max(freqs))

        u, v, etype = e
        rows.append({
            "u": u, "v": v, "etype": etype,
            "mean_freq": mean_f,
            "std_across_seeds": std_f,
            "min_seed_freq": min_f,
            "max_seed_freq": max_f,
            "conf_min": min_f,
            "conf_mean_minus_std": max(0.0, mean_f - std_f),
            "n_seeds": n_seeds,
            "n_boot_per_seed": n_boot,
            "alpha": (alpha if method in ["pc", "fci"] else np.nan),
            "method": method,
            "mode": mode
        })

    df = pd.DataFrame(rows).sort_values(
        by=["conf_min", "conf_mean_minus_std", "mean_freq"],
        ascending=False
    ).reset_index(drop=True)

    return df


# --------------------------- Gold table ---------------------------

def make_gold_table(results, grasp_thr=0.60, fci_skel_thr=0.60):
    def edge_key(u, v, etype): return (u, v, etype)

    all_edges = set()
    for reg, reg_out in results.items():
        if "grasp" in reg_out:
            d = reg_out["grasp"]
            for _, r in d.iterrows():
                if float(r["conf_min"]) >= grasp_thr:
                    all_edges.add(edge_key(r["u"], r["v"], r["etype"]))

        if "fci_skeleton" in reg_out:
            d = reg_out["fci_skeleton"]
            for _, r in d.iterrows():
                if float(r["conf_min"]) >= fci_skel_thr:
                    all_edges.add(edge_key(r["u"], r["v"], r["etype"]))

    all_edges = sorted(list(all_edges))

    rows = []
    for (u, v, etype) in all_edges:
        row = {"edge": f"{u} {etype} {v}", "u": u, "v": v, "etype": etype}
        for reg, reg_out in results.items():
            present = False

            # GRaSP directed presence
            if not present and "grasp" in reg_out:
                d = reg_out["grasp"]
                hit = d[(d["u"] == u) & (d["v"] == v) & (d["etype"] == etype)]
                if len(hit) and float(hit["conf_min"].iloc[0]) >= grasp_thr:
                    present = True

            # FCI skeleton presence (latent-robust coupling)
            if not present and "fci_skeleton" in reg_out:
                a, b = sorted([u, v])
                d = reg_out["fci_skeleton"]
                hit = d[(d["u"] == a) & (d["v"] == b) & (d["etype"] == "---")]
                if len(hit) and float(hit["conf_min"].iloc[0]) >= fci_skel_thr:
                    present = True

            row[reg] = "✓" if present else "✗"
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------- Run + save incrementally ---------------------------

def safe_to_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_all_regimes_and_save(
    SmBFOdata,
    outdir=OUTDIR,
    n_seeds=5,
    n_boot=100,
    alpha=0.1,
    depth=-1,
    seed_boot_base=12345,
):
    results = {}
    meta = {
        "n_seeds": n_seeds,
        "n_boot": n_boot,
        "alpha": alpha,
        "depth": depth,
        "seed_boot_base": seed_boot_base,
        "regimes": REGIMES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(outdir, "run_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for reg_name, idxs in REGIMES.items():
        print(f"\n=== Regime {reg_name} | snapshots {idxs} ===")
        df_reg = build_regime_df(SmBFOdata, idxs)
        X_reg, var_names = standardize_within_regime(df_reg)

        # Save the regime dataframe used (optional but very reproducible)
        safe_to_csv(df_reg, os.path.join(outdir, f"regime_{reg_name}_data.csv"))

        reg_out = {}

        # GRaSP
        df_grasp = consensus_seed_bootstrap(
            X_reg, var_names, method="grasp",
            n_seeds=n_seeds, n_boot=n_boot,
            alpha=alpha, depth=depth,
            seed_boot_base=seed_boot_base,
            mode="directed"
        )
        path = os.path.join(outdir, f"regime_{reg_name}_grasp_directed.csv")
        safe_to_csv(df_grasp, path)
        reg_out["grasp"] = df_grasp

        # PC (directed)
        df_pc = consensus_seed_bootstrap(
            X_reg, var_names, method="pc",
            n_seeds=n_seeds, n_boot=n_boot,
            alpha=alpha, depth=depth,
            seed_boot_base=seed_boot_base,
            mode="directed"
        )
        path = os.path.join(outdir, f"regime_{reg_name}_pc_directed.csv")
        safe_to_csv(df_pc, path)
        reg_out["pc_directed"] = df_pc

        # FCI directed
        df_fci_dir = consensus_seed_bootstrap(
            X_reg, var_names, method="fci",
            n_seeds=n_seeds, n_boot=n_boot,
            alpha=alpha, depth=depth,
            seed_boot_base=seed_boot_base,
            mode="directed"
        )
        path = os.path.join(outdir, f"regime_{reg_name}_fci_directed.csv")
        safe_to_csv(df_fci_dir, path)
        reg_out["fci_directed"] = df_fci_dir

        # FCI skeleton
        df_fci_skel = consensus_seed_bootstrap(
            X_reg, var_names, method="fci",
            n_seeds=n_seeds, n_boot=n_boot,
            alpha=alpha, depth=depth,
            seed_boot_base=seed_boot_base,
            mode="skeleton"
        )
        path = os.path.join(outdir, f"regime_{reg_name}_fci_skeleton.csv")
        safe_to_csv(df_fci_skel, path)
        reg_out["fci_skeleton"] = df_fci_skel

        # Save a small summary per regime
        summary = {
            "regime": reg_name,
            "snapshots": idxs,
            "n_rows": int(df_reg.shape[0]),
            "n_vars": int(df_reg.shape[1]),
            "var_names": var_names,
            "files": {
                "grasp_directed": f"regime_{reg_name}_grasp_directed.csv",
                "pc_directed": f"regime_{reg_name}_pc_directed.csv",
                "fci_directed": f"regime_{reg_name}_fci_directed.csv",
                "fci_skeleton": f"regime_{reg_name}_fci_skeleton.csv",
                "data": f"regime_{reg_name}_data.csv",
            }
        }
        with open(os.path.join(outdir, f"regime_{reg_name}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        results[reg_name] = reg_out

    # Final gold table
    gold = make_gold_table(results, grasp_thr=0.60, fci_skel_thr=0.60)
    safe_to_csv(gold, os.path.join(outdir, "gold_table.csv"))

    return results, gold


def main():
    # Update this path if not on Colab
    npz_path = "./SmBFOdata.npz"

    data = np.load(npz_path, allow_pickle=True)
    SmBFOdata = data["SmBFOdata"]

    print(SmBFOdata.dtype)
    print(SmBFOdata.shape)
    print(SmBFOdata[0].keys())

    results, gold = run_all_regimes_and_save(
        SmBFOdata,
        outdir=OUTDIR,
        **DEFAULTS
    )

    print("\nSaved results to:", OUTDIR)
    print("Gold table head:\n", gold.head(20))


if __name__ == "__main__":
    main()
