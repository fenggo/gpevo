"""Remove duplicate crystal structures from an ASE trajectory.

Ported from ``~/ct22/data22/dedup_traj.py`` and exposed as ``gpevo dedup``.

A structure is a duplicate of another only when BOTH hold:

  1. packing similarity  : pymatgen ``StructureMatcher`` matches the two cells
                           (rotation-, origin- and cell-setting-invariant, the
                           same criterion as COMPACK)
  2. energetic degeneracy: |E_i - E_j| < ``de_tol`` (default 2 meV)

Duplicates are grouped by complete-linkage clustering, so every pair inside a
reported cluster satisfies both criteria (single linkage is transitive and
over-merges: a~b and b~c does not imply a~c).

One representative is written per cluster. Selection order:

  1. ids listed in ``keep_id`` (e.g. structures discussed in a manuscript);
     if two requested ids turn out to be duplicates of each other, only the
     first survives -- intended behaviour, not a bug
  2. otherwise the lowest-energy member

Only structures with the same atom count AND chemical formula are compared, so
datasets mixing different unit-cell contents (Z = 1 / 2 / 4) work in one call.

Usage::

    gpevo dedup structures.traj
    gpevo dedup structures.traj -o out.traj -r report.csv
    gpevo dedup structures.traj --keep-id 322 563 606 742
    gpevo dedup structures.traj --dry-run
"""
from __future__ import annotations

import os

import numpy as np


def _energies(images):
    """(energies, n_missing).  Never raises on a frame without a calculator."""
    values = []
    missing = 0
    for atoms in images:
        try:
            values.append(float(atoms.get_potential_energy()))
            continue
        except Exception:
            pass
        info = getattr(atoms, "info", None) or {}
        raw = info.get("energy", info.get("free_energy"))
        if raw is None:
            values.append(float("nan"))
            missing += 1
        else:
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(float("nan"))
                missing += 1
    return np.asarray(values, dtype=float), missing


def _cliques(adjacency):
    """Deterministic complete-linkage partition of a small undirected graph.

    All maximal cliques are enumerated, then the partition is built greedily
    by repeatedly taking the largest remaining maximal clique (ties broken by
    the sorted index tuple), so the result does not depend on iteration order.
    Every returned cluster is a clique, therefore no pair inside it is merely
    transitively related through a third structure.
    """
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from((i, j) for i in adjacency for j in adjacency[i] if i < j)
    cliques = ([tuple(sorted(c)) for c in nx.find_cliques(graph)]
               if graph.number_of_edges() else [])
    cliques = [c for c in cliques if len(c) > 1]
    cliques.sort(key=lambda c: (-len(c), c))

    covered = set()
    clusters = []
    for clique in cliques:
        free = [i for i in clique if i not in covered]
        if len(free) < 2:
            continue
        clusters.append(sorted(free))
        covered |= set(free)
    return clusters


def dedup(traj, output=None, report=None, de_tol=0.002, ltol=0.1, stol=0.3,
          angle_tol=3.0, keep_id=None, dry_run=False, quiet=False):
    """Deduplicate *traj*; return a result dict and optionally write files.

    Parameters mirror the original ``dedup_traj.py``.  ``quiet=True`` suppresses
    the human-readable report but still writes ``output`` / ``report``.
    """
    from ase.io import read, write
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise RuntimeError("gpevo dedup requires pymatgen") from exc

    def say(*args):
        if not quiet:
            print(*args)

    try:
        images = list(read(traj, index=":"))
    except Exception as exc:
        raise RuntimeError(
            f"no structures could be read from {traj}: {exc}") from exc
    n_total = len(images)
    if n_total == 0:
        raise RuntimeError(f"no structures in {traj}")

    energies, n_missing = _energies(images)
    use_energy = n_missing == 0
    if not use_energy:
        say(f"warning: {n_missing} frame(s) have no energy; skipping the "
            "energy-degeneracy criterion (packing similarity only)")

    preferred = set(keep_id or ())
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol,
                               primitive_cell=True, scale=False,
                               attempt_supercell=False, allow_subset=False)
    adaptor = AseAtomsAdaptor()

    # Only compare structures with the same composition / atom count.
    by_size = {}
    for i, atoms in enumerate(images):
        by_size.setdefault((len(atoms), atoms.get_chemical_formula()), []).append(i)

    clusters = []
    for members in by_size.values():
        if len(members) < 2:
            continue

        # Candidate pairs: energy-degenerate only (cheap, and criterion 2
        # anyway).  A generous prefilter avoids building pymatgen structures
        # for pairs that cannot possibly match.
        dmax = de_tol * 6.0
        candidates = []
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                if not use_energy or abs(energies[i] - energies[j]) < dmax:
                    candidates.append((i, j))
        if not candidates:
            continue
        say(f"[{len(images[members[0]])} atoms] {len(candidates)} candidate "
            f"pairs (|dE| < {dmax * 1000:.0f} meV)")

        structures = {i: adaptor.get_structure(images[i]) for i in members}

        adjacency = {i: set() for i in members}
        edges = []
        for i, j in candidates:
            if use_energy and abs(energies[i] - energies[j]) >= de_tol:
                continue
            if matcher.fit(structures[i], structures[j]):
                rms = matcher.get_rms_dist(structures[i], structures[j])
                edges.append((i, j, float(rms[0]) if rms else float("nan")))
                adjacency[i].add(j)
                adjacency[j].add(i)
        say(f"[{len(images[members[0]])} atoms] {len(edges)} duplicate edges")
        clusters += _cliques(adjacency)

    # One representative per cluster.
    redundant = set()
    groups = []
    for cluster in clusters:
        chosen = [i for i in cluster if i in preferred]
        rep = chosen[0] if chosen else min(
            cluster, key=lambda i: (energies[i] if use_energy else i))
        groups.append((sorted(cluster), rep))
        redundant.update(i for i in cluster if i != rep)

    keep = [i for i in range(n_total) if i not in redundant]

    say("\ninput structures            : %d" % n_total)
    say("duplicate clusters          : %d" % len(groups))
    say("redundant structures removed: %d" % len(redundant))
    say("distinct structures kept    : %d" % len(keep))
    if groups:
        sizes = sorted((len(c) for c, _ in groups), reverse=True)
        say("cluster sizes               : %s" %
            {s: sizes.count(s) for s in sorted(set(sizes), reverse=True)})
        say("\n%-6s %-6s %-7s %s" % ("kept", "size", "dE(meV)", "cluster members"))
        for cluster, rep in sorted(groups, key=lambda g: -len(g[0])):
            span = (1000 * (np.nanmax(energies[cluster]) - np.nanmin(energies[cluster]))
                    if use_energy else float("nan"))
            say("%-6d %-6d %-7.2f %s" % (rep, len(cluster), span, cluster))

    if report:
        with open(report, "w", encoding="utf-8") as fh:
            fh.write("duplicate_id,representative_id,cluster_size,cluster_ids\n")
            for cluster, rep in sorted(groups, key=lambda g: -len(g[0])):
                for i in cluster:
                    if i != rep:
                        fh.write("%d,%d,%d,%s\n" %
                                 (i, rep, len(cluster),
                                  " ".join(str(x) for x in cluster)))
        say("\nwrote duplicate report: %s" % report)

    out = output or (os.path.splitext(traj)[0] + "_dedup.traj")
    if dry_run:
        say("\ndry run: no trajectory written")
    else:
        write(out, [images[i] for i in keep])
        say("wrote %s (%d structures)" % (out, len(keep)))

    return {
        "n_total": n_total,
        "groups": groups,
        "redundant": sorted(redundant),
        "keep": keep,
        "output": None if dry_run else out,
        "report": report,
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove duplicate crystal structures from an ASE trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("traj", help="input trajectory (ASE .traj / POSCAR / CIF)")
    parser.add_argument("-o", "--output", default=None,
                        help="output trajectory; default: <traj>_dedup.traj")
    parser.add_argument("-r", "--report", default=None,
                        help="write the duplicate list to this CSV file")
    parser.add_argument("--de-tol", type=float, default=0.002,
                        help="energy degeneracy tolerance in eV")
    parser.add_argument("--ltol", type=float, default=0.1,
                        help="StructureMatcher fractional lattice tolerance")
    parser.add_argument("--stol", type=float, default=0.3,
                        help="StructureMatcher site tolerance")
    parser.add_argument("--angle-tol", type=float, default=3.0,
                        help="StructureMatcher interaxial angle tolerance (deg)")
    parser.add_argument("--keep-id", type=int, nargs="*", default=None,
                        help="indices (0-based) to prefer as cluster representatives")
    parser.add_argument("--dry-run", action="store_true",
                        help="report duplicates but write no output trajectory")
    parser.add_argument("--quiet", action="store_true", help="suppress the report")
    args = parser.parse_args(argv)
    dedup(traj=args.traj, output=args.output, report=args.report,
          de_tol=args.de_tol, ltol=args.ltol, stol=args.stol,
          angle_tol=args.angle_tol, keep_id=args.keep_id,
          dry_run=args.dry_run, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
