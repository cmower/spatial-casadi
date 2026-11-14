"""Run with uv run python benchmarks/run.py; timings are informational."""

import argparse
import importlib.util
import json
import platform
import tempfile
import time
from pathlib import Path

import casadi as cs
import numpy as np

from spatial_casadi import Rotation


def measure(name, kind, length, rotation=Rotation):
    started = time.perf_counter()
    x = kind.sym("x", 3, length)
    r = rotation.identity()
    for i in range(length):
        r = r * rotation.from_euler("xyz", x[:, i])
    value = cs.vec(r.as_matrix())
    f = cs.Function("value", [x], [value])
    jac = cs.Function("jacobian", [x], [cs.jacobian(value, x)])
    hess = cs.Function(
        "hessian", [x], [cs.hessian(cs.sumsqr(value - cs.vec(cs.DM.eye(3))), x)[0]]
    )
    build_ms = 1000 * (time.perf_counter() - started)
    sample = np.full((3, length), 0.1)
    timings = {}
    for label, function in [("value", f), ("jacobian", jac), ("hessian", hess)]:
        function(sample)
        start = time.perf_counter()
        for _ in range(200):
            function(sample)
        timings[label + "_us"] = round(1e6 * (time.perf_counter() - start) / 200, 2)
    with tempfile.TemporaryDirectory() as directory:
        generator = cs.CodeGenerator("function.c")
        for function in [f, jac, hess]:
            generator.add(function)
        filename = generator.generate(directory + "/")
        code_bytes = Path(filename).stat().st_size
    return {
        "name": name,
        "kind": kind.__name__,
        "length": length,
        "build_ms": round(build_ms, 2),
        "nodes": [f.n_nodes(), jac.n_nodes(), hess.n_nodes()],
        "code_bytes": code_bytes,
        **timings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-source", type=Path, help="Optional 1.x spatial.py for comparison"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    implementations = [("current", Rotation)]
    if args.legacy_source:
        spec = importlib.util.spec_from_file_location(
            "legacy_spatial", args.legacy_source
        )
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        implementations.append(("legacy", legacy.Rotation))
    report = {
        "python": platform.python_version(),
        "casadi": cs.__version__,
        "platform": platform.platform(),
        "results": [
            measure(name, kind, length, rotation)
            for name, rotation in implementations
            for kind in [cs.SX, cs.MX]
            for length in [1, 10, 50]
        ],
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
