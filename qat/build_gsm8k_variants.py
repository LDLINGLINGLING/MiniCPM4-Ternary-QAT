#!/usr/bin/env python3
"""Deterministic, CPU-only GSM8K paired-variant builder. Run with python -I.

The first --freeze-only invocation commits the sampled IDs before questions are
reviewed. Curated specifications are independently assistant-reviewed; they are
not human review or model-performance filtering. No models or services are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
from fractions import Fraction
import sys

NUMBER = re.compile(r"-?(?:\d+(?:\.\d+)?|\d+/[1-9]\d*)\Z")
OPERATIONS = {"add", "sub", "mul", "div"}

SEED = 20261109
COUNT = 100
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def freeze(source, output):
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = random.Random(SEED).sample(range(len(rows)), COUNT)
    frozen = {
        "schema_version": 1, "seed": SEED, "sample_count": COUNT,
        "source_split": "test", "source_sha256": sha256(source),
        "source_count": len(rows), "id_convention": "zero-based JSONL row index",
        "sampling": "random.Random(seed).sample(range(source_count), sample_count); ordered, no replacements",
        "source_ids": ids,
        "selection_policy": "Frozen before variant drafting; no model per-question results consulted; exclusions never replaced",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "frozen_ids.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != frozen:
            raise ValueError("Frozen IDs/source metadata differ; refusing to overwrite")
    else:
        dump(path, frozen)
    return rows, frozen


def number(text):
    if not isinstance(text, str) or not NUMBER.fullmatch(text):
        raise ValueError(f"Not a literal rational number: {text!r}")
    return Fraction(text)


def calculate(expression, env):
    """Interpret a closed JSON arithmetic tree; never eval/exec source text."""
    if isinstance(expression, str):
        return env[expression] if expression in env else number(expression)
    if not isinstance(expression, list) or len(expression) < 2 or expression[0] not in OPERATIONS:
        raise ValueError(f"Invalid arithmetic node: {expression!r}")
    operation, *children = expression
    args = [calculate(child, env) for child in children]
    if operation == "add":
        return sum(args, Fraction(0))
    if operation == "mul":
        result = Fraction(1)
        for value in args:
            result *= value
        return result
    if operation == "sub":
        if len(args) < 2:
            raise ValueError("Subtraction requires at least two operands")
        return args[0] - sum(args[1:], Fraction(0))
    if len(args) != 2 or args[1] == 0:
        raise ValueError("Division requires two operands and a nonzero divisor")
    return args[0] / args[1]


def render(expression, env):
    if isinstance(expression, str):
        return f"{expression}[{env[expression]}]" if expression in env else expression
    symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
    return "(" + f" {symbols[expression[0]]} ".join(render(child, env) for child in expression[1:]) + ")"


def enforce(value, constraint):
    if value < 0:
        raise ValueError(f"Negative physical count/mass/money: {value}")
    if constraint == "integer" and value.denominator != 1:
        raise ValueError(f"Nonintegral count: {value}")
    if constraint == "money" and (value * 100).denominator != 1:
        raise ValueError(f"Not cent-exact: {value}")
    if constraint == "positive" and value <= 0:
        raise ValueError(f"Nonpositive magnitude: {value}")
    if constraint not in {"integer", "money", "positive"}:
        raise ValueError(f"Unknown constraint: {constraint}")


def solve(spec, version):
    env = {name: number(values[version]) for name, values in spec["values"].items()}
    if any(value <= 0 for value in env.values()):
        raise ValueError("All curated input parameters must be positive")
    trace = []
    for name, expression, unit, constraint, explanation in spec["steps"]:
        if name in env:
            raise ValueError(f"Duplicate solver variable: {name}")
        value = calculate(expression, env)
        enforce(value, constraint)
        trace.append({"name": name, "expression": expression,
                      "substitution": render(expression, env), "value": str(value),
                      "unit": unit, "constraint": constraint, "explanation": explanation})
        env[name] = value
    if not trace:
        raise ValueError("Solver has no steps")
    for check in spec.get("checks", []):
        kind, *operands = check
        values = [calculate(item, env) for item in operands]
        valid = (kind == "eq" and len(values) == 2 and values[0] == values[1]
                 or kind == "gt" and len(values) == 2 and values[0] > values[1]
                 or kind == "integer" and len(values) == 1 and values[0].denominator == 1)
        if not valid:
            raise ValueError(f"Relational constraint failed: {check!r}")
    return env[trace[-1]["name"]], trace


def answer_text(value):
    """Finite decimal or integer without floating-point rounding."""
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError(f"Answer is not a terminating decimal: {value}")
    places = max(twos, fives)
    scaled = abs(value.numerator) * (10 ** places // value.denominator)
    digits = str(scaled).zfill(places + 1)
    return ("-" if value < 0 else "") + digits[:-places] + "." + digits[-places:]


def replace_exact(question, replacements):
    result = question
    for old, new in replacements:
        if not old or old == new or old not in result:
            raise ValueError(f"Missing/empty/identity replacement: {old!r}")
        result = result.replace(old, new)
    if result == question:
        raise ValueError("Variant must differ from original")
    return result


def numeric_signature(question):
    return re.findall(r"\d+(?:[.,]\d+)*", question)


def reference_final(answer):
    if answer.count("####") != 1:
        raise ValueError("Expected exactly one GSM8K answer marker")
    return number(answer.split("####", 1)[1].strip().replace(",", ""))


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def build(source, output, specs_path, rows, frozen):
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    by_id = {spec["source_id"]: spec for spec in specs}
    if len(by_id) != len(specs) or set(by_id) != set(frozen["source_ids"]):
        raise ValueError("Reviews must cover every frozen source ID exactly once")
    datasets = {"original": [], "equivalent": [], "numeric": []}
    exclusions = []
    audit = []
    for source_id in frozen["source_ids"]:
        source_row, spec = rows[source_id], by_id[source_id]
        pair_id = f"gsm8k-test-{source_id:04d}"
        question, reference_answer = source_row["question"], source_row["answer"]
        original_value = reference_final(reference_answer)
        equivalent_question = replace_exact(question, spec["equivalent"])
        if numeric_signature(question) != numeric_signature(equivalent_question):
            raise ValueError(f"Equivalent rewrite changed digits: {source_id}")
        common = {"pair_id": pair_id, "source_id": source_id,
                  "original_answer": answer_text(original_value),
                  "review_status": "assistant-reviewed"}
        common_solver = {"method": "reference-final-only", "review_status": "assistant-reviewed",
                         "equivalence_review": "Only listed local wording/entity substitutions; mathematical relationships, units and quantities preserved. Source ambiguities are not repaired.",
                         "equivalent_replacements": spec["equivalent"]}
        if "exclusion" in spec:
            if "numeric" in spec or "steps" in spec:
                raise ValueError("Excluded source must not have a numeric variant")
            exclusion = {"pair_id": pair_id, "source_id": source_id,
                         "type": "numeric", "reason": spec["exclusion"],
                         "review_status": "assistant-reviewed", "replacement_sampled": False}
            exclusions.append(exclusion)
            common_solver["numeric_exclusion"] = spec["exclusion"]
            audit.append({**common, "numeric_status": "excluded", "reason": spec["exclusion"]})
        else:
            original_solved, original_trace = solve(spec, 0)
            numeric_value, numeric_trace = solve(spec, 1)
            if original_solved != original_value:
                raise ValueError(f"Independently solved original disagrees with reference for {source_id}: {original_solved} != {original_value}")
            if numeric_value == original_value:
                raise ValueError(f"Numeric answer did not change for {source_id}")
            numeric_question = replace_exact(question, spec["numeric"])
            solver = {"method": "closed-json-arithmetic-tree/Fraction",
                      "review_status": "assistant-reviewed", "review": spec["review"],
                      "variables": {name: {"original": vals[0], "numeric": vals[1], "unit": vals[2]}
                                    for name, vals in spec["values"].items()},
                      "original_steps": original_trace, "numeric_steps": numeric_trace,
                      "relational_checks": spec.get("checks", []),
                      "original_exact_answer": str(original_solved),
                      "numeric_exact_answer": str(numeric_value),
                      "source_final_answer_verified": True,
                      "unit_review": "Each step's unit and semantic relation independently assistant-reviewed; integer/nonnegative/positive/cent-exact constraints executed. Units are annotations, not an automatic dimensional proof.",
                      "difficulty_change": "Same quantities' roles, equations, reasoning steps and rounding regime; numeric magnitude changes only. Empirical difficulty equivalence is not claimed."}
            solution = "\n".join(f"{step['explanation']} {step['substitution']} = {answer_text(Fraction(step['value']))} {step['unit']}." for step in numeric_trace)
            solution += f"\n#### {answer_text(numeric_value)}"
            datasets["numeric"].append({**common, "type": "numeric", "variant_type": "numeric",
                                       "question": numeric_question, "answer": solution,
                                       "change_notes": {"replacements": spec["numeric"], "review": spec["review"]},
                                       "solver": solver})
            common_solver.update({"method": solver["method"], "original_steps": original_trace,
                                  "original_exact_answer": str(original_solved), "review": spec["review"],
                                  "source_final_answer_verified": True})
            audit.append({**common, "numeric_status": "included", "numeric_answer": answer_text(numeric_value),
                          "review": spec["review"], "checks_passed": True})
        for variant_type, variant_question in (("original", question), ("equivalent", equivalent_question)):
            datasets[variant_type].append({**common, "type": variant_type, "variant_type": variant_type,
                                          "question": variant_question,
                                          "answer": f"#### {answer_text(original_value)}",
                                          "change_notes": [] if variant_type == "original" else spec["equivalent"],
                                          "solver": common_solver})
    # Validation happens before publishing any generated dataset.
    for variant_type, dataset in datasets.items():
        if len({row["pair_id"] for row in dataset}) != len(dataset):
            raise ValueError(f"Duplicate pair IDs in {variant_type}")
        if len({row["question"] for row in dataset}) != len(dataset):
            raise ValueError(f"Duplicate questions in {variant_type}")
    included = {row["source_id"] for row in datasets["numeric"]}
    if included & {row["source_id"] for row in exclusions} or len(included) + len(exclusions) != COUNT:
        raise ValueError("Inclusion/exclusion coverage is not exhaustive and disjoint")
    for variant_type, dataset in datasets.items():
        write_jsonl(output / f"{variant_type}.jsonl", dataset)
    write_jsonl(output / "review_audit.jsonl", audit)
    files = ["frozen_ids.json", "original.jsonl", "equivalent.jsonl", "numeric.jsonl", "review_audit.jsonl"]
    manifest = {"schema_version": 1, "seed": SEED, "sample_count": COUNT,
                "source": {"split": "test", "sha256": sha256(source), "count": len(rows)},
                "source_ids": frozen["source_ids"], "counts": {key: len(value) for key, value in datasets.items()},
                "numeric_source_ids": [row["source_id"] for row in datasets["numeric"]],
                "excluded_count": len(exclusions), "exclusions": exclusions,
                "files_sha256": {name: sha256(output / name) for name in files},
                "review_specs_sha256": sha256(specs_path), "builder_sha256": sha256(Path(__file__)),
                "review_status": "assistant-reviewed", "human_review_performed": False,
                "self_tests": self_test(),
                "protocol": {"frozen_before_variant_drafting": True, "no_model_outputs_consulted": True,
                             "no_replacement_sampling": True, "no_gpu_or_external_services": True,
                             "source_solution_use": "Only #### final numeric key read for validation after independent per-question arithmetic trees were drafted; no source rationale copied into any variant.",
                             "numeric_truth": "Independent semantic review plus exact Fraction original-and-new calculations; answers must differ.",
                             "equivalence": "Conservative exact local entity/wording substitutions, digit signatures unchanged; underlying source ambiguities retained.",
                             "comparison": "Use all 100 for original/equivalent; restrict both original and numeric to numeric_source_ids for paired numeric comparison.",
                             "exclusion_bias": "Ambiguous/ill-posed cases excluded by mathematical audit, never by model results. Numeric subset is easier to verify and may not represent all GSM8K items.",
                             "review_limit": "Assistant review is not independent human review; mathematical proof is relative to stated/recorded ordinary word-problem conventions, not a claim that toy real-world premises are empirical facts."},
                "reproduction": "python -I qat/build_gsm8k_variants.py --freeze-only; python -I qat/build_gsm8k_variants.py",
                "runtime": {"python": sys.version.split()[0], "arithmetic": "fractions.Fraction"}}
    dump(output / "manifest.json", manifest)
    print(json.dumps({"counts": manifest["counts"], "excluded": len(exclusions),
                      "manifest_sha256": sha256(output / "manifest.json")}, sort_keys=True))


def self_test():
    """CPU-only regression tests for the closed solver and strict constraints."""
    tests = []

    def check(name, passed):
        if not passed:
            raise AssertionError(name)
        tests.append(name)

    def rejects(name, function, *args):
        try:
            function(*args)
        except (ValueError, TypeError, ZeroDivisionError):
            tests.append(name)
        else:
            raise AssertionError(f"Expected rejection: {name}")

    check("fraction_exact_decimal", calculate(["add", "0.1", "0.2"], {}) == Fraction(3, 10))
    check("nested_fraction_tree", calculate(["div", ["sub", "16.8", "0.8"], "2/5"], {}) == 40)
    check("subtraction_order", calculate(["sub", "10", "2", "3"], {}) == 5)
    check("unit_fraction_answer", answer_text(Fraction(21, 2)) == "10.5")
    check("small_decimal_answer", answer_text(Fraction(1, 200)) == "0.005")
    check("negative_decimal_answer", answer_text(Fraction(-1, 8)) == "-0.125")
    check("reference_commas", reference_final("unused rationale\n#### 12,345") == 12345)
    check("reference_negative", reference_final("#### -2.5") == Fraction(-5, 2))
    check("replacement_all_mentions", replace_exact("5 baskets; all 5 baskets", [["5 baskets", "6 baskets"]]) == "6 baskets; all 6 baskets")
    rejects("reject_code_literal", calculate, "__import__('os').system('true')", {})
    rejects("reject_arbitrary_operator", calculate, ["pow", "2", "3"], {})
    rejects("reject_infix_expression", calculate, "2+3", {})
    rejects("reject_zero_divisor", calculate, ["div", "1", "0"], {})
    rejects("reject_float_node", calculate, 0.1, {})
    rejects("reject_fractional_people", enforce, Fraction(3, 2), "integer")
    rejects("reject_negative_count", enforce, Fraction(-1), "integer")
    rejects("reject_subcent_money", enforce, Fraction(1, 1000), "money")
    rejects("reject_zero_positive_mass", enforce, Fraction(0), "positive")
    rejects("reject_unrecognized_constraint", enforce, Fraction(1), "unknown")
    rejects("reject_missing_replacement", replace_exact, "five baskets", [["six", "seven"]])
    rejects("reject_identity_replacement", replace_exact, "five baskets", [["five", "five"]])
    rejects("reject_missing_answer_marker", reference_final, "answer 3")
    rejects("reject_repeated_answer_marker", reference_final, "#### 3\n#### 4")
    rejects("reject_trailing_answer_text", reference_final, "#### 3 dollars")
    rejects("reject_nonterminating_answer", answer_text, Fraction(1, 3))
    return {"passed": len(tests), "failed": 0, "tests": tests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "eval_data/gsm8k/test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval_data/gsm8k_variants_v1")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run CPU-only regression tests without reading datasets")
    parser.add_argument("--specs", type=Path, default=ROOT / "eval_data/gsm8k_variants_v1/review_specs.json")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return
    if not args.freeze_only and not (args.output_dir / "frozen_ids.json").exists():
        parser.error("Run --freeze-only first; IDs must be frozen before variant generation")
    rows, frozen = freeze(args.source, args.output_dir)
    if args.freeze_only:
        print(json.dumps({"frozen": len(frozen["source_ids"]), "seed": SEED, "source_sha256": frozen["source_sha256"]}))
    else:
        build(args.source, args.output_dir, args.specs, rows, frozen)


if __name__ == "__main__":
    main()
