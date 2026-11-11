"""Run manual evaluations; preserve every version as a simple text file."""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# This file lives inside agents/reference_cases. Go up two folders to find
# the project root, so direct execution can still import the agents package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents import config
from agents.defense import create_report as defense
from agents.prosecution import create_report as prosecution
from agents.reference_cases import build_reference_context


def main():
    # This is a live experiment, not an offline test. Each case requests two
    # reports; each report currently needs a draft call and an editing call.
    # --help only shows instructions and makes no API calls.
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate live reports for the three reference cases (uses the paid API)."
    )
    parser.add_argument("tag", help="New experiment name, using letters, numbers and underscores.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "output" / "output",
        help="Folder for the three TXT files; defaults to the current output/output folder.",
    )
    args = parser.parse_args()
    tag = args.tag
    if not tag.replace("_", "").isalnum():
        parser.error("Use only letters, numbers and underscores for the experiment name.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse duplicate names before spending API credit on any case.
    for stem in ("output1stcase", "output2ndcase", "output3rdcase"):
        if (output_dir / f"{stem}_{tag}.txt").exists():
            parser.error("An output with this experiment name already exists. Choose a new tag.")
    # Save the code version so we can trace which code produced the reports.
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=project_root
    ).strip()
    for case, stem in [
        ("01_concentrated_transfers", "output1stcase"),
        ("02_regular_payments", "output2ndcase"),
        ("03_insufficient_information", "output3rdcase"),
    ]:
        context = build_reference_context(case)
        path = output_dir / f"{stem}_{tag}.txt"
        # Mode 'x' creates a NEW file and refuses to overwrite an existing result.
        # Both agents' final reports go into this one file for this case.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(
                f"Case: {case}\nVersion: {tag}\nProvider: {config.PROVIDER}\nConfigured model: {config.MODEL}\nResponse budget per call: {config.ANTHROPIC_MAX_TOKENS}\nWorkflow: draft + one revision per role\nCode commit: {commit}\nStarted (UTC): {datetime.now(timezone.utc).isoformat()}\n\n"
            )
            for role, create in [("PROSECUTION", prosecution), ("DEFENSE", defense)]:
                print("Running", tag, case, role, flush=True)
                try:
                    report = create(context)
                # Keep failed runs visible in the results instead of hiding them.
                # Continue to the other report, but never label the failure as a pass.
                except Exception as error:
                    message = (
                        str(error).replace(config.API_KEY, "[REDACTED]")
                        if config.API_KEY
                        else str(error)
                    )
                    stream.write(f"{role}: NOT GRADED - {type(error).__name__}: {message}\n\n")
                    stream.flush()
                    print(type(error).__name__, message, flush=True)
                    continue
                for number, finding in enumerate(report.findings, 1):
                    stream.write(
                        f"{role} - CONCLUSION {number}\n{finding.claim}\nEvidence: {', '.join(finding.evidence_ids)}\n\n"
                    )
                stream.flush()
                print("Saved", path.name, role, len(report.findings), "findings", flush=True)


# Importing this file must not start paid requests. Run main only as a script.
if __name__ == "__main__":
    main()
