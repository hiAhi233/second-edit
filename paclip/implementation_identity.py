"""Conservative executable-source identity, shared by checkpoints and clean caches."""
from pathlib import Path
from paclip.data.fewshot import file_sha256


def implementation_fingerprint(scope="training", package_root=None):
    package = Path(package_root) if package_root else Path(__file__).resolve().parent
    if scope == "training":
        # Include transitive local dependencies rather than maintaining a partial
        # hand-written list. Diagnostic tests are not executed by training.
        paths = [p for p in package.rglob("*.py") if "diagnostics" not in p.relative_to(package).parts]
        paths += [package.parent / n for n in ("run_brain_novelty.py", "prepare_patient_splits.py")]
    elif scope == "features":
        paths = [package / "implementation_identity.py", package / "config.py"]
        paths += list((package / "models").rglob("*.py"))
        paths += list((package / "data").rglob("*.py"))
        paths += list((package / "prompts").rglob("*.py"))
    else:
        raise ValueError("Unknown implementation identity scope")
    return {p.relative_to(package.parent).as_posix(): file_sha256(p) for p in sorted(set(paths))}
