import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
WRAPPER = ROOT / "scripts/run_daily.sh"


def _run_fake(tmp_path, fail_step):
    fake = tmp_path / "fake-python"
    fake.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  */manage.py) step=ingestion ;;\n"
        "  */prospective.py) step=prospective ;;\n"
        "  */discovery.py) exit 0 ;;\n"
        "esac\n"
        f"[ \"$step\" = \"{fail_step}\" ] && echo 'deliberate {fail_step} failure' >&2 && exit 7\n"
        "exit 0\n"
    )
    fake.chmod(0o755)
    env = {"PATH": "/usr/bin:/bin", "DAILY_PYTHON": str(fake), "DAILY_LOG_DIR": str(tmp_path), "HOME": str(tmp_path)}
    return subprocess.run([str(WRAPPER)], cwd=ROOT, env=env, text=True, capture_output=True)


def test_wrapper_propagates_ingestion_failure_with_record(tmp_path):
    result = _run_fake(tmp_path, "ingestion")
    assert result.returncode != 0
    assert "DAILY FAILURE: ingestion step failed" in (tmp_path / "prospective_daily.error.log").read_text()


def test_wrapper_propagates_prospective_failure_with_record(tmp_path):
    result = _run_fake(tmp_path, "prospective")
    assert result.returncode != 0
    assert "DAILY FAILURE: prospective step failed" in (tmp_path / "prospective_daily.error.log").read_text()


def test_wrapper_propagates_both_required_step_failures(tmp_path):
    result = _run_fake(tmp_path, "ingestion")
    assert result.returncode != 0
    result = _run_fake(tmp_path, "prospective")
    assert result.returncode != 0
