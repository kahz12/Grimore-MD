"""B-06 regression: argv-shape match for Grimore daemon, not substring."""
import pytest

import grimore.utils.system as sysmod


@pytest.mark.parametrize(
    "argv, expected",
    [
        # Background form spawned by start_daemon_background.
        (["/usr/bin/python3", "-m", "grimore", "daemon"], True),
        (["/usr/bin/python3", "-m", "grimore", "daemon", "--json"], True),
        # Console-script foreground form.
        (["/home/u/venv/bin/grimore", "daemon"], True),
        (["/home/u/venv/bin/grimore", "daemon", "run"], True),
        # NOT a daemon: other grimore subcommands.
        (["/home/u/venv/bin/grimore", "scan"], False),
        (["/usr/bin/python3", "-m", "grimore", "ask", "hi"], False),
        # NOT a daemon: unrelated processes whose argv contains "grimore".
        (["/usr/bin/vim", "/path/to/Grimore/grimore/cli.py"], False),
        (["bash", "-c", "echo grimore daemon"], False),
        # Edge cases.
        ([], False),
        (["only-one-arg"], False),
    ],
)
def test_is_grimore_process_argv_shape(monkeypatch, argv, expected):
    monkeypatch.setattr(sysmod, "_read_cmdline_argv", lambda pid: argv)
    assert sysmod._is_grimore_process(pid=0) is expected
