"""B-06 regression: argv-shape match for Grimoire daemon, not substring."""
import pytest

import grimoire.utils.system as sysmod


@pytest.mark.parametrize(
    "argv, expected",
    [
        # Background form spawned by start_daemon_background.
        (["/usr/bin/python3", "-m", "grimoire", "daemon"], True),
        (["/usr/bin/python3", "-m", "grimoire", "daemon", "--json"], True),
        # Console-script foreground form.
        (["/home/u/venv/bin/grimoire", "daemon"], True),
        (["/home/u/venv/bin/grimoire", "daemon", "run"], True),
        # NOT a daemon: other grimoire subcommands.
        (["/home/u/venv/bin/grimoire", "scan"], False),
        (["/usr/bin/python3", "-m", "grimoire", "ask", "hi"], False),
        # NOT a daemon: unrelated processes whose argv contains "grimoire".
        (["/usr/bin/vim", "/path/to/Grimoire/grimoire/cli.py"], False),
        (["bash", "-c", "echo grimoire daemon"], False),
        # Edge cases.
        ([], False),
        (["only-one-arg"], False),
    ],
)
def test_is_grimoire_process_argv_shape(monkeypatch, argv, expected):
    monkeypatch.setattr(sysmod, "_read_cmdline_argv", lambda pid: argv)
    assert sysmod._is_grimoire_process(pid=0) is expected
