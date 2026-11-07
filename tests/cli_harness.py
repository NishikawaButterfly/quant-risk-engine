"""The in-process CLI harness that every CLI test goes through."""

from __future__ import annotations

import contextlib
import io

from quantrisk.cli import main


def run_cli(arguments: list[str]) -> tuple[int, str]:
    """Run the CLI in-process, returning the status a process would exit with.

    A returned code comes back with stdout. A ``SystemExit`` is never let
    through: its payload becomes the status the interpreter would exit with,
    mirroring its top-level handling, so an abort is an ordinary result a
    test can assert on rather than an exception that can end the run.

    - ``None`` is 0, with stdout;
    - an int is that status, with stdout when it is 0 and with stderr
      otherwise, which is where argparse writes its usage and error;
    - anything else, such as the CLI's ``error: ...`` message, is 1, with
      that message.
    """

    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(arguments)
    except SystemExit as exc:
        if exc.code is None:
            return 0, stdout.getvalue()
        if isinstance(exc.code, int):
            return exc.code, stdout.getvalue() if exc.code == 0 else stderr.getvalue()
        return 1, str(exc.code)
    return code, stdout.getvalue()
