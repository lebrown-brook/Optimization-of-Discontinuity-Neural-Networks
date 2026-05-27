# CLAUDE.md

## Code Style
- Keep implementations modular
- Do NOT use defensive fallbacks (e.g. avoid `dict.get(key, default)`), instead let errors raise to indicate to me that something is not in the form it was expected to be
- Do NOT introduce backwards compatibility unless I tell you to do so
- Do NOT work around a problem with shortcuts or hardcoded fixes — diagnose the root cause, and discuss any non-standard approach with me before implementing it
- Comments should orient the reader or explain non-obvious logic; most lines need none. Do NOT modify existing comments in code you're not otherwise changing


## Other things NOT to Do
- Do NOT auto-commit — always ask before any git operations
- Do NOT modify experiment results or data files
- Do NOT add dependencies without flagging them first

## Running Python Files

Conda is at `C:\Users\brook\miniconda3` and is not on `PATH`. Use `conda run` via PowerShell:

```powershell
& C:\Users\brook\miniconda3\Scripts\conda.exe run -n deltann python relative/path/to/script.py
```

For scripts inside a package (so the editable install resolves imports correctly), use module syntax:

```powershell
& C:\Users\brook\miniconda3\Scripts\conda.exe run -n deltann python -m src.script_name
```

The `source ~/.bash_profile && conda activate deltann` pattern does **not** work — conda is not on the bash `PATH` in this environment.


## Useful Documentation

- `./README.md` — project orientation, install, quick start
- `./OVERVIEW.md` — deep technical reference (models, algorithms, config, pipeline, caveats)

