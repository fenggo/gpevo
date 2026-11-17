"""``gpevo lib`` / ``gpevo ffield`` dispatch to the uspexkit converters."""
from __future__ import annotations


def test_lib_and_ffield_cli_dispatch(monkeypatch):
    import gpevo.uspexkit.core as core
    from gpevo.cli import main

    calls = []
    monkeypatch.setattr(core, "lib", lambda **kw: calls.append(("lib", kw)))
    monkeypatch.setattr(core, "ffield",
                        lambda **kw: calls.append(("ffield", kw)))

    main(["lib", "--json", "a.json", "--lib", "b.lib"])
    main(["ffield", "--jsonfile", "c.json", "--ffield", "d"])

    assert calls == [
        ("lib", {"jsonfile": "a.json", "libfile": "b.lib"}),
        ("ffield", {"jsonfile": "c.json", "ffieldfile": "d"}),
    ]


def test_lib_and_ffield_registered():
    from gpevo.cli import _build_parser

    sub_action = next(a for a in _build_parser()._actions
                      if isinstance(getattr(a, "choices", None), dict))
    assert "lib" in sub_action.choices
    assert "ffield" in sub_action.choices
    for name in ("lib", "ffield"):
        dests = {a.dest for a in sub_action.choices[name]._actions}
        assert "jsonfile" in dests
