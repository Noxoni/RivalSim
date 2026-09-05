"""Focused CPU checks for final-only publication recovery."""

import pytest
import torch

from benchmarks import finalize_rival2_ssl_strong_exploration as final


def test_recursive_finite_state_check():
    assert final.finite_tree({"state": [torch.tensor([1.0]), 2.0, None, "metadata"]})
    assert not final.finite_tree({"state": [torch.tensor([float("nan")])]})
    assert not final.finite_tree({"evaluation": float("inf")})


def test_recovery_refuses_unbound_error_before_any_write(monkeypatch):
    monkeypatch.setattr(final, "read", lambda _: {"stderr_sha256": "EXPECTED"})
    monkeypatch.setattr(final.s.c.engine, "sha256_file", lambda _: "CHANGED")

    def forbidden_write(*_args, **_kwargs):
        pytest.fail("must not publish when bound failure evidence differs")

    monkeypatch.setattr(final.s.c.engine, "write_json", forbidden_write)
    monkeypatch.setattr(final.shutil, "copyfile", forbidden_write)
    with pytest.raises(ValueError, match="publication recovery input changed: stderr"):
        final.recover_publication({})
