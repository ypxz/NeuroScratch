from __future__ import annotations


def test_import_package() -> None:
    import neuroscratch

    assert neuroscratch.__all__ == ["export", "model", "reference", "training"]
