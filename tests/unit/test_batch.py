from pathlib import Path

from echopype_fm_std import batch


def test_run_folder_skips_raw_files_with_existing_target_output(tmp_path, monkeypatch):
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()
    processed_raw = raw_directory / "processed.raw"
    pending_raw = raw_directory / "pending.raw"
    processed_raw.touch()
    pending_raw.touch()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "navigation:\n"
        "  platform: TEST\n"
        "output:\n"
        "  directory: output\n"
        "  targets_directory: targets\n",
        encoding="utf-8",
    )
    target_directory = tmp_path / "output" / "targets"
    target_directory.mkdir(parents=True)
    (target_directory / "processed_single_targets.csv").touch()

    calls: list[Path] = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs["raw_path"])
        return {"raw_path": kwargs["raw_path"], "stages_completed": []}

    monkeypatch.setattr(batch, "run_pipeline", fake_run_pipeline)

    results = batch.run_folder(
        raw_directory=raw_directory,
        navigation_db=tmp_path / "navigation.db",
        config_path=config_path,
        output_path=tmp_path / "output",
    )

    assert calls == [pending_raw]
    assert [result["raw_path"] for result in results] == [pending_raw]
