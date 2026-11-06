from pathlib import Path

from echopype_fm_std import batch


def test_run_folder_resumes_only_matching_config(tmp_path, monkeypatch):
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
    run_directory = tmp_path / "output" / "archive" / "amber-47"
    target_directory = run_directory / "targets"
    target_directory.mkdir(parents=True)
    (run_directory / "config.yaml").write_text(config_path.read_text(), encoding="utf-8")
    (target_directory / "processed_single_targets.csv").touch()

    calls: list[Path] = []
    call_kwargs: list[dict] = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs["raw_path"])
        call_kwargs.append(kwargs)
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
    assert "echo_data" not in results[0]
    output_directory = call_kwargs[0]["output_directory"]
    assert output_directory.parent == tmp_path / "output" / "archive"
    assert output_directory == run_directory
    assert output_directory.name.split("-")[0] in batch._BATCH_RUN_WORDS
    assert output_directory.name.split("-")[1].isdigit()
    assert len(output_directory.name.split("-")[1]) == 2


def test_run_folder_creates_new_directory_for_different_config(tmp_path, monkeypatch):
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()
    raw_path = raw_directory / "sample.raw"
    raw_path.touch()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "navigation:\n  platform: TEST\noutput:\n  directory: output\n",
        encoding="utf-8",
    )
    old_run = tmp_path / "output" / "archive" / "amber-47"
    old_run.mkdir(parents=True)
    (old_run / "config.yaml").write_text(
        "navigation:\n  platform: TEST\noutput:\n  directory: output\nbottom:\n  method: legacy\n",
        encoding="utf-8",
    )

    calls: list[dict] = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)
        return {"raw_path": kwargs["raw_path"], "stages_completed": []}

    monkeypatch.setattr(batch, "run_pipeline", fake_run_pipeline)
    batch.run_folder(
        raw_directory,
        tmp_path / "navigation.db",
        config_path,
        output_path=tmp_path / "output",
    )

    assert len(calls) == 1
    assert calls[0]["output_directory"] != old_run
    assert calls[0]["output_directory"].parent == tmp_path / "output"


def test_new_batch_directory_is_marked_before_workers_start(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "navigation:\n  platform: TEST\noutput:\n  directory: output\n",
        encoding="utf-8",
    )

    output_directory = batch._create_batch_output_directory(
        tmp_path / "output",
        config_path,
    )

    assert (output_directory / "config.yaml").read_text(encoding="utf-8") == (
        config_path.read_text(encoding="utf-8")
    )
    assert batch._find_matching_run_directory(tmp_path / "output", config_path) == (
        output_directory
    )


def test_run_folder_rejects_nonpositive_worker_count(tmp_path):
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()
    (raw_directory / "sample.raw").touch()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("navigation:\n  platform: TEST\n", encoding="utf-8")

    try:
        batch.run_folder(
            raw_directory,
            tmp_path / "navigation.db",
            config_path,
            workers=0,
        )
    except ValueError as exc:
        assert str(exc) == "workers must be at least 1."
    else:
        raise AssertionError("run_folder should reject workers=0")
