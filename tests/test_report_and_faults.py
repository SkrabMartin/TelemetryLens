from dataclasses import asdict

from telemetrylens import faultcheck, insights, report, schema


def _ctx(parsed, cleaned, analysed):
    return report.ReportContext(
        source_name="demo.csv",
        import_id=parsed.import_id,
        import_config=parsed.config.to_dict(),
        cleaning_options={"drop_exact_duplicates": True},
        cleaning_decisions=cleaned.decisions,
        validation_issues=[asdict(i) for i in parsed.issues],
        filters={
            "machines": ["CNC_02"],
            "start": analysed.timestamp.min(),
            "end": analysed.timestamp.max(),
            "statuses": [],
        },
        thresholds=schema.default_thresholds(),
        anomaly_settings={"z_limit": 6, "group_by": "phase", "direction": "high"},
        tables={
            "Threshold summary": insights.threshold_summary(analysed, ["temperature_c"], schema.default_thresholds())
        },
    )


def test_report_is_deterministic_and_complete(parsed, cleaned, analysed):
    a = report.to_markdown(_ctx(parsed, cleaned, analysed))
    b = report.to_markdown(_ctx(parsed, cleaned, analysed))
    assert a == b
    assert parsed.import_id in a
    assert "2026-10-22 06:00:00Z" in a
    for decision in cleaned.decisions:
        assert decision in a
    assert "Correlation between sensors does not establish" in a
    assert report.to_json(_ctx(parsed, cleaned, analysed)) == report.to_json(_ctx(parsed, cleaned, analysed))


def test_analysis_id_changes_with_settings(parsed, cleaned, analysed):
    ctx = _ctx(parsed, cleaned, analysed)
    other = _ctx(parsed, cleaned, analysed)
    other.thresholds = {**other.thresholds, "temperature_c": {"warn": 70, "critical": 90}}
    assert ctx.analysis_id != other.analysis_id


def test_every_planted_fault_is_detected(synthetic, parsed, raw_quality, analysed):
    det = faultcheck.detection_table(
        synthetic.ground_truth,
        raw_quality,
        parsed.issues,
        insights.violation_episodes(analysed, "temperature_c", 75),
        insights.flag_anomalies(analysed, parsed.sensors),
    )
    assert det.detected.all(), det[~det.detected]
