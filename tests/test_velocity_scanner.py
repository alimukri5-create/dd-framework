from __future__ import annotations

from datetime import date

import velocity_scanner as vs


def test_shariah_debt_hard_fail() -> None:
    signal = vs.check_shariah(
        {
            "sector": "Technology",
            "industry": "Software",
            "totalDebt": 40,
            "marketCap": 100,
        }
    )

    assert signal.severity == "no-go"
    assert "debt/mcap" in signal.message


def test_fresh_shelf_is_no_go() -> None:
    signal = vs.check_offering_overhang(
        {"filings": [{"form": "S-3/A", "filing_date": "2026-06-05"}]},
        as_of=date(2026, 6, 11),
    )

    assert signal.severity == "no-go"
    assert "S-3/A" in signal.message


def test_missing_data_caps_go_at_check() -> None:
    result = vs.finalize_result(
        "THIN",
        "GO",
        "RISK-ON",
        [
            vs.Signal("Shariah", "PASS", "pass", "synthetic pass", 0.4),
            vs.Signal("RS", "PASS", "pass", "synthetic pass", 0.5),
        ],
        [],
    )

    assert result.verdict == "CHECK"
    assert result.thin_data is True
    assert any("Thin data" in signal.message for signal in result.signals)


def test_batch_verdict_ordering() -> None:
    results = [
        vs.ScanResult("BAD", "NO-GO", 0.9, "RISK-ON", []),
        vs.ScanResult("OK", "GO", 0.9, "RISK-ON", []),
        vs.ScanResult("MID", "CHECK", 0.9, "RISK-ON", []),
    ]

    lines = vs.format_batch(results).splitlines()

    assert lines[1].startswith("OK")
    assert lines[2].startswith("MID")
    assert lines[3].startswith("BAD")
