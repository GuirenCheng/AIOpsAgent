"""评测纯函数（app.eval_metrics）单元测试：无 Milvus / LLM，全确定性。"""
import json

import pytest

from app.eval_metrics import (
    build_report,
    check_regression,
    hit_k,
    load_diagnosis_scenarios,
    load_retrieval_golden,
    precision_k,
    scalar_metrics,
    score_response,
)


def test_load_retrieval_golden_valid(tmp_path):
    f = tmp_path / "g.json"
    f.write_text(
        json.dumps([{"query": "q", "relevant_file": "a.md"}], ensure_ascii=False),
        encoding="utf-8",
    )
    assert load_retrieval_golden(str(f)) == [{"query": "q", "relevant_file": "a.md"}]


def test_load_retrieval_golden_missing_field(tmp_path):
    f = tmp_path / "g.json"
    f.write_text(json.dumps([{"query": "q"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_retrieval_golden(str(f))


def test_load_diagnosis_scenarios_file_and_dir(tmp_path):
    single = tmp_path / "s.json"
    single.write_text(json.dumps({"name": "x"}), encoding="utf-8")
    assert load_diagnosis_scenarios(str(single)) == [{"name": "x"}]

    d = tmp_path / "scenarios"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"name": "a"}), encoding="utf-8")
    (d / "b.json").write_text(json.dumps({"name": "b"}), encoding="utf-8")
    got = load_diagnosis_scenarios(str(d))
    assert [s["name"] for s in got] == ["a", "b"]


def test_hit_k_and_precision_k():
    ranked = [("a.md", 0.9), ("b.md", 0.8), ("a.md", 0.7), ("c.md", 0.6)]
    assert hit_k(ranked, "a.md", 1) is True
    assert hit_k(ranked, "a.md", 3) is True
    assert hit_k(ranked, "c.md", 3) is False
    assert precision_k(ranked, "a.md", 3) == pytest.approx(2 / 3)


def test_score_response_pass_and_fail():
    expected = {"root_cause_keywords": ["CPU", "死循环"], "min_keyword_hits": 2}
    ok, matched = score_response("根因是 CPU 死循环", expected)
    assert ok is True
    assert "CPU" in matched and "死循环" in matched

    ok2, _ = score_response("内存不足", expected)
    assert ok2 is False


def test_build_report_and_scalar_metrics():
    report = build_report(
        {"n": 10, "top_k": 3, "threshold": 0.5, "hit_at_3": 1.0, "precision_at_3": 0.6},
        {"n": 5, "passed": 4, "accuracy": 0.8},
    )
    assert report["timestamp"]
    assert scalar_metrics(report) == {
        "retrieval.hit@3": 1.0,
        "retrieval.precision@3": 0.6,
        "diagnosis.accuracy": 0.8,
    }


def test_check_regression_no_regression():
    cur = build_report({"hit_at_3": 1.0, "precision_at_3": 0.6}, {"accuracy": 0.8})
    base = build_report({"hit_at_3": 1.0, "precision_at_3": 0.6}, {"accuracy": 0.8})
    assert check_regression(cur, base) == []


def test_check_regression_detects_drop():
    cur = build_report({"hit_at_3": 0.8, "precision_at_3": 0.6}, {"accuracy": 0.8})
    base = build_report({"hit_at_3": 1.0, "precision_at_3": 0.6}, {"accuracy": 0.8})
    reg = check_regression(cur, base)
    assert any("retrieval.hit@3" in m for m in reg)


def test_check_regression_respects_tolerance():
    # 诊断准确率容差 0.10：跌 0.04 不算退化
    cur = build_report({"hit_at_3": 1.0, "precision_at_3": 0.6}, {"accuracy": 0.76})
    base = build_report({"hit_at_3": 1.0, "precision_at_3": 0.6}, {"accuracy": 0.8})
    assert check_regression(cur, base) == []


def test_check_regression_missing_metric_skipped():
    # 只跑检索（缺诊断）时，缺失的 stage 不参与对比、不算退化
    cur = build_report({"hit_at_3": 1.0}, {})
    base = build_report({"hit_at_3": 1.0, "precision_at_3": 0.6}, {"accuracy": 0.8})
    assert check_regression(cur, base) == []
