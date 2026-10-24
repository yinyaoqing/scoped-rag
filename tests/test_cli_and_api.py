from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from scoped_rag.api.app import build_app
from scoped_rag.cli import main


@pytest.fixture
def dsn(tmp_path) -> str:
    return f"sqlite:///{(tmp_path / 'cli.sqlite3').as_posix()}"


def test_init_creates_the_roles(dsn, capsys):
    assert main(["--dsn", dsn, "init", "--department", "hr"]) == 0
    out = capsys.readouterr().out
    assert "g:admin" in out and "g:hr" in out


def test_demo_then_ask_shows_the_permission_difference(dsn, capsys):
    assert main(["--dsn", dsn, "demo"]) == 0
    capsys.readouterr()

    main(["--dsn", dsn, "ask", "經理級的年度調薪區間是多少", "--as", "u:alice", "--k", "20"])
    alice = capsys.readouterr().out
    main(["--dsn", dsn, "ask", "經理級的年度調薪區間是多少", "--as", "u:bob", "--k", "20"])
    bob = capsys.readouterr().out

    assert "薪資調整級距" not in alice
    assert "薪資調整級距" in bob


def test_ask_without_a_question_is_an_error(dsn, capsys):
    main(["--dsn", dsn, "demo"])
    capsys.readouterr()
    assert main(["--dsn", dsn, "ask", "--as", "u:alice"]) == 2


def test_ask_stops_at_the_budget(dsn, tmp_path, capsys):
    main(["--dsn", dsn, "demo"])
    capsys.readouterr()
    questions = tmp_path / "q.txt"
    questions.write_text("育嬰假多久\n調薪級距\n", encoding="utf-8")
    # The fake provider is free, so a zero budget stops before the first call.
    main(
        [
            "--dsn",
            dsn,
            "ask",
            "--as",
            "u:alice",
            "--questions-file",
            str(questions),
            "--max-cost",
            "0",
        ]
    )
    assert "stopped at budget" in capsys.readouterr().out


def test_ingest_reports_a_file_it_cannot_parse(dsn, tmp_path, capsys):
    main(["--dsn", dsn, "init"])
    capsys.readouterr()
    good = tmp_path / "policy.md"
    good.write_text("出差規定。出差需事先申請並檢附行程。", encoding="utf-8")
    bad = tmp_path / "image.png"
    bad.write_bytes(b"\x89PNG\r\n")
    assert main(["--dsn", dsn, "ingest", str(good), str(bad), "--grant", "g:employee"]) == 0
    captured = capsys.readouterr()
    assert "policy.md" in captured.out
    assert "image.png" in captured.err


def test_eval_command_passes_and_writes_a_report(tmp_path, capsys):
    report = tmp_path / "report.json"
    assert main(["eval", "--report", str(report)]) == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["leak_rate"] == 0.0 and data["passed"] is True
    assert "leak_rate" in capsys.readouterr().out


def test_web_ui_answers_differently_per_identity(dsn, capsys):
    main(["--dsn", dsn, "demo"])
    capsys.readouterr()
    client = TestClient(build_app(dsn=dsn, max_distance=1.2))

    home = client.get("/")
    assert home.status_code == 200
    assert "u:alice" in home.text

    alice = client.post(
        "/ask", data={"who": "u:alice", "question": "經理級的年度調薪區間是多少", "k": 20}
    )
    bob = client.post(
        "/ask", data={"who": "u:bob", "question": "經理級的年度調薪區間是多少", "k": 20}
    )
    assert "薪資調整級距" not in alice.text
    assert "薪資調整級距" in bob.text


def test_documents_page_lists_grants(dsn, capsys):
    main(["--dsn", dsn, "demo"])
    capsys.readouterr()
    client = TestClient(build_app(dsn=dsn))
    page = client.get("/documents")
    assert "g:manager" in page.text and "restricted" in page.text


def test_upload_indexes_a_file(dsn, tmp_path, capsys):
    main(["--dsn", dsn, "init"])
    capsys.readouterr()
    client = TestClient(build_app(dsn=dsn, max_distance=1.2))
    files = {"file": ("travel.md", "出差規定。出差需事先申請。".encode(), "text/markdown")}
    resp = client.post(
        "/upload",
        files=files,
        data={"title": "出差規定", "visibility": "org", "grants": ""},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert "出差規定" in resp.text
