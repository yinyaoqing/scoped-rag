"""Command line: init, demo, ingest, ask, eval, serve.

The Windows console is cp950, so stdout is reconfigured to UTF-8 with replacement
rather than letting a Chinese document title raise UnicodeEncodeError mid-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .answer.generate import DEFAULT_MAX_DISTANCE, answer_question
from .answer.providers import ClaudeProvider, FakeProvider, LlmProvider
from .embed.fake import FakeEmbedder
from .evaluate import run as run_eval
from .evaluate import write_report
from .ingest.chunk import chunk_text
from .ingest.parse import parse_many
from .store.repo import Store, seed_roles

DEMO_CASES = Path("evals/leakage/cases.yaml")


def _out() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - non-standard stream
            pass


def _embedder(kind: str, dim: int):
    if kind == "local":
        from .embed.local import LocalEmbedder

        return LocalEmbedder()
    return FakeEmbedder(dim=dim)


def _provider(kind: str) -> LlmProvider:
    return ClaudeProvider() if kind == "claude" else FakeProvider()


def _open(args, *, init: bool = False) -> tuple[Store, object]:
    embedder = _embedder(args.embedder, args.dim)
    store = Store.open(args.dsn, dim=embedder.dim)
    if init:
        store.init_schema()
    return store, embedder


def cmd_init(args) -> int:
    store, _ = _open(args, init=True)
    seed_roles(store, departments=args.department or [])
    store.commit()
    groups = [p for p in store.principals("group")]
    print(
        f"schema ready at {args.dsn or 'the default DSN'}; groups: "
        + ", ".join(g[0] for g in groups)
    )
    store.close()
    return 0


def cmd_demo(args) -> int:
    """Load the eval corpus as a working dataset, so the UI has something in it."""
    import yaml

    spec = yaml.safe_load(DEMO_CASES.read_text(encoding="utf-8"))
    root = DEMO_CASES.parent
    store, embedder = _open(args, init=True)
    seed_roles(store)
    for gid, name in (spec.get("groups") or {}).items():
        store.add_group(gid, name)
    for uid, info in (spec.get("users") or {}).items():
        store.add_user(uid, info.get("name", uid))
        for gid in info.get("groups") or []:
            store.add_member(uid, gid)
    for doc in spec.get("documents") or []:
        body = (root / doc["file"]).read_text(encoding="utf-8")
        store.add_document(
            doc["title"],
            source=doc["file"],
            visibility=doc.get("visibility", "restricted"),
            doc_id=doc["id"],
        )
        specs = chunk_text(body, size=120, overlap=24)
        store.add_chunks(
            doc["id"], specs, embedder.embed_documents([c.text for c in specs]), embedder.name
        )
        for principal in doc.get("grants") or []:
            store.grant(doc["id"], principal)
    store.commit()
    print(
        f"loaded {len(spec['documents'])} documents and {len(spec['users'])} users; "
        "try: scoped-rag ask '經理級的年度調薪區間是多少' --as u:alice"
    )
    store.close()
    return 0


def cmd_ingest(args) -> int:
    store, embedder = _open(args)
    ok, bad = parse_many(args.files)
    for parsed in ok:
        title = args.title or parsed.path.stem
        doc_id = store.add_document(
            title,
            source=str(parsed.path),
            owner_id=args.owner,
            visibility=args.visibility,
        )
        specs = chunk_text(parsed.text, size=args.chunk_size, overlap=args.overlap)
        store.add_chunks(
            doc_id, specs, embedder.embed_documents([c.text for c in specs]), embedder.name
        )
        for principal in args.grant or []:
            store.grant(doc_id, principal)
        print(f"{parsed.path.name}: {len(specs)} chunks as {doc_id} ({args.visibility})")
    store.commit()
    for failure in bad:
        print(f"skipped {failure.path.name}: {failure.reason}", file=sys.stderr)
    store.close()
    return 1 if bad and not ok else 0


def cmd_ask(args) -> int:
    store, embedder = _open(args)
    provider = _provider(args.provider)
    questions = [args.question] if args.question else []
    if args.questions_file:
        questions += [
            line.strip()
            for line in Path(args.questions_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not questions:
        print("nothing to ask: pass a question or --questions-file", file=sys.stderr)
        return 2

    spent = 0.0
    for question in questions:
        if args.max_cost is not None and spent >= args.max_cost:
            print(f"stopped at budget: US${spent:.4f} of US${args.max_cost:.4f}")
            break
        result = answer_question(
            store,
            embedder,
            provider,
            asker=args.as_user,
            question=question,
            k=args.k,
            max_distance=args.max_distance,
        )
        spent += result.cost_usd
        print(f"\nQ ({args.as_user}): {question}")
        print(f"A: {result.text}")
        if result.retrieved:
            print(
                "  可見來源: "
                + ", ".join(f"{h.title}[{h.chunk_id}] {h.distance:.3f}" for h in result.retrieved)
            )
        else:
            print("  可見來源: 無")
    print(f"\ncost US${spent:.4f}")
    store.close()
    return 0


def cmd_eval(args) -> int:
    report = run_eval(args.cases)
    if args.report:
        write_report(report, args.report)
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0 if report.passed else 1


def cmd_serve(args) -> int:
    import uvicorn

    from .api.app import build_app

    uvicorn.run(
        build_app(dsn=args.dsn, embedder_kind=args.embedder, provider_kind=args.provider),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    _out()
    parser = argparse.ArgumentParser(prog="scoped-rag", description=__doc__)
    parser.add_argument("--dsn", default=None, help="sqlite:///file.sqlite3 or postgresql://...")
    parser.add_argument("--embedder", choices=["fake", "local"], default="fake")
    parser.add_argument("--dim", type=int, default=64, help="dimension for the fake embedder")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create the schema and seed the three roles")
    p_init.add_argument("--department", action="append", help="extra department group")
    p_init.set_defaults(func=cmd_init)

    p_demo = sub.add_parser("demo", help="load the eval corpus as a working dataset")
    p_demo.set_defaults(func=cmd_demo)

    p_ing = sub.add_parser("ingest", help="parse, chunk, embed and store files")
    p_ing.add_argument("files", nargs="+")
    p_ing.add_argument("--title")
    p_ing.add_argument("--owner")
    p_ing.add_argument("--visibility", choices=["org", "restricted"], default="restricted")
    p_ing.add_argument("--grant", action="append", help="principal id, repeatable")
    p_ing.add_argument("--chunk-size", type=int, default=400)
    p_ing.add_argument("--overlap", type=int, default=60)
    p_ing.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask as a given user")
    p_ask.add_argument("question", nargs="?")
    p_ask.add_argument("--as", dest="as_user", required=True, help="principal id, e.g. u:alice")
    p_ask.add_argument("--questions-file")
    p_ask.add_argument("--k", type=int, default=5)
    p_ask.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE)
    p_ask.add_argument("--provider", choices=["fake", "claude"], default="fake")
    p_ask.add_argument("--max-cost", type=float, default=None, help="USD budget for this run")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="run the privilege-escalation eval")
    p_eval.add_argument("--cases", default=str(DEMO_CASES))
    p_eval.add_argument("--report", default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_serve = sub.add_parser("serve", help="run the web UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--provider", choices=["fake", "claude"], default="fake")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
