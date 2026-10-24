"""The demo web app.

Identity is chosen from a dropdown. That is a deliberate shortcut, not an oversight:
the project is about what happens *after* you know who is asking, and a real
deployment would put a session behind this. It is called out in the UI so nobody
mistakes it for an auth story.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..answer.generate import DEFAULT_MAX_DISTANCE, answer_question
from ..answer.providers import ClaudeProvider, FakeProvider
from ..embed.fake import FakeEmbedder
from ..ingest.chunk import chunk_text
from ..ingest.parse import parse_file
from ..store.repo import Store

TEMPLATES = Path(__file__).resolve().parent.parent / "web"


def build_app(
    *,
    dsn: str | None = None,
    embedder_kind: str = "fake",
    provider_kind: str = "fake",
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> FastAPI:
    if embedder_kind == "local":
        from ..embed.local import LocalEmbedder

        embedder = LocalEmbedder()
    else:
        embedder = FakeEmbedder(dim=64)
    provider = ClaudeProvider() if provider_kind == "claude" else FakeProvider()

    app = FastAPI(title="scoped-rag")
    templates = Jinja2Templates(directory=str(TEMPLATES))

    def open_store() -> Store:
        store = Store.open(dsn, dim=embedder.dim)
        store.init_schema()
        return store

    def people(store: Store) -> list[tuple[str, str, str]]:
        return store.principals("user")

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request, who: str = ""):
        store = open_store()
        try:
            users = people(store)
            who = who or (users[0][0] if users else "")
            return templates.TemplateResponse(
                request,
                "index.html",
                {
                    "users": users,
                    "who": who,
                    "groups": store.groups_of(who) if who else [],
                    "is_admin": store.is_admin(who) if who else False,
                    "question": "",
                    "answer": None,
                },
            )
        finally:
            store.close()

    @app.post("/ask", response_class=HTMLResponse)
    def ask(request: Request, who: str = Form(...), question: str = Form(...), k: int = Form(5)):
        store = open_store()
        try:
            result = answer_question(
                store,
                embedder,
                provider,
                asker=who,
                question=question,
                k=k,
                max_distance=max_distance,
            )
            return templates.TemplateResponse(
                request,
                "index.html",
                {
                    "users": people(store),
                    "who": who,
                    "groups": store.groups_of(who),
                    "is_admin": store.is_admin(who),
                    "question": question,
                    "answer": result,
                },
            )
        finally:
            store.close()

    @app.get("/documents", response_class=HTMLResponse)
    def documents(request: Request):
        store = open_store()
        try:
            rows = [(d, store.grants_of(d.id)) for d in store.documents()]
            return templates.TemplateResponse(
                request,
                "documents.html",
                {
                    "rows": rows,
                    "principals": store.principals(),
                },
            )
        finally:
            store.close()

    @app.post("/upload")
    async def upload(
        file: UploadFile,
        title: str = Form(""),
        visibility: str = Form("restricted"),
        grants: str = Form(""),
    ):
        store = open_store()
        try:
            suffix = Path(file.filename or "upload.txt").suffix or ".txt"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = Path(tmp.name)
            try:
                text = parse_file(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)
            doc_id = store.add_document(
                title or Path(file.filename or "未命名").stem,
                source=file.filename,
                visibility=visibility,
            )
            specs = chunk_text(text)
            store.add_chunks(
                doc_id, specs, embedder.embed_documents([c.text for c in specs]), embedder.name
            )
            for principal in [g.strip() for g in grants.split(",") if g.strip()]:
                store.grant(doc_id, principal)
            store.commit()
        finally:
            store.close()
        return RedirectResponse("/documents", status_code=303)

    return app
