from __future__ import annotations

import pytest

from scoped_rag.embed.fake import FakeEmbedder
from scoped_rag.ingest.chunk import chunk_text
from scoped_rag.store.repo import Store, seed_roles

# One small corpus, reused by every test. Three documents with different reach:
# an org-wide handbook, a leave policy for employees, and a salary band document
# only managers may read.
HANDBOOK = "公司手冊。上班時間為上午九點到下午六點，彈性上下班區間為一小時。"
LEAVE = "育嬰假規定。員工年資滿六個月得申請育嬰留職停薪，最長兩年。申請需於三十日前提出。"
SALARY = "薪資調整級距。經理級年度調薪區間為百分之三到百分之八，由部門主管核定後送人資。"


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=64)


@pytest.fixture
def store(embedder: FakeEmbedder) -> Store:
    s = Store.open("sqlite://:memory:", dim=embedder.dim)
    s.init_schema()
    seed_roles(s, departments=["hr"])
    s.add_user("u:alice", "Alice（員工）")
    s.add_user("u:bob", "Bob（經理）")
    s.add_user("u:root", "Root（管理員）")
    s.add_member("u:alice", "g:employee")
    s.add_member("u:bob", "g:employee")
    s.add_member("u:bob", "g:manager")
    s.add_member("u:root", "g:admin")

    for doc_id, title, body, visibility, grantees in [
        ("handbook", "公司手冊", HANDBOOK, "org", []),
        ("leave", "育嬰假規定", LEAVE, "restricted", ["g:employee"]),
        ("salary", "薪資調整級距", SALARY, "restricted", ["g:manager"]),
    ]:
        s.add_document(title, source=f"{doc_id}.md", visibility=visibility, doc_id=doc_id)
        specs = chunk_text(body, size=40, overlap=8)
        vectors = embedder.embed_documents([c.text for c in specs])
        s.add_chunks(doc_id, specs, vectors, embedder.name)
        for g in grantees:
            s.grant(doc_id, g)
    s.commit()
    yield s
    s.close()
