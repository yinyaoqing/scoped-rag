# scoped-rag

**受權限約束的企業文件問答。權限過濾在檢索之內，不在檢索之後。**

企業知識庫的 RAG 有一個和一般 RAG demo 完全不同的要求：它不能把使用者無權開啟的文件洩漏出去。常見作法是先做向量檢索取 top-k，再把無權查看的結果濾掉——這有兩個問題：

1. **安全性**：過濾後的結果數量會隨權限變動，洩漏文件的存在性；若被濾掉的內容已經進了提示，更是直接洩漏。
2. **品質**：「top-k 之後過濾」不等於「過濾之後取 top-k」。有權限的使用者會拿到比應得更差的結果，因為前 k 名被他看不到的文件佔滿了。

scoped-rag 把權限謂詞和向量排序放進同一次查詢，由資料庫一起求值：先過濾、再排序、才取 k。並用一組越權檢索評測證明它。

## 評測結果

```
22 cases   leak_rate 0.0   permitted_recall 1.0   refusal_correctness 1.0
```

`leak_rate` 是唯一的硬指標：只要有一個禁止片段進入檢索集，整場評測就算失敗。`permitted_recall` 是它的對照——一個什麼都不回傳的檢索器永遠不會洩漏，也永遠沒用。

評測**刻意只衡量權限範圍，不衡量檢索品質**：它用決定性的假 embedder 與假 LLM 執行，不需任何 API key、可直接進 CI，把「權限有沒有洩漏」和「嵌入模型好不好」分開。後者換模型就會變，前者必須恆為真。

這 22 個案例對每一種檢索設定都要成立：純向量、關鍵字、混合，以及各自加上重排，共六種，全部 `leak_rate 0.0`。

`evals/leakage/cases.yaml` 的四類案例：

| 類別 | 問的是 |
| --- | --- |
| `must_not_retrieve` | 文件裡有答案，但提問者無權讀取 |
| `must_retrieve` | 有權限時確實取得（防過度封鎖） |
| `boundary` | 撤銷授權、退出部門、文件在部門間搬移、個人與群組雙重授權 |
| `refuses_when_nothing_permitted` | 整個語料都無權限時拒答，而不是猜 |

## 檢索品質：混合檢索與重排

權限評測之外另有一組品質評測（`evals/retrieval/set.yaml`：虛構公司的 26 份文件、46 個問題），用真實的嵌入模型與 cross-encoder 在本機執行：

| 設定 | hit@1 | hit@3 | MRR | 每題耗時 |
| --- | --- | --- | --- | --- |
| 向量 | 0.674 | 0.913 | 0.807 | 11 ms |
| 關鍵字（BM25） | 0.674 | 0.739 | 0.720 | 12 ms |
| 混合（RRF） | 0.739 | 0.913 | 0.833 | 24 ms |
| 混合 + 重排 | 0.826 | 0.978 | 0.899 | 372 ms |

重排是明確的改善；混合檢索則有得有失（口語改寫的問題反而變差），所以預設模式仍是純向量。評測集小且由作者自寫，這些數字是設定之間的相對比較，不是基準測試。完整的表、各題型的結果與限制見 [docs/decisions/hybrid-and-rerank.md](docs/decisions/hybrid-and-rerank.md)。

關鍵字檢索有一個和權限直接相關的設計：BM25 的語料統計（詞的文件頻率、平均長度）**只從提問者可讀的片段計算**，和排序在同一次查詢完成。若用全語料統計，一份看不到的文件會改變看得到的文件的分數，排序就成了探測隱藏文件的旁路。

## 試跑

```bash
uv sync --group dev
uv run scoped-rag demo                  # 載入範例語料（5 份文件、5 個身分）
uv run scoped-rag ask "經理級的年度調薪區間是多少" --as u:alice --k 20
uv run scoped-rag ask "經理級的年度調薪區間是多少" --as u:bob   --k 20
uv run scoped-rag ask "經理級的年度調薪區間是多少" --as u:bob --mode hybrid
uv run scoped-rag eval
uv run scoped-rag eval --mode hybrid --rerank   # 同一組案例，換檢索設定
uv run scoped-rag eval-quality --rerank fake    # 品質評測的管線（假模型，數字無意義）
uv run scoped-rag serve                 # http://127.0.0.1:8000
```

既有的資料庫要先 `uv run scoped-rag reindex` 建關鍵字索引，才能用 `--mode keyword` 或 `hybrid`；沒建會直接報錯，不會悄悄退回純向量。

同一個問題，Alice（業務部員工）拿不到薪資級距文件，Bob（經理）拿得到。Web UI 的身分切換器就是為了讓這個差異看得見。

## 架構

```
src/scoped_rag/
  ingest/parse.py     檔案 → 純文字（txt/md/pdf/docx；一個壞檔不擋整批）
  ingest/chunk.py     切分，保留原文偏移，優先切在句界
  embed/base.py       Embedder 協定
  embed/local.py      sentence-transformers（預設，本機 GPU，零 API 成本）
  embed/fake.py       決定性雜湊向量（測試與評測，不需模型）
  store/dialect.py    SQLite 與 PostgreSQL 的「唯一差異」都收在這裡
  store/schema.py     DDL
  store/repo.py       持久層（不含任何權限判斷）
  retrieve/acl.py     權限謂詞：唯一一份
  retrieve/search.py  向量、關鍵字（BM25）兩條檢索各自在單次查詢完成權限過濾 + 排序；RRF 融合；重排
  terms.py            關鍵字檢索的切詞（建索引與查詢共用一份）
  rerank/base.py      Reranker 協定（只收文字，拿不到 store）
  rerank/local.py     本機 cross-encoder（BAAI/bge-reranker-base）
  rerank/fake.py      決定性的假重排器（測試用）
  answer/providers.py LlmProvider 協定（Claude 實作 + 決定性 fake），每次呼叫記成本
  answer/generate.py  以允許的片段生成答案，驗證引用，必要時拒答
  api/app.py + web/   最小 Web UI
  evaluate.py         越權檢索評測
  quality.py          檢索品質評測
```

### 兩種後端，同一段 SQL

為了讓測試與評測在零基礎設施下跑完，儲存層支援兩個後端，而**差異只有向量型別與距離運算式**：

| | 正式 | 測試／評測 |
| --- | --- | --- |
| 資料庫 | PostgreSQL + pgvector | SQLite |
| 向量型別 | `vector(n)` | BLOB（float32） |
| 距離 | `embedding <=> :q` | `cosine(vector, :q)`（Python 註冊的函式） |

ACL 謂詞、join、排序與取 k 的 SQL 完全相同。這不是為了方便，是為了正確：把權限邏輯寫兩遍必然發散，而「權限不洩漏」的測試必須涵蓋正式後端。設定 `SCOPED_RAG_DSN` 指向 Postgres 即切換。

### 兩道防線

1. 模型只看得到允許的片段。
2. 生成後逐一驗證答案引用的片段編號都在允許集合內；不在就拋例外，不悄悄丟棄。

`Answer` 同時保留 `retrieved`（權限放行的）與 `used`（夠相關、進了提示的）。兩者分開是刻意的：權限保證是關於 `retrieved`，而相關度門檻是產品設定，不該有機會掩蓋洩漏。

## 資料模型

角色（管理員／經理／員工）與部門都是 `principals` 裡 `kind='group'` 的列，所以一筆授權可以指向個人或群組，不需要兩套程式路徑。管理員可讀全部，寫成 `retrieve/acl.py` 裡一個明確的分支，而不是藏在角色表裡。

## 刻意不做

知識圖譜、OCR、雲端嵌入供應商（只留介面）、查詢改寫（HyDE、multi-query）、語意切分、多租戶計費、真實身分驗證（demo 以下拉選單指定身分，UI 中明白標示）、雲端部署。這些是取捨，不是遺漏。

## 測試

```bash
uv run pytest            # 65 個測試，全離線，不需 API key 也不需模型
uv run ruff check .
```

同一套測試與兩個評測可以改指向 PostgreSQL 執行（每個 store 建立前會清空該資料庫，請用專門的測試庫）：

```bash
uv sync --group dev --extra pg
SCOPED_RAG_TEST_DSN=postgresql://USER:PASSWORD@127.0.0.1:5432/scoped_rag_test uv run pytest
```

其中幾個值得一看：`test_hidden_documents_do_not_move_visible_scores` 確認隱藏文件不會改變可見文件的 BM25 分數；`test_filter_runs_before_ranking` 對任何「先檢索後過濾」的實作都會失敗；`test_a_broken_predicate_is_caught` 把權限謂詞換成全放行，確認評測真的會抓到——否則一個讓所有人看到所有文件的回歸，也會顯示 `leak_rate` 0。

---

## English summary

**scoped-rag — permission-aware RAG where the access filter runs inside retrieval, not after it.**

Enterprise knowledge bases must never surface a document the asker cannot open. The common approach — retrieve top-k, then drop what the user may not see — leaks (result counts reveal hidden documents) and ranks worse (your top k gets eaten by rows you never see). scoped-rag puts the access predicate and the vector ordering in one query, so the database filters first, ranks what survived, and then takes k.

The claim is backed by a 22-case privilege-escalation eval: `leak_rate 0.0`, `permitted_recall 1.0`, `refusal_correctness 1.0`. The eval measures access control only, deliberately: it runs on a deterministic fake embedder and fake LLM, needs no API key, and keeps "did permissions leak" separate from "is the embedding model good".

Retrieval has three modes — vector, keyword (BM25) and their reciprocal-rank fusion — plus an optional local cross-encoder re-ranker. Each leg carries the access predicate in its own query, and BM25's corpus statistics are computed over the asker's readable set inside that query, so a hidden document cannot shift the scores of visible ones. On a 46-question synthetic set, hybrid + re-rank lifts hit@1 from 0.674 to 0.826 with zero leaks in every configuration; the limits of that measurement are written up in `docs/decisions/hybrid-and-rerank.md`.

Storage supports PostgreSQL + pgvector in production and SQLite in tests, and the only differences are the vector type and the distance expression — the ACL predicate, joins, ordering and LIMIT are the same SQL on both, so the tests that prove no leakage cover the production backend too.

Stack: Python 3.12, FastAPI, pgvector, sentence-transformers (local embeddings, no API cost), Claude for generation with per-call cost accounting and a `--max-cost` budget.
