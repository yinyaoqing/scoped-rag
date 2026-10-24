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

`evals/leakage/cases.yaml` 的四類案例：

| 類別 | 問的是 |
| --- | --- |
| `must_not_retrieve` | 文件裡有答案，但提問者無權讀取 |
| `must_retrieve` | 有權限時確實取得（防過度封鎖） |
| `boundary` | 撤銷授權、退出部門、文件在部門間搬移、個人與群組雙重授權 |
| `refuses_when_nothing_permitted` | 整個語料都無權限時拒答，而不是猜 |

## 試跑

```bash
uv sync --group dev
uv run scoped-rag demo                  # 載入範例語料（5 份文件、5 個身分）
uv run scoped-rag ask "經理級的年度調薪區間是多少" --as u:alice --k 20
uv run scoped-rag ask "經理級的年度調薪區間是多少" --as u:bob   --k 20
uv run scoped-rag eval
uv run scoped-rag serve                 # http://127.0.0.1:8000
```

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
  retrieve/search.py  單次查詢完成權限過濾 + 向量排序
  answer/providers.py LlmProvider 協定（Claude 實作 + 決定性 fake），每次呼叫記成本
  answer/generate.py  以允許的片段生成答案，驗證引用，必要時拒答
  api/app.py + web/   最小 Web UI
  evaluate.py         越權檢索評測
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

知識圖譜、OCR、雲端嵌入供應商（只留介面）、re-ranking、多租戶計費、真實身分驗證（demo 以下拉選單指定身分，UI 中明白標示）、雲端部署。這些是取捨，不是遺漏。

## 測試

```bash
uv run pytest            # 32 個測試，全離線，不需 API key
uv run ruff check .
```

其中兩個值得一看：`test_filter_runs_before_ranking` 對任何「先檢索後過濾」的實作都會失敗；`test_a_broken_predicate_is_caught` 把權限謂詞換成全放行，確認評測真的會抓到——否則一個讓所有人看到所有文件的回歸，也會顯示 `leak_rate` 0。

---

## English summary

**scoped-rag — permission-aware RAG where the access filter runs inside retrieval, not after it.**

Enterprise knowledge bases must never surface a document the asker cannot open. The common approach — retrieve top-k, then drop what the user may not see — leaks (result counts reveal hidden documents) and ranks worse (your top k gets eaten by rows you never see). scoped-rag puts the access predicate and the vector ordering in one query, so the database filters first, ranks what survived, and then takes k.

The claim is backed by a 22-case privilege-escalation eval: `leak_rate 0.0`, `permitted_recall 1.0`, `refusal_correctness 1.0`. The eval measures access control only, deliberately: it runs on a deterministic fake embedder and fake LLM, needs no API key, and keeps "did permissions leak" separate from "is the embedding model good".

Storage supports PostgreSQL + pgvector in production and SQLite in tests, and the only differences are the vector type and the distance expression — the ACL predicate, joins, ordering and LIMIT are the same SQL on both, so the tests that prove no leakage cover the production backend too.

Stack: Python 3.12, FastAPI, pgvector, sentence-transformers (local embeddings, no API cost), Claude for generation with per-call cost accounting and a `--max-cost` budget.
