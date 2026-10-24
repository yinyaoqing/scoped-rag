# scoped-rag — 受權限約束的企業文件問答

Design doc, 2026-10-24.

## 一句話

企業知識庫的 RAG 不能把使用者無權開啟的文件洩漏出去。**scoped-rag 把權限過濾放進檢索本身，而不是檢索後再篩**，並用一組「越權檢索」評測證明它確實沒洩漏。

## 為什麼這樣做

常見作法是先做向量檢索取 top-k，再把使用者無權看的結果濾掉。這有兩個問題：

1. **安全性**：過濾後的結果數量會隨使用者權限變動，洩漏文件的存在性；若把被濾掉的內容送進提示，更是直接洩漏。
2. **品質**：「top-k 之後過濾」不等於「過濾之後取 top-k」。前者會讓有權限的使用者拿到比應得更差的結果——明明有更相關的文件，卻因為前 k 名被無權文件佔滿而取不到。

所以權限謂詞必須和向量排序在同一次查詢裡求值，由資料庫一起算。這是本專案的核心架構主張，也是唯一值得在面試裡講的句子。

## 範圍

一週時間盒。做出可展示、可量測、可離線測試的最小系統。

**做**：文件解析與切分、可插拔嵌入（本機為預設）、Postgres + pgvector 儲存、含 ACL 的權限範圍檢索、Claude 生成答案並驗證引用、最小 Web UI（含身分切換器）、CLI、越權檢索評測。

**不做**（明確砍掉，不是忘記）：知識圖譜、OCR、雲端嵌入供應商（只留介面）、re-ranking 模型、多租戶計費、真實身分驗證（demo 以身分切換器代替，文件中標明是捷徑）、雲端部署。

## 架構

```
src/scoped_rag/
  ingest/
    parse.py        檔案 → 純文字 + 位置偏移（PDF、DOCX、Markdown、TXT）
    chunk.py        文字 → 帶穩定 id 與偏移的 chunks
  embed/
    base.py         Embedder 協定：embed_documents / embed_query / dim / name
    local.py        sentence-transformers（預設，離線）
    fake.py         決定性雜湊向量（測試與評測用，無需模型）
  store/
    schema.py       DDL，兩種方言各一份
    dialect.py      方言差異只有「向量距離運算式」與型別
    repo.py         documents / chunks / embeddings / principals / grants 的存取
  retrieve/
    acl.py          權限謂詞：產生與方言無關的 SQL 片段與參數
    search.py       單次查詢完成「權限過濾 + 向量排序」
  answer/
    providers.py    LlmProvider 協定（Claude 實作；fake 實作供測試）
    generate.py     以允許的 chunks 生成答案，並驗證每個引用都在允許集合內
  api/app.py        FastAPI：上傳、提問、管理、身分切換
  web/              Jinja2 模板 + 少量原生 JS，無建置步驟
  cli.py            ingest / ask / eval / costs
evals/leakage/
  cases.yaml        越權檢索案例
  corpus/           案例用的小型文件
tests/              pytest，全離線
```

每個模組的對外介面都能獨立理解與測試：`retrieve` 不知道答案怎麼生成，`answer` 不知道權限怎麼算，`embed` 與 `store` 互不相識。

## 資料模型

語彙刻意對齊台灣企業 AI 導入商的產品（管理員／經理／員工 + 部門分組 + 個別檔案授權）。

| 表 | 欄位 | 說明 |
| --- | --- | --- |
| `principals` | id, kind（`user`\|`group`）, name | 使用者與群組同表，授權對象統一 |
| `memberships` | user_id, group_id | 使用者屬於哪些群組（部門） |
| `documents` | id, title, source, owner_id, visibility（`restricted`\|`org`）, created_at | `org` 表示全組織可讀 |
| `grants` | document_id, principal_id | 授權給個別使用者或群組。`restricted` 文件只有被授權者可讀 |
| `chunks` | id, document_id, ord, text, start, end | 偏移用於回溯原文位置 |
| `embeddings` | chunk_id, embedder, dim, vector | 記錄產生它的 embedder，換模型不會混用 |

角色（admin／manager／employee）不是另一張表，而是 `principals` 裡的群組；admin 以「可讀全部」的特例處理，並在程式中明確標示。

## 權限謂詞

一段與方言無關的 SQL：文件可讀的條件是「`visibility = 'org'`」或「存在一筆 `grants` 指向此使用者」或「存在一筆 `grants` 指向此使用者所屬的群組」。它只產生 SQL 片段與參數，不碰向量。

檢索查詢把這段謂詞放進 `WHERE`，與向量距離的 `ORDER BY` 一起交給資料庫，再取 `LIMIT k`。因此先過濾、後排序、才取 k——順序由資料庫保證，不是由應用層補救。

## 兩種後端，同一段 SQL

為了讓測試與評測在零基礎設施下跑完，儲存層支援兩個後端，**差異只有向量距離運算式與型別**：

| | 正式 | 測試／評測 |
| --- | --- | --- |
| 資料庫 | PostgreSQL + pgvector | SQLite |
| 向量型別 | `vector(n)` | BLOB（float32 陣列） |
| 距離運算式 | `embedding <=> :q` | `cosine(vector, :q)`（以 Python 註冊的自訂函式） |

ACL 謂詞、join、排序與取 k 的 SQL 完全相同，因此「權限不洩漏」這件事在兩個後端由同一組測試覆蓋。這也避免了把過濾邏輯寫兩遍而發散——那正是這類系統最容易出的錯。

預設走 SQLite，設定 `SCOPED_RAG_DSN` 指向 Postgres 時改走 pgvector。

## 嵌入與生成

- **嵌入**：`Embedder` 協定。預設 `local.py` 用 sentence-transformers 跑在本機 GPU，零 API 成本、可離線。`fake.py` 產生決定性的雜湊向量，供測試與評測使用，無需下載模型。雲端供應商只留介面，v1 不實作。
- **生成**：`LlmProvider` 協定，Claude 為唯一實作（沿用既有的 `ANTHROPIC_API_KEY`）。提示只含允許的 chunks，每段標上文件標題與 chunk id，並要求答案逐句標註引用。
- **引用驗證**：生成後逐一檢查答案引用的 chunk id 是否都在允許集合內。不在就是程式錯誤，直接失敗而非悄悄放過。這是第二道防線，形狀與 job-radar 的履歷 lint 相同。
- **拒答**：允許的 chunks 為空，或相似度都低於門檻時，回「找不到你有權限查看的相關文件」，不猜。

## 評測

`evals/leakage/cases.yaml`，每個案例是（語料、提問者、問題、期望）。三類：

1. **`must_not_retrieve`** — 語料中存在能回答問題的內容，但提問者無權讀取。期望：檢索集內**零**個禁止 chunk，且答案為拒答。
2. **`must_retrieve`** — 提問者有權讀取。期望：答案引用正確來源。這類防的是「為了安全而過度封鎖」。
3. **`boundary`** — 退出群組、撤銷授權、文件在部門間移動、授權同時來自個人與群組等邊界。

第四類：

4. **`refuses_when_nothing_permitted`** — 提問者對整個語料都無權限。期望：檢索集為空且答案為拒答。

指標：

- `leak_rate` — 禁止 chunk 進入檢索集的比例。**必須為 0**，非 0 即評測失敗。
- `permitted_recall` — 有權限案例中正確取回期望 chunk 的比例。
- `refusal_correctness` — 「該拒答時拒答、不該拒答時不拒答」的比例。

**評測刻意只衡量權限範圍，不衡量檢索品質。** 它以 `fake` embedder 與 `fake` LLM 執行：決定性、可進 CI、不需任何 API key，而且把「權限有沒有洩漏」與「嵌入模型好不好」分開——後者換模型就會變，前者是本專案的主張，必須恆為真。因此有權限案例的判定是「期望的 chunk 出現在檢索集內」，而不是「排名第一」；拒答案例則用「整個語料都無權限」來構造，不依賴相似度門檻。

門檻（`DEFAULT_MAX_DISTANCE`）是產品行為而非評測對象，所以它是呼叫參數，真實 embedder 應自行量測適合的值。

指標形狀刻意沿用 job-radar 的 `accuracy` / `blocked_recall` / `cost_usd`，兩個專案同一套紀律。

## Web UI

FastAPI + Jinja2 + 少量原生 JS，無 npm、無建置步驟（一週時間盒，且讓 repo 容易被審閱）。三個畫面：

1. **上傳文件** — 選擇授權對象（個別使用者或部門）與 `org`／`restricted`。
2. **提問** — 顯示答案與引用來源。
3. **身分切換器** — 以不同身分對同一個問題發問，看答案與可見來源的差異。

身分切換器就是整個 demo 的重點：同一個問題，經理看到答案，員工被拒答。

## 錯誤處理

- 解析失敗：記下檔案與原因，該檔不進庫，其餘繼續。不讓一個壞檔擋住整批。
- 嵌入維度不符：拒絕寫入並指出是哪個 embedder 產生的，避免混用模型的向量。
- 引用驗證失敗：視為程式錯誤，丟出例外。
- LLM 呼叫失敗：重試三次後放棄，回報失敗原因，不回退成「無引用的猜測」。
- 成本：記錄每次呼叫的 token 成本，`--max-cost` 用完即停並回報，沿用 job-radar 的作法。

## 測試

pytest，全離線。覆蓋：解析與切分（偏移正確）、ACL 謂詞（各種授權組合）、檢索（先過濾後排序、兩後端一致）、引用驗證、拒答路徑、CLI。評測另外成一組。

## 交付物

- README（繁中為主，附英文摘要），說明核心主張與如何重現評測。
- 評測報告：`leak_rate` 0、`permitted_recall` 與 `answer_accuracy` 數字。這是求職時要的「數據」。
- 一份架構決策短文：為什麼過濾要在查詢內而非查詢後，附「過濾後取 k」與「取 k 後過濾」的品質差異範例。

## 後續階段（不在 v1）

雲端嵌入供應商與成本／延遲／準確度對照表、GCP 部署與 Vertex AI、re-ranking、知識圖譜、真實身分驗證、LINE 入口（可接既有的公開專案 line-group-assistant）。
