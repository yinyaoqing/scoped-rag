# 為什麼權限過濾要在查詢內，而不是查詢後

決策日期：2026-10-24。狀態：已採用。

## 決策

檢索時，權限謂詞與向量距離排序放進**同一次 SQL 查詢**，由資料庫先過濾、再排序、才取 `LIMIT k`。不採用「先取 top-k，再把無權查看的結果濾掉」。

## 兩種作法的差別

```sql
-- 採用：過濾在查詢內
SELECT c.id, ..., (e.vector <=> :q) AS distance
FROM chunks c
JOIN documents d ON d.id = c.document_id
JOIN embeddings e ON e.chunk_id = c.id
WHERE e.embedder = :embedder
  AND (d.visibility = 'org'
       OR EXISTS (SELECT 1 FROM grants g
                  WHERE g.document_id = d.id AND g.principal_id = :asker)
       OR EXISTS (SELECT 1 FROM grants g JOIN memberships m ON m.group_id = g.principal_id
                  WHERE g.document_id = d.id AND m.user_id = :asker))
ORDER BY distance ASC
LIMIT :k;
```

後過濾則是：同一句查詢去掉 `AND (...)`，取回 k 筆，再在應用層丟掉無權查看的列。

## 理由一：後過濾會洩漏

被丟掉的那幾列已經離開資料庫、進了應用程式的記憶體。只要有一處疏漏——記錄日誌、錯誤回報、把原始結果一起送進提示、或單純是未來某次重構——內容就外流了。更細微的是，**結果數量本身就是訊息**：同一個問題，有權限的人拿到 5 筆、無權限的人拿到 3 筆，那 2 筆的差額就告訴後者「有兩份我看不到的文件談到這件事」。對人資調查紀錄或併購文件這類資料，光是「存在」就是機密。

過濾在查詢內時，無權查看的列從不離開資料庫，結果數量也只反映提問者自己的語料。

## 理由二：後過濾的排序品質更差

「top-k 之後過濾」不等於「過濾之後取 top-k」。前者讓有權限的使用者去跟他看不到的文件競爭 k 個名額。

在本專案的範例語料上實測（`FakeEmbedder`，`k=3`）：

| 提問者 | 可讀文件 | 先檢索後過濾 | 過濾在查詢內 |
| --- | --- | --- | --- |
| `u:alice`（業務部員工）問調薪級距 | handbook, leave, sales | **2/3 筆** — handbook(0.946), sales(1.049) | **3/3 筆** — handbook(0.946), sales(1.049), leave(1.073) |
| `u:bob`（經理）問申訴人身分 | handbook, leave, salary | **2/3 筆** — leave(0.948), salary(0.982) | **3/3 筆** — leave(0.948), salary(0.982), handbook(1.293) |

兩個案例都損失三分之一的脈絡額度。這不是誤差，是結構性的：無權查看的列佔掉名額之後，就沒有任何下游步驟能把它們補回來。把 k 調大可以緩解，但緩解的方式是多讀無權查看的列，等於把理由一的風險放大。

## 代價

- 權限謂詞變成每次檢索查詢的一部分，索引要照顧到它（`grants(document_id)`、`memberships(user_id)`）。
- 近似最近鄰索引與選擇性過濾的互動需要注意：pgvector 的 HNSW 在高選擇性過濾下可能需要調 `hnsw.ef_search`，或改用部分索引。本專案語料小，尚未需要處理；量大時這會是第一個要量測的地方。
- 權限邏輯必須存在於 SQL 層，因此它必須是與方言無關的——這也是儲存層只允許「向量型別」與「距離運算式」兩處差異的原因。把這段謂詞寫兩遍，兩份就會發散，而發散的那一刻保證就失效了。

## 怎麼確認它還成立

- `tests/test_scoped_search.py::test_filter_runs_before_ranking` — 對任何先檢索後過濾的實作都會失敗。
- `tests/test_scoped_search.py::test_result_count_does_not_reveal_hidden_documents` — 檢驗理由一的數量洩漏。
- `tests/test_eval.py::test_a_broken_predicate_is_caught` — 把謂詞換成全放行，確認評測會抓到；否則「`leak_rate` 0」可能只是因為評測本身失效了。
