# rag-lab

一句話說這個專案做什麼。

## 怎麼跑

```bash
uv sync
uv run pytest
uv run python -m rag_lab
```

## 這個專案獨有的

- 資料在哪、要不要先下載
- port：（要開 port 先查 ~/Projects/PORTS.md 並登記）
- 禁區：不能動的檔案或不能做的事

## 約定（家裡層已經有的，這裡只寫差異）

- log 照 `logging-conventions` skill；進入點呼叫 `setup_logging("rag_lab")`
- 收工更新 `HANDOFF.md`（`handoff` skill）
- 新功能先 spec（`docs/superpowers/specs/`），核准後才寫 `src/`
- 頁面：新頁或整頁改版用 `frontend-design`；推銷頁／作品集可改用 `taste-skill`，兩者不同時用；既有頁面變好看用 `polishing-pages`；`minimalist-ui` 只在叫名字時。有 `DESIGN.md` 就先讀它。寫完 hook 會自動跑 avoid-ai-design，再跑 `polishing-pages/scripts/measure.mjs`
- 說「做完了」之前先跑 `superpowers:verification-before-completion`，再寫 HANDOFF

## 寫程式之前，照這個順序問

1. 這需要存在嗎？不需要就不寫。
2. 專案裡已經有了嗎？重用。
3. 標準庫做得到嗎？用標準庫。
4. 已裝的套件做得到嗎？用它，不加新依賴。
5. 一行能寫完嗎？一行。
6. 都不行，才寫最少的可動程式碼。

不砍的：log、例外處理、測試、邊界檢查。

## 不要做的

- 不要在這裡寫實驗紀錄、數字、歷史；那些放 `docs/`。這個檔超過 60 行就該搬東西出去。
