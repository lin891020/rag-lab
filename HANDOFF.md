# HANDOFF

最後更新：2026-10-28（mini）。

## 現在在哪一步

專案剛開。設計文件 `docs/superpowers/specs/2026-10-28-rag-lab-design.md` 是 Mike 在 ~/Projects 根目錄的
對話裡逐題談定的（九題：產出、範圍、結構、資料、模型、收尾、repo、節奏），**Mike 尚未整份核准**。
一個審稿 subagent 正在以「要照它開工的人」的角度讀，可能回來一輪問題。

還沒有任何實驗。模型（bge-m3、qwen3:8b）還沒拉，Mike 說等計畫核准再拉。

## 已驗證

- 骨架：`uv run pytest` 綠、`uv run pre-commit run --all-files` 綠（scaffold 自己跑的）
- mini 的 ollama 目前只有 `gpt-oss:20b`（2026-10-28 查）

## 只是寫了、還沒驗證

- CLAUDE.md、本檔、spec：都是文字，沒有可驗證的東西

## 下一步

1. 審稿 subagent 的問題回來 → 補進 spec → Mike 核准整份
2. 核准後：拉 `bge-m3`、`qwen3:8b`
3. 第一個實驗 `experiments/01-chunking/`：先 plan mode 提計畫（要查前人做法、找可合法下載的台灣電工法規 PDF），核准才寫程式

## 背景一句話

實務上這類題目會用 RAG／LLM／OCR 算配電回路規格。
對方 2026-11-02 聯絡。但這個實驗室**沒有期限**，對方聯絡與否都繼續。
