import type { IndexStatus, PaperStatus } from "../api/types";

export type AskGate = {
  canOpenComposer: boolean;
  closeReadAvailable: boolean;
  blockedReason: string | null;
  indexHint: string | null;
  indexHintTone: "info" | "bad" | null;
};

export function paperAskAccess(input: {
  preview: boolean;
  waiting: boolean;
  status?: PaperStatus | null;
  indexStatus?: IndexStatus | null;
  indexError?: string | null;
}): AskGate {
  if (input.preview) {
    return {
      canOpenComposer: true,
      closeReadAvailable: true,
      blockedReason: null,
      indexHint: null,
      indexHintTone: null,
    };
  }
  if (input.status === "needs_ocr") {
    return hardBlock("扫描件文本过少，请换可复制文本的 PDF 后再提问。");
  }
  if (input.status === "ingest_failed") {
    return hardBlock("解析失败，无法开始问答。");
  }
  if (input.waiting) {
    return hardBlock("论文仍在解析，请稍候。");
  }
  if (input.status !== "ready") {
    return hardBlock("论文仍在解析，请稍候。");
  }

  const indexStatus = input.indexStatus ?? "pending";
  if (indexStatus === "ready") {
    return {
      canOpenComposer: true,
      closeReadAvailable: true,
      blockedReason: null,
      indexHint: null,
      indexHintTone: null,
    };
  }
  if (indexStatus === "failed") {
    const extra = input.indexError ? `：${input.indexError}` : "";
    return {
      canOpenComposer: true,
      closeReadAvailable: false,
      blockedReason: null,
      indexHint: `本篇索引失败${extra}。问这篇暂时不可用；仍可问本地库或在 arXiv 检索，也可重建索引。`,
      indexHintTone: "bad",
    };
  }
  if (indexStatus === "skipped") {
    return {
      canOpenComposer: true,
      closeReadAvailable: false,
      blockedReason: null,
      indexHint: "这篇没有可检索正文，无法精读提问。仍可问本地库或在 arXiv 检索相关论文。",
      indexHintTone: "info",
    };
  }
  return {
    canOpenComposer: true,
    closeReadAvailable: false,
    blockedReason: null,
    indexHint: "本篇检索索引还在建，问这篇请稍候。问本地库或在 arXiv 检索不受影响。",
    indexHintTone: "info",
  };
}

function hardBlock(reason: string): AskGate {
  return {
    canOpenComposer: false,
    closeReadAvailable: false,
    blockedReason: reason,
    indexHint: null,
    indexHintTone: null,
  };
}
