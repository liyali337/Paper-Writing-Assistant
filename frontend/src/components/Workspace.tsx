import { useCallback, useEffect, useRef, useState, type CSSProperties, type MouseEvent as ReactMouseEvent } from "react";

import {
  formatApiError,
  getFigures,
  getPaper,
  getSections,
  rebuildPaperIndex,
  reparsePaper,
  sourcePdfUrl,
  uploadPaper,
} from "../api/client";
import { HttpError } from "../api/types";
import type { Figure, Paper, Section } from "../api/types";
import { demoFigures, demoPaper, demoSections } from "../data/demo";
import { AskPanel, type AskMessage } from "./AskPanel";
import { Lightbox } from "./FigureViews";
import { Outline } from "./Outline";
import { PdfPane, type SourceView } from "./PdfPane";
import { AppHeader } from "./AppHeader";
import { RefreshIcon, TreeIcon } from "./icons";

export type WorkspaceProps = {
  file: File | null;
  preview: boolean;
  paperId?: string;
  onReset: () => void;
  onLoadPreview: () => void;
};

type Chip = { label: string; tone: "ok" | "run" | "warn" | "bad" | "mute" };

type ReturnFrame = { paperId: string; messages: AskMessage[] };

export function Workspace({
  file,
  preview,
  paperId: initialPaperId,
  onReset,
  onLoadPreview,
}: WorkspaceProps) {
  const objectUrl = useRef<string | null>(file ? URL.createObjectURL(file) : null);
  const workspaceRef = useRef<HTMLDivElement>(null);
  const [outlineOn, setOutlineOn] = useState(true);
  const [analysisW, setAnalysisW] = useState("42%");
  const [dragging, setDragging] = useState(false);
  const [sourceView, setSourceView] = useState<SourceView>("sections");
  const [page, setPage] = useState(1);
  const [paperId, setPaperId] = useState<string | null>(preview ? "demo" : initialPaperId ?? null);
  const [paper, setPaper] = useState<Paper | null>(preview ? demoPaper : null);
  const [sections, setSections] = useState<Section[]>(preview ? demoSections : []);
  const [figures, setFigures] = useState<Figure[]>(preview ? demoFigures : []);
  const [notice, setNotice] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<Figure | undefined>();
  const [uploading, setUploading] = useState(false);
  const [docPages, setDocPages] = useState(0);
  const [focusSection, setFocusSection] = useState<string | null>(null);
  const [ingestGen, setIngestGen] = useState(0);
  const indexKick = useRef(false);
  const restoringRef = useRef(false);
  const [askMessages, setAskMessages] = useState<AskMessage[]>([]);
  const [returnStack, setReturnStack] = useState<ReturnFrame[]>([]);

  useEffect(() => {
    return () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, []);

  useEffect(() => {
    if (preview || !file) return;
    let cancelled = false;
    setUploading(true);
    uploadPaper(file)
      .then((result) => {
        if (!cancelled) setPaperId(result.paper_id);
      })
      .catch((error) => {
        if (!cancelled) setNotice(formatApiError(error));
      })
      .finally(() => {
        if (!cancelled) setUploading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [file, preview]);

  useEffect(() => {
    if (restoringRef.current) {
      restoringRef.current = false;
      return;
    }
    setAskMessages([]);
  }, [paperId, preview]);

  useEffect(() => {
    if (preview || !paperId) return;
    let cancelled = false;
    let timer = 0;
    indexKick.current = false;

    const tick = async () => {
      try {
        const next = await getPaper(paperId);
        if (cancelled) return;
        setPaper(next);
        if (next.status === "ready" || next.status === "needs_ocr") {
          const [sec, fig] = await Promise.all([getSections(paperId), getFigures(paperId)]);
          if (cancelled) return;
          setSections(Array.isArray(sec) ? sec : []);
          setFigures(Array.isArray(fig) ? fig : []);
        }
        const ingesting = next.status === "queued" || next.status === "parsing";
        const indexing =
          next.status === "ready" && (next.index_status === "pending" || !next.index_status);
        if (indexing && !indexKick.current) {
          indexKick.current = true;
          try {
            await rebuildPaperIndex(paperId);
          } catch {
            indexKick.current = false;
          }
          if (cancelled) return;
        }
        const done =
          next.status === "needs_ocr" ||
          next.status === "ingest_failed" ||
          (next.status === "ready" && next.index_status != null && next.index_status !== "pending");
        if (!done) timer = window.setTimeout(() => void tick(), ingesting || indexing ? 4000 : 1600);
      } catch (error) {
        if (cancelled) return;
        if (error instanceof HttpError && error.status === 501) {
          setNotice(formatApiError(error));
          return;
        }
        timer = window.setTimeout(() => void tick(), 2400);
      }
    };

    void tick();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [paperId, preview, ingestGen]);

  const onSplit = useCallback((event: ReactMouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    const root = workspaceRef.current;
    if (!root) return;
    setDragging(true);
    const onMove = (move: MouseEvent) => {
      const rect = root.getBoundingClientRect();
      if (rect.width <= 0) return;
      const outline = outlineOn
        ? root.querySelector(".outline")?.getBoundingClientRect().width ?? 248
        : 0;
      const minAnalysis = 280;
      const minSource = 360;
      const splitterW = 7;
      const maxAnalysis = Math.max(minAnalysis, rect.width - outline - splitterW - minSource);
      const next = Math.min(maxAnalysis, Math.max(minAnalysis, rect.right - move.clientX));
      setAnalysisW(`${(next / rect.width) * 100}%`);
    };
    const onUp = () => {
      setDragging(false);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [outlineOn]);

  const pageCount = Math.max(
    paper?.page_count || 0,
    docPages,
    sections.reduce((max, section) => Math.max(max, section.page_end), 0),
    1,
  );

  const syncPage = useCallback(
    (next: number) => {
      setPage(Math.min(pageCount, Math.max(1, next)));
    },
    [pageCount],
  );

  const jumpToPage = useCallback(
    (next: number) => {
      syncPage(next);
      setSourceView("pdf");
    },
    [syncPage],
  );

  const jumpToSection = useCallback(
    (next: number, sectionId: string) => {
      syncPage(next);
      setFocusSection(sectionId);
      setSourceView("sections");
    },
    [syncPage],
  );

  const onAskJump = useCallback(
    (next: number, sectionId?: string) => {
      if (sectionId) jumpToSection(next, sectionId);
      else jumpToPage(next);
    },
    [jumpToPage, jumpToSection],
  );

  const onPageCount = useCallback((count: number) => {
    setDocPages(count);
  }, []);

  useEffect(() => {
    if (!focusSection || sourceView !== "sections") return;
    const node = document.getElementById(`sec-${focusSection}`);
    const scroller = node?.closest(".source-scroll");
    if (!node || !(scroller instanceof HTMLElement)) return;
    const top = node.getBoundingClientRect().top - scroller.getBoundingClientRect().top + scroller.scrollTop - 12;
    scroller.scrollTo({ top, behavior: "smooth" });
  }, [focusSection, sourceView]);

  async function onReparse() {
    if (preview || !paperId) return;
    try {
      setNotice(null);
      await reparsePaper(paperId);
      setPaper((current) => (current ? { ...current, status: "queued" } : current));
      setSections([]);
      setFigures([]);
      setIngestGen((value) => value + 1);
    } catch (error) {
      setNotice(formatApiError(error));
    }
  }

  function openLibraryPaper(nextId: string) {
    if (!nextId || nextId === paperId) return;
    if (paperId) {
      setReturnStack((current) => [...current, { paperId, messages: askMessages }]);
    }
    setPage(1);
    setFocusSection(null);
    setAskMessages([]);
    setPaperId(nextId);
  }

  function onLeavePaper() {
    const prev = returnStack[returnStack.length - 1];
    if (!prev) {
      onReset();
      return;
    }
    restoringRef.current = true;
    setReturnStack((current) => current.slice(0, -1));
    setPage(1);
    setFocusSection(null);
    setPaperId(prev.paperId);
    setAskMessages(prev.messages);
  }

  const chips = statusChips(paper, uploading, preview, notice);
  const waiting =
    !preview &&
    !notice &&
    (uploading || !paper || paper.status === "queued" || paper.status === "parsing");

  const paperTitle = paper?.title || file?.name || paper?.filename || null;
  const askReady = preview || (!waiting && paper?.status === "ready");
  const askBlocked =
    paper?.status === "needs_ocr"
      ? "扫描件文本过少，请换可复制文本的 PDF 后再提问。"
      : paper?.status === "ingest_failed"
        ? "解析失败，无法开始问答。"
        : waiting
          ? "论文仍在解析，请稍候。"
          : null;
  const askHint =
    paper?.status === "ready" && (paper.index_status === "pending" || !paper.index_status)
      ? "本篇精读索引还在建。可以先问本地库或在 arXiv 上检索；问这篇原文请稍候。"
      : paper?.status === "ready" && paper.index_status === "failed"
        ? `本篇精读索引失败${paper.index_error ? `：${paper.index_error}` : ""}。仍可问本地库或 arXiv，或点重建索引。`
        : paper?.status === "ready" && paper.index_status === "skipped"
          ? "这篇没有可检索的正文，无法精读提问；仍可问本地库或在 arXiv 上检索。"
          : null;

  return (
    <div className="app-shell">
      <AppHeader
        onBrand={onReset}
        center={<span>{paper?.title || file?.name || paper?.filename || "未命名论文"}</span>}
        actions={
          <>
            {chips.map((chip) => (
              <span key={chip.label} className={`chip is-${chip.tone}`}>
                <i />
                {chip.label}
              </span>
            ))}
            <button
              className="ghost-btn"
              type="button"
              disabled={preview || !paperId || uploading || waiting}
              onClick={() => void onReparse()}
            >
              <RefreshIcon />
              重跑解析
            </button>
            <button className="ghost-btn" type="button" onClick={onLeavePaper}>
              {returnStack.length ? "返回上一篇" : "换一篇"}
            </button>
          </>
        }
      />
      <div
        ref={workspaceRef}
        className={`workspace${outlineOn ? "" : " is-outline-off"}`}
        style={{ "--analysis": analysisW } as CSSProperties}
      >
        <aside className="pane outline">
          <div className="outline-head">目录</div>
          <Outline
            sections={sections}
            activePage={page}
            onJump={(page, sectionId) => {
              if (sectionId) jumpToSection(page, sectionId);
            }}
          />
        </aside>
        <PdfPane
          view={sourceView}
          onView={setSourceView}
          pdfUrl={preview ? null : paperId ? sourcePdfUrl(paperId) : objectUrl.current}
          sections={sections}
          figures={figures}
          paperId={preview ? "demo" : paperId}
          paperTitle={paperTitle}
          paperAuthors={paper?.authors ?? []}
          preview={preview}
          focusSection={focusSection}
          page={page}
          pageCount={pageCount}
          onPage={syncPage}
          onOpenFigure={setLightbox}
          onPageCount={onPageCount}
        />
        <div
          className={`splitter${dragging ? " is-on" : ""}`}
          onMouseDown={onSplit}
          role="separator"
          aria-orientation="vertical"
        />
        <section className="pane analysis">
          <div className="analysis-head">
            <div className="tabs">
              <button className="is-on" type="button">
                问答
              </button>
            </div>
            <button
              className="icon-btn"
              type="button"
              aria-label="目录"
              onClick={() => setOutlineOn((value) => !value)}
            >
              <TreeIcon />
            </button>
          </div>
          {waiting || (!preview && paper?.status === "ready" && (paper.index_status === "pending" || !paper.index_status)) ? (
            <div className="progress-track">
              <div className="progress-bar" />
            </div>
          ) : null}
          <div className="analysis-body is-ask">
            {notice && !preview ? (
              <div className={`callout${/M1|M2|实现|ingest/.test(notice) ? " is-info" : " is-bad"}`} style={{ margin: "16px 16px 0" }}>
                <div>{notice}。左侧仍可阅读刚上传的 PDF。</div>
                <button className="ghost-btn" type="button" onClick={onLoadPreview} style={{ marginTop: 8, color: "inherit" }}>
                  查看问答界面预览
                </button>
              </div>
            ) : null}
            <AskPanel
              paperId={preview ? "demo" : paperId}
              preview={preview}
              ready={Boolean(askReady)}
              blockedReason={askBlocked}
              indexHint={askHint}
              indexStatus={preview ? "ready" : paper?.index_status ?? "pending"}
              figures={figures}
              messages={askMessages}
              onMessages={setAskMessages}
              onJump={onAskJump}
              onOpenFigure={setLightbox}
              onOpenPaper={preview ? undefined : openLibraryPaper}
              onRebuildIndex={
                preview || !paperId
                  ? undefined
                  : async () => {
                      await rebuildPaperIndex(paperId);
                      setIngestGen((value) => value + 1);
                    }
              }
            />
          </div>
        </section>
      </div>
      <Lightbox
        figure={lightbox}
        paperId={preview ? "demo" : paperId}
        preview={preview}
        onClose={() => setLightbox(undefined)}
      />
    </div>
  );
}

function statusChips(
  paper: Paper | null,
  uploading: boolean,
  preview: boolean,
  notice: string | null,
): Chip[] {
  if (preview) {
    return [
      { label: "界面预览", tone: "run" },
      { label: "问答示意", tone: "ok" },
    ];
  }
  const chips: Chip[] = [];
  if (uploading) chips.push({ label: "上传中", tone: "run" });
  if (paper) {
    chips.push({
      label:
        paper.status === "ready"
          ? `已分解 · ${paper.figure_count} 图`
          : paper.status === "needs_ocr"
            ? "需要文本型 PDF"
            : paper.status === "ingest_failed"
              ? "解析失败"
              : "解析中",
      tone:
        paper.status === "ready"
          ? "ok"
          : paper.status === "needs_ocr" || paper.status === "ingest_failed"
            ? "bad"
            : "run",
    });
    if (paper.translate_status === "ready" || paper.translate_status === "partial") {
      chips.push({
        label: paper.translate_status === "partial" ? "译文部分" : "译文就绪",
        tone: paper.translate_status === "partial" ? "warn" : "ok",
      });
    }
    chips.push({ label: "问答待接入", tone: "mute" });
  } else if (notice) {
    chips.push({ label: "等待解析接口", tone: "warn" });
  }
  return chips;
}
