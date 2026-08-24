import { useCallback, useEffect, useRef, useState, type CSSProperties, type MouseEvent as ReactMouseEvent } from "react";

import {
  formatApiError,
  getFigures,
  getIntro,
  getMethod,
  getPaper,
  getSections,
  refreshIntro,
  refreshMethod,
  reparsePaper,
  sourcePdfUrl,
  uploadPaper,
} from "../api/client";
import { HttpError } from "../api/types";
import type { Figure, MethodExplain, Paper, PaperIntro, Section } from "../api/types";
import { demoFigures, demoIntro, demoMethod, demoPaper, demoSections } from "../data/demo";
import { Lightbox } from "./FigureViews";
import { IntroPanel } from "./IntroPanel";
import { Outline } from "./Outline";
import { PdfPane, type SourceView } from "./PdfPane";
import { LogoMark, RefreshIcon, TreeIcon } from "./icons";
import { MethodPanel } from "./MethodPanel";

export type WorkspaceProps = {
  file: File | null;
  preview: boolean;
  onReset: () => void;
  onLoadPreview: () => void;
};

type Tab = "intro" | "method";

type Chip = { label: string; tone: "ok" | "run" | "warn" | "bad" | "mute" };

export function Workspace({ file, preview, onReset, onLoadPreview }: WorkspaceProps) {
  const objectUrl = useRef<string | null>(file ? URL.createObjectURL(file) : null);
  const [outlineOn, setOutlineOn] = useState(true);
  const [analysisW, setAnalysisW] = useState("42%");
  const [dragging, setDragging] = useState(false);
  const [tab, setTab] = useState<Tab>("intro");
  const [sourceView, setSourceView] = useState<SourceView>("sections");
  const [page, setPage] = useState(1);
  const [paperId, setPaperId] = useState<string | null>(preview ? "demo" : null);
  const [paper, setPaper] = useState<Paper | null>(preview ? demoPaper : null);
  const [sections, setSections] = useState<Section[]>(preview ? demoSections : []);
  const [figures, setFigures] = useState<Figure[]>(preview ? demoFigures : []);
  const [intro, setIntro] = useState<PaperIntro | null>(preview ? demoIntro : null);
  const [method, setMethod] = useState<MethodExplain | null>(preview ? demoMethod : null);
  const [notice, setNotice] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<Figure | undefined>();
  const [uploading, setUploading] = useState(false);
  const [docPages, setDocPages] = useState(0);
  const [focusSection, setFocusSection] = useState<string | null>(null);
  const [ingestGen, setIngestGen] = useState(0);

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
    if (preview || !paperId) return;
    let cancelled = false;
    let timer = 0;

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
        if (next.intro_status === "ready" || next.intro_status === "partial") {
          try {
            setIntro(await getIntro(paperId));
          } catch (error) {
            if (!(error instanceof HttpError && error.status === 202)) throw error;
          }
        }
        if (next.method_status === "ready" || next.method_status === "partial") {
          try {
            setMethod(await getMethod(paperId));
          } catch (error) {
            if (!(error instanceof HttpError && error.status === 202)) throw error;
          }
        }
        const ingesting = next.status === "queued" || next.status === "parsing";
        const done =
          ["ready", "needs_ocr", "ingest_failed"].includes(next.status) &&
          next.intro_status !== "pending" &&
          next.method_status !== "pending";
        if (!done) timer = window.setTimeout(() => void tick(), ingesting ? 5000 : 1600);
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
    setDragging(true);
    const onMove = (move: MouseEvent) => {
      const width = window.innerWidth - move.clientX;
      const pct = Math.min(58, Math.max(32, (width / window.innerWidth) * 100));
      setAnalysisW(`${pct}%`);
    };
    const onUp = () => {
      setDragging(false);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, []);

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

  async function onRefresh() {
    if (preview || !paperId) return;
    try {
      if (tab === "intro") setIntro(await refreshIntro(paperId));
      else setMethod(await refreshMethod(paperId));
    } catch (error) {
      setNotice(formatApiError(error));
    }
  }

  const chips = statusChips(paper, uploading, preview, notice);
  const waiting =
    !preview &&
    !notice &&
    (uploading || !paper || paper.status === "queued" || paper.status === "parsing");

  const paperTitle = paper?.title || file?.name || paper?.filename || null;

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" type="button" onClick={onReset}>
          <span className="brand-mark">
            <LogoMark />
          </span>
          <span className="brand-name">论文理解</span>
        </button>
        <div className="topbar-file">
          <span>{paper?.title || file?.name || paper?.filename || "未命名论文"}</span>
        </div>
        <div className="topbar-actions">
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
          <button className="ghost-btn" type="button" onClick={onReset}>
            换一篇
          </button>
        </div>
      </header>
      <div className={`workspace${outlineOn ? "" : " is-outline-off"}`} style={{ "--analysis": analysisW } as CSSProperties}>
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
              <button className={tab === "intro" ? "is-on" : ""} type="button" onClick={() => setTab("intro")}>
                总体介绍
              </button>
              <button className={tab === "method" ? "is-on" : ""} type="button" onClick={() => setTab("method")}>
                方法详解
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
            <button
              className="ghost-btn"
              type="button"
              disabled={preview || !paperId}
              onClick={() => void onRefresh()}
            >
              <RefreshIcon />
              刷新
            </button>
          </div>
          {waiting ? <div className="progress-track"><div className="progress-bar" /></div> : null}
          <div className="analysis-body">
            {notice && !preview ? (
              <div className={`callout${/M1|M2|实现|ingest/.test(notice) ? " is-info" : " is-bad"}`}>
                <div>{notice}。左侧仍可阅读刚上传的 PDF。</div>
                <button className="ghost-btn" type="button" onClick={onLoadPreview} style={{ marginTop: 8, color: "inherit" }}>
                  查看讲解界面预览
                </button>
              </div>
            ) : null}
            {paper?.status === "needs_ocr" ? (
              <div className="callout is-bad">扫描件文本过少，请换可复制文本的 PDF。不会生成介绍或方法。</div>
            ) : null}
            {waiting && !notice ? (
              <div className="skeleton">
                <div className="skel" style={{ width: "70%", height: 22 }} />
                <div className="skel" style={{ width: "100%" }} />
                <div className="skel" style={{ width: "92%" }} />
                <div className="skel" style={{ width: "84%" }} />
                <div className="skel" style={{ width: "60%", marginTop: 18 }} />
              </div>
            ) : null}
            {intro || method || sections.length || (!waiting && !notice) ? (
              tab === "intro" ? (
              <IntroPanel
                intro={intro}
                sections={sections}
                figures={figures}
                paperId={preview ? "demo" : paperId}
                preview={preview}
                onJump={jumpToPage}
                onOpenFigure={setLightbox}
                onJumpSectionTitle={(title) => {
                  const hit = sections.find((section) => section.title === title);
                  if (hit) jumpToSection(hit.page_start, hit.section_id);
                }}
              />
              ) : (
              <MethodPanel
                method={method}
                sections={sections}
                figures={figures}
                paperId={preview ? "demo" : paperId}
                preview={preview}
                focusSectionId={focusSection}
                onJump={jumpToPage}
                onOpenFigure={setLightbox}
              />
              )
            ) : null}
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
      { label: "介绍就绪", tone: "ok" },
      { label: "方法就绪", tone: "ok" },
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
    chips.push(explainChip("介绍", paper.intro_status));
    chips.push(explainChip("方法", paper.method_status));
  } else if (notice) {
    chips.push({ label: "等待解析接口", tone: "warn" });
  }
  return chips;
}

function explainChip(name: string, status: Paper["intro_status"]): Chip {
  if (status === "ready") return { label: `${name}就绪`, tone: "ok" };
  if (status === "partial") return { label: `${name}部分`, tone: "warn" };
  if (status === "failed") return { label: `${name}失败`, tone: "bad" };
  if (status === "skipped") return { label: `${name}跳过`, tone: "mute" };
  return { label: `${name}生成中`, tone: "run" };
}
