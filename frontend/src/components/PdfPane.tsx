import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PDFWorker, getDocument, type PDFDocumentProxy, type RenderTask } from "pdfjs-dist";
import PdfJsWorker from "pdfjs-dist/build/pdf.worker.min.mjs?worker";

import { ensureTranslations, formatApiError } from "../api/client";
import type { Figure, Section, SectionTranslation } from "../api/types";
import { demoTranslations } from "../data/demo";
import { DemoDiagram } from "./DemoDiagram";
import { Chevron } from "./icons";
import { SectionReader, type LangMode } from "./SectionReader";

export type SourceView = "pdf" | "sections";

type Props = {
  view: SourceView;
  onView: (view: SourceView) => void;
  pdfUrl: string | null;
  sections: Section[];
  figures: Figure[];
  paperId: string | null;
  paperTitle: string | null;
  preview: boolean;
  focusSection: string | null;
  page: number;
  pageCount: number;
  onPage: (page: number) => void;
  onOpenFigure: (figure: Figure) => void;
  onPageCount?: (count: number) => void;
};

export function PdfPane({
  view,
  onView,
  pdfUrl,
  sections,
  figures,
  paperId,
  paperTitle,
  preview,
  focusSection,
  page,
  pageCount,
  onPage,
  onOpenFigure,
  onPageCount,
}: Props) {
  const safeCount = Math.max(pageCount, 1);
  const [showEn, setShowEn] = useState(true);
  const [showZh, setShowZh] = useState(true);
  const [translations, setTranslations] = useState<Map<string, SectionTranslation>>(new Map());
  const [translating, setTranslating] = useState(false);
  const [translateError, setTranslateError] = useState<string | null>(null);

  const lang = useMemo<LangMode>(() => ({ showEn, showZh }), [showEn, showZh]);

  const toggleEn = useCallback(() => {
    setShowEn((current) => {
      if (current && !showZh) return current;
      return !current;
    });
  }, [showZh]);

  const toggleZh = useCallback(() => {
    setShowZh((current) => {
      if (current && !showEn) return current;
      return !current;
    });
  }, [showEn]);

  useEffect(() => {
    if (!showZh) {
      setTranslating(false);
      return;
    }
    if (preview) {
      setTranslations(new Map(demoTranslations.map((item) => [item.section_id, item])));
      setTranslateError(null);
      setTranslating(false);
      return;
    }
    if (!paperId || sections.length === 0) return;

    let cancelled = false;
    setTranslating(true);
    setTranslateError(null);
    ensureTranslations(paperId)
      .then((payload) => {
        if (cancelled) return;
        setTranslations(new Map(payload.sections.map((item) => [item.section_id, item])));
      })
      .catch((error) => {
        if (cancelled) return;
        setTranslateError(formatApiError(error));
      })
      .finally(() => {
        if (!cancelled) setTranslating(false);
      });

    return () => {
      cancelled = true;
    };
  }, [paperId, preview, sections.length, showZh]);

  return (
    <section className="pane pdf-pane">
      <div className="pdf-toolbar">
        <div className="tabs source-tabs">
          <button className={view === "pdf" ? "is-on" : ""} type="button" onClick={() => onView("pdf")}>
            原PDF
          </button>
          <button
            className={view === "sections" ? "is-on" : ""}
            type="button"
            onClick={() => onView("sections")}
          >
            章节原文
          </button>
        </div>
        {view === "sections" ? (
          <div className="lang-toggles">
            <button
              className={showEn ? "is-on" : ""}
              type="button"
              aria-pressed={showEn}
              onClick={toggleEn}
            >
              英文
            </button>
            <button
              className={showZh ? "is-on" : ""}
              type="button"
              aria-pressed={showZh}
              onClick={toggleZh}
            >
              中文
            </button>
            {translating ? <span className="lang-status">翻译中…</span> : null}
          </div>
        ) : null}
        <span className="grow" />
        {view === "pdf" ? (
          <div className="page-nav">
            <button
              className="icon-btn"
              type="button"
              aria-label="上一页"
              disabled={page <= 1}
              onClick={() => onPage(page - 1)}
            >
              <Chevron dir="left" />
            </button>
            <input
              aria-label="页码"
              value={page}
              onChange={(event) => {
                const next = Number(event.target.value);
                if (Number.isFinite(next)) onPage(next);
              }}
            />
            <span style={{ color: "var(--faint)", fontSize: 12 }}>/ {safeCount}</span>
            <button
              className="icon-btn"
              type="button"
              aria-label="下一页"
              disabled={page >= safeCount}
              onClick={() => onPage(page + 1)}
            >
              <Chevron dir="right" />
            </button>
          </div>
        ) : null}
      </div>
      {view === "pdf" ? (
        pdfUrl ? (
          <PdfPages pdfUrl={pdfUrl} page={page} onPage={onPage} onPageCount={onPageCount} />
        ) : (
          <PaperMock sections={sections} page={page} />
        )
      ) : (
        <div className="source-scroll">
          <SectionReader
            sections={sections}
            figures={figures}
            paperId={paperId}
            preview={preview}
            paperTitle={paperTitle}
            focusId={focusSection}
            lang={lang}
            translations={translations}
            translating={translating}
            translateError={translateError}
            onJump={(nextPage) => {
              onPage(nextPage);
              onView("pdf");
            }}
            onOpenFigure={onOpenFigure}
          />
        </div>
      )}
    </section>
  );
}

function PdfPages({
  pdfUrl,
  page,
  onPage,
  onPageCount,
}: {
  pdfUrl: string;
  page: number;
  onPage: (page: number) => void;
  onPageCount?: (count: number) => void;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const jumping = useRef(false);
  const [doc, setDoc] = useState<PDFDocumentProxy | null>(null);
  const [nativeSrc, setNativeSrc] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let loaded: PDFDocumentProxy | null = null;
    let worker: PDFWorker | null = null;
    let port: Worker | null = null;
    setDoc(null);
    setNativeSrc(null);
    const abort = new AbortController();

    const run = async () => {
      try {
        const response = await fetch(pdfUrl, { signal: abort.signal });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = new Uint8Array(await response.arrayBuffer());
        if (cancelled) return;
        port = new PdfJsWorker();
        worker = PDFWorker.create({ port, name: "paper-pdf" });
        const next = await getDocument({ data, worker }).promise;
        loaded = next;
        if (cancelled) {
          void next.cleanup();
          return;
        }
        setDoc(next);
        onPageCount?.(next.numPages);
      } catch (error) {
        if (cancelled || abort.signal.aborted) return;
        console.error("pdf.js failed to open document", error);
        setNativeSrc(pdfUrl);
      }
    };

    void run();
    return () => {
      cancelled = true;
      abort.abort();
      worker?.destroy();
      port?.terminate();
      void loaded?.cleanup();
    };
  }, [pdfUrl, onPageCount]);

  useEffect(() => {
    if (!doc) return;
    jumping.current = true;
    const node = rootRef.current?.querySelector(`[data-pdf-page="${page}"]`);
    node?.scrollIntoView({ block: "start" });
    const timer = window.setTimeout(() => {
      jumping.current = false;
    }, 280);
    return () => window.clearTimeout(timer);
  }, [doc, page]);

  useEffect(() => {
    const root = rootRef.current;
    if (!root || !doc) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (jumping.current) return;
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        const next = Number((visible?.target as HTMLElement | undefined)?.dataset.pdfPage);
        if (Number.isFinite(next) && next > 0) onPage(next);
      },
      { root, threshold: 0.45 },
    );
    root.querySelectorAll("[data-pdf-page]").forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, [doc, onPage]);

  if (nativeSrc) {
    return <iframe className="pdf-frame" title="原 PDF" src={nativeSrc} />;
  }

  if (!doc) {
    return (
      <div className="pdf-scroll">
        <div className="pdf-page is-loading">正在打开 PDF…</div>
      </div>
    );
  }

  return (
    <div className="pdf-scroll" ref={rootRef}>
      {Array.from({ length: doc.numPages }, (_, index) => (
        <PdfPageCanvas key={index + 1} doc={doc} pageNumber={index + 1} />
      ))}
    </div>
  );
}

function PdfPageCanvas({ doc, pageNumber }: { doc: PDFDocumentProxy; pageNumber: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    let cancelled = false;
    let lastWidth = 0;
    let renderTask: RenderTask | null = null;

    const draw = async () => {
      const width = Math.max(240, wrap.clientWidth);
      if (Math.abs(width - lastWidth) < 2 && canvas.height) return;
      lastWidth = width;
      const pdfPage = await doc.getPage(pageNumber);
      if (cancelled) return;
      renderTask?.cancel();
      const base = pdfPage.getViewport({ scale: 1 });
      const scale = width / base.width;
      const viewport = pdfPage.getViewport({ scale });
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(viewport.width * dpr);
      canvas.height = Math.floor(viewport.height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${Math.floor(viewport.height)}px`;
      renderTask = pdfPage.render({
        canvas,
        viewport,
        transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : undefined,
        background: "#ffffff",
      });
      try {
        await renderTask.promise;
      } catch {
        /* 尺寸变化会取消上一帧渲染 */
      }
    };

    void draw();
    const observer = new ResizeObserver(() => {
      void draw();
    });
    observer.observe(wrap);
    return () => {
      cancelled = true;
      renderTask?.cancel();
      observer.disconnect();
    };
  }, [doc, pageNumber]);

  return (
    <div className="pdf-page" data-pdf-page={pageNumber} ref={wrapRef}>
      <canvas ref={canvasRef} />
    </div>
  );
}

function PaperMock({ sections, page }: { sections: Section[]; page: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const max = sections.length ? Math.max(1, ...sections.map((section) => section.page_end)) : 0;
  const pages = max ? Array.from({ length: max }, (_, index) => index + 1) : [];

  useEffect(() => {
    const node = ref.current?.querySelector(`[data-page="${page}"]`);
    node?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [page]);

  if (pages.length === 0) {
    return (
      <div className="paper-scroll">
        <div className="empty" style={{ textAlign: "center", paddingTop: 48 }}>
          <h3>PDF 预览</h3>
          <p>上传 PDF 后将在此打开。界面预览会用排版稿代替真实文件。</p>
        </div>
      </div>
    );
  }

  return (
    <div className="paper-scroll" ref={ref}>
      {pages.map((n) => {
        const onPageSections = sections.filter(
          (section) => n >= section.page_start && n <= section.page_end,
        );
        const first = onPageSections[0];
        return (
          <article key={n} className={`sheet${n === page ? " is-flash" : ""}`} data-page={n}>
            <div className="sheet-num">{n}</div>
            {n === 1 ? (
              <>
                <h1 className="sheet-title">
                  HieraAlign: Hierarchical Representation Alignment for Vision-Language Models
                </h1>
                <div className="sheet-authors">A. Chen · B. Liu · C. Zhang</div>
              </>
            ) : (
              <p className="sheet-kicker">{first?.title ?? `Page ${n}`}</p>
            )}
            <div className="sheet-body">
              {onPageSections.map((section) => (
                <div key={section.section_id}>
                  {n !== 1 ? <strong>{section.title}. </strong> : null}
                  {section.text}
                  {section.figure_ids.map((id) => (
                    <figure className="sheet-fig" key={id}>
                      <DemoDiagram figureId={id} />
                    </figure>
                  ))}
                </div>
              ))}
            </div>
          </article>
        );
      })}
    </div>
  );
}
