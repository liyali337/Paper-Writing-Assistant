import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PDFWorker, getDocument, type PDFDocumentProxy, type RenderTask } from "pdfjs-dist";
import PdfJsWorker from "pdfjs-dist/build/pdf.worker.min.mjs?worker";

import { formatApiError, cancelTranslations, getTranslations, startTranslations } from "../api/client";
import type { Figure, PaperTranslation, Section, SectionTranslation } from "../api/types";
import { HttpError } from "../api/types";
import { demoFigureTranslations, demoTitleZh, demoTranslations } from "../data/demo";
import { DemoDiagram } from "./DemoDiagram";
import { Chevron } from "./icons";
import { isMetaSection, SectionReader, type LangMode } from "./SectionReader";

export type SourceView = "pdf" | "sections";

function usableTranslation(zh: SectionTranslation): boolean {
  if (!zh.text_zh.trim()) return false;
  return /[\u4e00-\u9fff]/.test(zh.text_zh);
}

type Props = {
  view: SourceView;
  onView: (view: SourceView) => void;
  pdfUrl: string | null;
  sections: Section[];
  figures: Figure[];
  paperId: string | null;
  paperTitle: string | null;
  paperAuthors?: string[];
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
  paperAuthors = [],
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
  const [figureCaptions, setFigureCaptions] = useState<Map<string, string>>(new Map());
  const [titleZh, setTitleZh] = useState<string | null>(null);
  const [translating, setTranslating] = useState(false);
  const [translateHint, setTranslateHint] = useState<string | null>(null);
  const [translateError, setTranslateError] = useState<string | null>(null);
  const [translateStopped, setTranslateStopped] = useState(false);
  const pollGeneration = useRef(0);
  const showZhRef = useRef(showZh);
  const figuresRef = useRef(figures);
  const sectionsRef = useRef(sections);
  showZhRef.current = showZh;
  figuresRef.current = figures;
  sectionsRef.current = sections;

  const lang = useMemo<LangMode>(() => ({ showEn, showZh }), [showEn, showZh]);
  const figuresWithCaptions = useMemo(
    () =>
      figures.map((figure) => ({
        ...figure,
        caption_zh: figureCaptions.get(figure.figure_id) ?? figure.caption_zh ?? null,
      })),
    [figures, figureCaptions],
  );

  const translateProgress = useMemo(() => {
    if (preview || sections.length === 0) return null;
    const translatable = sections.filter(
      (section) =>
        section.text.trim() &&
        section.kind !== "references" &&
        !isMetaSection(section, paperTitle) &&
        !/^(references|bibliography|参考文献)$/i.test(section.title.trim()),
    );
    const total = translatable.length;
    const done = translatable.filter((section) => {
      const zh = translations.get(section.section_id);
      return zh && !zh.partial && usableTranslation(zh);
    }).length;
    return { done, total };
  }, [preview, paperTitle, sections, translations]);

  const applyPayload = useCallback((payload: PaperTranslation) => {
    setTranslations(new Map(payload.sections.map((item) => [item.section_id, item])));
    setFigureCaptions(
      new Map((payload.figures ?? []).map((item) => [item.figure_id, item.caption_zh])),
    );
    setTitleZh(payload.title_zh?.trim() || null);
  }, []);

  const loadTranslations = useCallback(
    async (mode: "auto" | "resume" | "refresh" = "auto") => {
      if (!paperId || preview || sections.length === 0) return;
      const gen = ++pollGeneration.current;
      const markWorking = (hint: string | null = null) => {
        setTranslating(true);
        setTranslateError(null);
        setTranslateStopped(false);
        setTranslateHint(hint);
      };
      setTranslateError(null);
      // 进入论文时先静默拉取已有译文，确认仍在进行中再显示「停止翻译」
      if (mode === "refresh") {
        markWorking("正在重新翻译，请稍候…");
      } else if (mode === "resume") {
        markWorking("继续翻译未完成的章节…");
      } else {
        setTranslating(false);
      }
      try {
        let payload: PaperTranslation;
        if (mode === "refresh") {
          setTranslations(new Map());
          setFigureCaptions(new Map());
          setTitleZh(null);
          payload = await startTranslations(paperId, true);
        } else if (mode === "resume") {
          payload = await startTranslations(paperId, false);
        } else {
          try {
            payload = await getTranslations(paperId);
          } catch (error) {
            if (error instanceof HttpError && error.status === 404) {
              markWorking(null);
              payload = await startTranslations(paperId, false);
            } else {
              throw error;
            }
          }
          if (payload.status === "cancelled") {
            applyPayload(payload);
            setTranslateStopped(true);
            setTranslateHint("翻译已终止，已完成的章节仍会保留。");
            return;
          }
          // pending 可能是服务重启后的僵尸任务：POST 触发后台续跑
          if (payload.status === "pending") {
            markWorking(null);
            payload = await startTranslations(paperId, false);
          }
        }
        if (gen !== pollGeneration.current) return;
        applyPayload(payload);
        if (payload.status === "failed") {
          setTranslateError(
            payload.error
              ? `翻译失败：${payload.error}`
              : "翻译失败，请检查 .env 中的 API Key / 模型名后重试",
          );
          return;
        }
        if (payload.status === "cancelled") {
          setTranslateStopped(true);
          setTranslateHint("翻译已终止，已完成的章节仍会保留。");
          return;
        }
        const captionedFigures = figuresRef.current.filter(
          (figure) => figure.kind !== "formula" && Boolean(figure.caption?.trim()),
        );
        const translatedCaptionIds = new Set(
          (payload.figures ?? [])
            .filter((item) => /[\u4e00-\u9fff]/.test(item.caption_zh || ""))
            .map((item) => item.figure_id),
        );
        if (
          showZhRef.current &&
          (payload.status === "ready" || payload.status === "partial") &&
          captionedFigures.some((figure) => !translatedCaptionIds.has(figure.figure_id))
        ) {
          payload = await startTranslations(paperId, false);
          if (gen !== pollGeneration.current) return;
          applyPayload(payload);
          if (payload.status === "pending") {
            markWorking("正在翻译图注…");
          }
        }
        let stagnantPolls = 0;
        let lastSectionCount = payload.sections.length;
        while (payload.status === "pending") {
          await new Promise((resolve) => window.setTimeout(resolve, 2500));
          if (gen !== pollGeneration.current) return;
          payload = await getTranslations(paperId);
          if (gen !== pollGeneration.current) return;
          applyPayload(payload);
          if (payload.status === "cancelled") {
            setTranslateStopped(true);
            setTranslateHint("翻译已终止，已完成的章节仍会保留。");
            return;
          }
          if (payload.sections.length > lastSectionCount) {
            lastSectionCount = payload.sections.length;
            stagnantPolls = 0;
            setTranslateHint(null);
          } else {
            stagnantPolls += 1;
            // 分块翻译时整章完成前章节数不变，60s 后仅提示仍进行中
            if (stagnantPolls === 24) {
              setTranslateHint("当前章节较长，仍在翻译中…");
            }
            // 5 分钟无新章节：尝试续跑后台任务（服务重启等场景）
            if (stagnantPolls > 0 && stagnantPolls % 120 === 0) {
              payload = await startTranslations(paperId, false);
              applyPayload(payload);
            }
            // 15 分钟仍 pending 才视为超时
            if (stagnantPolls >= 360) {
              setTranslateError("翻译超时，请点击「重新翻译」继续");
              return;
            }
          }
        }
        if (payload.status === "failed") {
          setTranslateError(
            payload.error
              ? `翻译失败：${payload.error}`
              : "翻译失败，请检查 .env 中的 API Key / 模型名后重试",
          );
          return;
        }
        if (payload.status === "cancelled") {
          setTranslateStopped(true);
          setTranslateHint("翻译已终止，已完成的章节仍会保留。");
          return;
        }
        applyPayload(payload);
        if (payload.status === "ready" || payload.status === "partial") {
          const missing = sectionsRef.current.some((section) => {
            if (!section.text.trim() || section.kind === "references") return false;
            if (isMetaSection(section, paperTitle)) return false;
            if (/^(references|bibliography|参考文献)$/i.test(section.title.trim())) return false;
            const zh = payload.sections.find((item) => item.section_id === section.section_id);
            return !zh || zh.partial || !usableTranslation(zh);
          });
          if (missing && mode !== "refresh") {
            payload = await startTranslations(paperId, false);
            if (gen !== pollGeneration.current) return;
            applyPayload(payload);
            if (payload.status === "pending") {
              markWorking("还有未完成的章节，正在继续翻译…");
            }
            while (payload.status === "pending") {
              await new Promise((resolve) => window.setTimeout(resolve, 2500));
              if (gen !== pollGeneration.current) return;
              payload = await getTranslations(paperId);
              if (gen !== pollGeneration.current) return;
              applyPayload(payload);
              if (payload.status === "cancelled") {
                setTranslateStopped(true);
                setTranslateHint("翻译已终止，已完成的章节仍会保留。");
                return;
              }
            }
          }
          if (payload.status === "ready" || payload.status === "partial") {
            setTranslateHint(null);
          }
        }
      } catch (error) {
        if (gen !== pollGeneration.current) return;
        setTranslateError(formatApiError(error));
      } finally {
        if (gen === pollGeneration.current) setTranslating(false);
      }
    },
    [applyPayload, paperId, paperTitle, preview, sections.length],
  );

  const stopTranslations = useCallback(async () => {
    if (!paperId || preview) return;
    pollGeneration.current += 1;
    setTranslateHint("正在终止翻译…");
    try {
      const payload = await cancelTranslations(paperId);
      applyPayload(payload);
      setTranslateError(null);
      setTranslateStopped(true);
      setTranslateHint("翻译已终止，已完成的章节仍会保留。");
    } catch (error) {
      setTranslateError(formatApiError(error));
    } finally {
      setTranslating(false);
    }
  }, [applyPayload, paperId, preview]);

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
      return;
    }
    if (preview) {
      setTranslations(new Map(demoTranslations.map((item) => [item.section_id, item])));
      setFigureCaptions(new Map(demoFigureTranslations.map((item) => [item.figure_id, item.caption_zh])));
      setTitleZh(demoTitleZh);
      setTranslateError(null);
      setTranslateHint(null);
      setTranslateStopped(false);
      setTranslating(false);
      return;
    }
    if (!paperId || sections.length === 0) return;

    void loadTranslations("auto");

    return () => {
      if (showZhRef.current) {
        pollGeneration.current += 1;
      }
    };
  }, [loadTranslations, paperId, preview, sections.length, showZh]);

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
          </div>
        ) : null}
        {view === "sections" ? (
          <div className="translate-actions">
            {translating ? (
              <button
                className="ghost-btn is-stop"
                type="button"
                disabled={preview || !paperId}
                onClick={() => void stopTranslations()}
              >
                停止翻译
              </button>
            ) : null}
            {!translating &&
            (translateStopped ||
              (translateProgress != null &&
                translateProgress.done < translateProgress.total &&
                translations.size > 0)) ? (
              <button
                className="ghost-btn"
                type="button"
                disabled={preview || !paperId}
                onClick={() => void loadTranslations("resume")}
              >
                继续翻译
              </button>
            ) : null}
            <button
              className="ghost-btn"
              type="button"
              disabled={preview || !paperId || translating}
              onClick={() => void loadTranslations("refresh")}
            >
              重新翻译
            </button>
            {translating && translateProgress ? (
              <span className="lang-status">
                翻译中 {translateProgress.done}/{translateProgress.total}
              </span>
            ) : translating ? (
              <span className="lang-status">翻译中…</span>
            ) : null}
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
            figures={figuresWithCaptions}
            paperId={paperId}
            preview={preview}
            paperTitle={paperTitle}
            paperTitleZh={titleZh}
            paperAuthors={paperAuthors}
            focusId={focusSection}
            lang={lang}
            translations={translations}
            translating={translating}
            translateHint={translateHint}
            translateProgress={translateProgress}
            translateError={translateError}
            onStopTranslate={() => void stopTranslations()}
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
