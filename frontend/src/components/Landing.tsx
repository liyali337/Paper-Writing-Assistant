import { useEffect, useState } from "react";

import { listPapers } from "../api/client";
import type { Paper, PaperStatus } from "../api/types";
import { AppHeader } from "./AppHeader";
import { ChatIcon, PdfIcon, SplitIcon } from "./icons";

type Props = {
  onBrand: () => void;
  onPicked: (file: File) => void;
  onOpen: (paperId: string) => void;
  onPreview: () => void;
};

type ChipTone = "ok" | "run" | "warn" | "bad" | "mute";

function statusChip(status: PaperStatus): { label: string; tone: ChipTone } {
  if (status === "ready") return { label: "可阅读", tone: "ok" };
  if (status === "needs_ocr") return { label: "需可复制文本", tone: "warn" };
  if (status === "ingest_failed") return { label: "解析失败", tone: "bad" };
  return { label: "解析中", tone: "run" };
}

function paperHeading(paper: Paper) {
  return paper.title?.trim() || paper.filename || "未命名论文";
}

function paperMeta(paper: Paper) {
  const bits: string[] = [];
  if (paper.authors.length) bits.push(paper.authors.slice(0, 3).join("、"));
  else if (paper.title?.trim() && paper.filename) bits.push(paper.filename);
  if (paper.page_count) bits.push(`${paper.page_count} 页`);
  if (paper.figure_count) bits.push(`${paper.figure_count} 图`);
  return bits.join(" · ");
}

export function Landing({ onBrand, onPicked, onOpen, onPreview }: Props) {
  const [over, setOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loadingList, setLoadingList] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listPapers()
      .then((items) => {
        if (!cancelled) setPapers(Array.isArray(items) ? items : []);
      })
      .catch(() => {
        if (!cancelled) setListError("无法加载已上传的论文");
      })
      .finally(() => {
        if (!cancelled) setLoadingList(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function handleFile(file: File) {
    setError(null);
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("请上传 PDF 文件");
      return;
    }
    onPicked(file);
  }

  const showLibrary = loadingList || listError || papers.length > 0;

  return (
    <div className="app-shell">
      <AppHeader onBrand={onBrand} />
      <div className="landing">
        <div className="landing-body">
          <p className="landing-kicker">论文精读</p>
          <h1>上传论文，对照原文提问</h1>
          <p className="lede">
            左侧读章节与 PDF，右侧针对这篇论文追问。引用带页码，一点即可跳回原文。
          </p>
          <label
            className={`dropzone${over ? " is-over" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={(event) => {
              event.preventDefault();
              setOver(false);
              const file = event.dataTransfer.files[0];
              if (file) handleFile(file);
            }}
          >
            <input
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) handleFile(file);
              }}
            />
            <span className="drop-icon">
              <PdfIcon />
            </span>
            <strong>将 PDF 拖到此处，或点击选择</strong>
            <span>可复制文本的论文 · 建议 50MB 以内</span>
          </label>
          {error ? <p className="error-line">{error}</p> : null}
          <div className="landing-links">
            <span>想先看成品长什么样</span>
            <button type="button" onClick={onPreview}>
              查看完整界面预览
            </button>
          </div>
          {showLibrary ? (
            <section className="paper-library" aria-label="已上传的论文">
              <h2>已上传的论文</h2>
              {loadingList ? <p className="paper-library-hint">正在加载列表…</p> : null}
              {listError ? <p className="paper-library-hint is-bad">{listError}</p> : null}
              {papers.length ? (
                <div className="paper-cards">
                  {papers.map((paper) => {
                    const chip = statusChip(paper.status);
                    const meta = paperMeta(paper);
                    return (
                      <button
                        key={paper.paper_id}
                        className="paper-card"
                        type="button"
                        onClick={() => onOpen(paper.paper_id)}
                      >
                        <span className="paper-card-icon">
                          <PdfIcon size={18} />
                        </span>
                        <span className="paper-card-body">
                          <span className="paper-card-row">
                            <span className="paper-card-title">{paperHeading(paper)}</span>
                            <span className={`chip is-${chip.tone}`}>
                              <i />
                              {chip.label}
                            </span>
                          </span>
                          {meta ? <span className="paper-card-meta">{meta}</span> : null}
                        </span>
                      </button>
                    );
                  })}
                </div>
              ) : null}
            </section>
          ) : null}
          <div className="landing-feats">
            <div className="feat">
              <div className="feat-ico">
                <PdfIcon size={16} />
              </div>
              <h3>对照原文</h3>
              <p>上传后立刻打开 PDF，讲解与引用里的页码可以跳页。</p>
            </div>
            <div className="feat">
              <div className="feat-ico">
                <SplitIcon />
              </div>
              <h3>章节精读</h3>
              <p>按标题拆开章节，中英对照阅读，插图可放大。</p>
            </div>
            <div className="feat">
              <div className="feat-ico">
                <ChatIcon size={16} />
              </div>
              <h3>论文问答</h3>
              <p>针对当前这篇追问细节，回答附原文引用，可跳回对应页。</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
