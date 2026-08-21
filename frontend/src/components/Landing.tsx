import { useState } from "react";

import { LogoMark, PdfIcon, SplitIcon } from "./icons";

type Props = {
  onPicked: (file: File) => void;
  onPreview: () => void;
};

export function Landing({ onPicked, onPreview }: Props) {
  const [over, setOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleFile(file: File) {
    setError(null);
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("请上传 PDF 文件");
      return;
    }
    onPicked(file);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <LogoMark />
          </span>
          <span className="brand-name">
            论文理解 <span className="brand-sub">Paper Lens</span>
          </span>
        </div>
      </header>
      <div className="landing">
        <div className="landing-body">
          <h1>上传论文，五分钟看懂方法</h1>
          <p className="lede">
            左侧对照原文，右侧读中文介绍与带原图的方法步骤。点击页码即可跳回 PDF。
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
          <div className="landing-feats">
            <div className="feat">
              <div className="feat-ico">
                <PdfIcon size={16} />
              </div>
              <h3>对照原文</h3>
              <p>上传后立刻打开 PDF，讲解里的页码可以跳页。</p>
            </div>
            <div className="feat">
              <div className="feat-ico">
                <SplitIcon />
              </div>
              <h3>总体介绍</h3>
              <p>问题、动机、贡献、阅读顺序，可选一张总览图。</p>
            </div>
            <div className="feat">
              <div className="feat-ico">
                <LogoMark size={16} />
              </div>
              <h3>方法详解</h3>
              <p>按步骤讲清流水线，步骤旁插入从 PDF 裁出的原图。</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
