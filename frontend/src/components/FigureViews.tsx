import { useEffect } from "react";

import { MathText } from "./MathText";
import type { Figure } from "../api/types";
import { DemoDiagram } from "./DemoDiagram";

function usableZh(en: string | undefined, zh: string | undefined): string | undefined {
  if (!zh || !zh.trim()) return undefined;
  if (en && zh.trim() === en.trim()) return undefined;
  if (!/[\u4e00-\u9fff]/.test(zh)) return undefined;
  return zh;
}

export function FigureCard({
  figure,
  paperId,
  preview,
  onOpen,
  lang,
}: {
  figure: Figure;
  paperId: string | null;
  preview: boolean;
  onOpen: (figure: Figure) => void;
  lang?: { showEn: boolean; showZh: boolean };
}) {
  const cardClass =
    figure.kind === "formula"
      ? "formula-card"
      : figure.kind === "algorithm"
        ? "algorithm-card"
        : "figure-card";
  return (
    <figure className={cardClass} onClick={() => onOpen(figure)}>
      <FigureVisual figure={figure} paperId={paperId} preview={preview} />
      {figure.kind === "formula" ? null : (
        <FigureCaption
          en={figure.caption || figure.label}
          zh={figure.caption_zh}
          lang={lang}
        />
      )}
    </figure>
  );
}

export function FigureVisual({
  figure,
  paperId,
  preview,
}: {
  figure: Figure;
  paperId: string | null;
  preview: boolean;
}) {
  if (preview || !paperId || !figure.storage_key || figure.storage_key === "demo") {
    return <DemoDiagram figureId={figure.figure_id} />;
  }
  return (
    <img
      alt={figure.caption ?? figure.label ?? (figure.kind === "formula" ? "formula" : "figure")}
      src={`/api/papers/${paperId}/figures/${figure.figure_id}`}
      style={
        figure.kind === "formula"
          ? {
              width: Math.max(48, Math.round(figure.width_px / 2)),
              maxWidth: "100%",
              height: "auto",
            }
          : undefined
      }
    />
  );
}

export function Lightbox({
  figure,
  paperId,
  preview,
  onClose,
}: {
  figure: Figure | undefined;
  paperId: string | null;
  preview: boolean;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!figure) return null;
  const en = [figure.label, figure.caption].filter(Boolean).join(" · ");
  const zh = usableZh(figure.caption || figure.label || "", figure.caption_zh);

  return (
    <button className="lightbox" type="button" onClick={onClose} aria-label="关闭大图">
      <FigureVisual figure={figure} paperId={paperId} preview={preview} />
      <p>
        {en}
        {zh ? (
          <>
            <br />
            <span className="text-zh">{zh}</span>
          </>
        ) : null}{" "}
        · p.{figure.page}
      </p>
    </button>
  );
}

function FigureCaption({
  en,
  zh,
  lang,
}: {
  en: string | null | undefined;
  zh?: string | null;
  lang?: { showEn: boolean; showZh: boolean };
}) {
  const showEn = lang?.showEn ?? true;
  const showZh = lang?.showZh ?? false;
  const zhText = usableZh(en ?? "", zh ?? undefined);
  if (!en && !zhText) return null;
  if (showEn && showZh && zhText) {
    return (
      <figcaption>
        {en ? <MathText text={en} inline /> : null}
        <span className="text-zh">
          <MathText text={zhText} inline />
        </span>
      </figcaption>
    );
  }
  const text = showZh && zhText ? zhText : en;
  if (!text) return null;
  return (
    <figcaption>
      <MathText text={text} inline />
    </figcaption>
  );
}
