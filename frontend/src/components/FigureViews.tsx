import { useEffect } from "react";

import type { Figure } from "../api/types";
import { DemoDiagram } from "./DemoDiagram";

type Props = {
  figure: Figure | undefined;
  paperId: string | null;
  preview: boolean;
  onClose: () => void;
};

export function FigureCard({
  figure,
  paperId,
  preview,
  onOpen,
}: {
  figure: Figure;
  paperId: string | null;
  preview: boolean;
  onOpen: (figure: Figure) => void;
}) {
  const isFormula = figure.kind === "formula";
  return (
    <figure className={isFormula ? "formula-card" : "figure-card"} onClick={() => onOpen(figure)}>
      <FigureVisual figure={figure} paperId={paperId} preview={preview} />
      {isFormula ? null : <figcaption>{figure.caption || figure.label}</figcaption>}
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

export function Lightbox({ figure, paperId, preview, onClose }: Props) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!figure) return null;

  return (
    <button className="lightbox" type="button" onClick={onClose} aria-label="关闭大图">
      <FigureVisual figure={figure} paperId={paperId} preview={preview} />
      <p>
        {figure.label}
        {figure.caption ? ` · ${figure.caption}` : ""} · p.{figure.page}
      </p>
    </button>
  );
}
