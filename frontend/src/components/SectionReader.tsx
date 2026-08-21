import type { Figure, Section } from "../api/types";
import { splitProseAndTables } from "../lib/mdTable";
import { FigureCard } from "./FigureViews";
import { MathText } from "./MathText";

type Props = {
  sections: Section[];
  figures: Figure[];
  paperId: string | null;
  preview: boolean;
  heading: string;
  note: string;
  focusId?: string | null;
  onJump: (page: number) => void;
  onOpenFigure: (figure: Figure) => void;
};

export function SectionReader({
  sections,
  figures,
  paperId,
  preview,
  heading,
  note,
  focusId,
  onJump,
  onOpenFigure,
}: Props) {
  if (sections.length === 0) {
    return (
      <div className="empty">
        <h3>{heading}</h3>
        <p>{note}</p>
      </div>
    );
  }

  return (
    <article className="prose">
      <h2>{heading}</h2>
      <p className="lead">{note}</p>
      {sections.map((section) => {
        const thumbs = figures.filter(
          (figure) =>
            section.figure_ids.includes(figure.figure_id) &&
            figure.kind !== "table_snapshot" &&
            figure.kind !== "formula",
        );
        return (
          <section
            key={section.section_id}
            id={`sec-${section.section_id}`}
            className={`section-read${focusId === section.section_id ? " is-focus" : ""}`}
            style={{ paddingLeft: section.level > 1 ? 12 : 0 }}
          >
            <div className="section-read-head">
              <h3>{section.title}</h3>
              <button className="cite" type="button" onClick={() => onJump(section.page_start)}>
                p.{section.page_start}
                {section.page_end !== section.page_start ? `–${section.page_end}` : ""}
              </button>
            </div>
            {section.text ? (
              <SectionBody
                text={section.text}
                figures={figures}
                paperId={paperId}
                preview={preview}
                onOpenFigure={onOpenFigure}
              />
            ) : null}
            {thumbs.map((figure) => (
              <FigureCard
                key={figure.figure_id}
                figure={figure}
                paperId={paperId}
                preview={preview}
                onOpen={onOpenFigure}
              />
            ))}
          </section>
        );
      })}
    </article>
  );
}

function SectionBody({
  text,
  figures,
  paperId,
  preview,
  onOpenFigure,
}: {
  text: string;
  figures: Figure[];
  paperId: string | null;
  preview: boolean;
  onOpenFigure: (figure: Figure) => void;
}) {
  const byId = new Map(figures.map((figure) => [figure.figure_id, figure]));
  const blocks = splitProseAndTables(text);
  return (
    <div className="section-read-body">
      {blocks.map((block, index) => {
        if (block.type === "figure") {
          const figure = byId.get(block.figureId);
          if (!figure) return null;
          return (
            <FigureCard
              key={`f-${block.figureId}-${index}`}
              figure={figure}
              paperId={paperId}
              preview={preview}
              onOpen={onOpenFigure}
            />
          );
        }
        if (block.type === "table") {
          return (
            <figure className="md-table-wrap" key={`t-${index}`}>
              <div className="md-table-scroll">
                <table className="md-table">
                  <thead>
                    <tr>
                      {block.headers.map((cell, cellIndex) => (
                        <th key={cellIndex}>{cell || "\u00a0"}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {row.map((cell, cellIndex) => (
                          <td key={cellIndex}>{cell || "\u00a0"}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {block.caption ? <figcaption>{block.caption}</figcaption> : null}
            </figure>
          );
        }
        return <p key={`p-${index}`}><MathText text={block.text} /></p>;
      })}
    </div>
  );
}

export function methodLikeSections(sections: Section[]): Section[] {
  const roots = new Set(
    sections
      .filter((section) => section.kind === "method" || section.kind === "experiment")
      .map((section) => section.section_id),
  );
  return sections.filter(
    (section) =>
      section.kind === "method" ||
      section.kind === "experiment" ||
      (section.parent_id != null && roots.has(section.parent_id)),
  );
}
