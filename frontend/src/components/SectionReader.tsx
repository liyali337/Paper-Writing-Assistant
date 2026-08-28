import type { Figure, Section, SectionTranslation } from "../api/types";
import {
  attachFigures,
  splitProseAndTables,
  splitReferenceEntries,
  type ProseBlock,
} from "../lib/mdTable";
import { FigureCard } from "./FigureViews";
import { MathDisplay, MathText, segmentByDisplayMath } from "./MathText";

export type LangMode = {
  showEn: boolean;
  showZh: boolean;
};

type Props = {
  sections: Section[];
  figures: Figure[];
  paperId: string | null;
  preview: boolean;
  paperTitle?: string | null;
  paperTitleZh?: string | null;
  paperAuthors?: string[];
  heading?: string;
  note?: string;
  focusId?: string | null;
  lang?: LangMode;
  translations?: Map<string, SectionTranslation>;
  translating?: boolean;
  translateHint?: string | null;
  translateProgress?: { done: number; total: number } | null;
  translateError?: string | null;
  onJump: (page: number, sectionId?: string) => void;
  onOpenFigure: (figure: Figure) => void;
};

export function SectionReader({
  sections,
  figures,
  paperId,
  preview,
  paperTitle,
  paperTitleZh,
  paperAuthors,
  heading,
  note,
  focusId,
  lang,
  translations,
  translating,
  translateHint,
  translateProgress,
  translateError,
  onJump,
  onOpenFigure,
}: Props) {
  const topTitle = paperTitle || heading;
  const mode = lang ?? { showEn: true, showZh: false };

  if (sections.length === 0) {
    return (
      <div className="empty">
        {topTitle ? (
          <PaperHeader title={topTitle} titleZh={paperTitleZh} authors={paperAuthors} lang={mode} />
        ) : null}
        {note ? <p>{note}</p> : null}
      </div>
    );
  }

  return (
    <article className="prose paper-read">
      {topTitle ? (
        <PaperHeader title={topTitle} titleZh={paperTitleZh} authors={paperAuthors} lang={mode} />
      ) : null}
      {note ? <p className="lead">{note}</p> : null}
      {mode.showZh && translating ? (
        <p className="translate-hint">
          正在按章节翻译
          {translateProgress ? `（${translateProgress.done}/${translateProgress.total}）` : ""}
          ，长章节会逐段更新译文，英文原文仍可阅读…
          {translateHint ? ` ${translateHint}` : ""}
        </p>
      ) : null}
      {mode.showZh && translateError ? (
        <p className="translate-hint is-bad">{translateError}</p>
      ) : null}
      {sections.map((section) => {
        const isFrontMatter = section.title.trim().toLowerCase() === "front matter";
        const zh = translations?.get(section.section_id);
        return (
          <section
            key={section.section_id}
            id={`sec-${section.section_id}`}
            className={[
              "section-read",
              `level-${Math.min(section.level, 3)}`,
              section.kind === "abstract" ? "is-abstract" : "",
              section.kind === "references" ? "is-refs" : "",
              focusId === section.section_id ? "is-focus" : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            <div className="section-read-head">
              <SectionTitle
                title={section.title}
                titleZh={isFrontMatter ? undefined : zh?.title_zh}
                showEn={mode.showEn}
                showZh={mode.showZh}
              />
              <button
                className="cite"
                type="button"
                onClick={() => onJump(section.page_start, section.section_id)}
              >
                p.{section.page_start}
                {section.page_end !== section.page_start ? `–${section.page_end}` : ""}
              </button>
            </div>
            {section.text ? (
              section.kind === "references" ? (
                <div className="section-read-body">
                  <ReferencesBody text={section.text} lang={mode} textZh={zh?.text_zh} />
                </div>
              ) : (
                <SectionBody
                  text={section.text}
                  textZh={isFrontMatter ? undefined : zh?.text_zh}
                  lang={mode}
                  figureIds={section.figure_ids}
                  figures={figures}
                  paperId={paperId}
                  preview={preview}
                  onOpenFigure={onOpenFigure}
                />
              )
            ) : null}
          </section>
        );
      })}
    </article>
  );
}

function PaperHeader({
  title,
  titleZh,
  authors,
  lang,
}: {
  title: string;
  titleZh?: string | null;
  authors?: string[];
  lang: LangMode;
}) {
  const zh = usableZh(title, titleZh ?? undefined);
  return (
    <header className="paper-read-head">
      {lang.showEn && lang.showZh && zh ? (
        <h2>
          {title}
          <span className="text-zh"> {zh}</span>
        </h2>
      ) : lang.showZh && zh ? (
        <h2>{zh}</h2>
      ) : (
        <h2>{title}</h2>
      )}
      {authors?.length ? <p className="paper-read-authors">{authors.join(" · ")}</p> : null}
    </header>
  );
}

function SectionTitle({
  title,
  titleZh,
  showEn,
  showZh,
}: {
  title: string;
  titleZh?: string;
  showEn: boolean;
  showZh: boolean;
}) {
  const zh = usableZh(title, titleZh);
  if (showEn && showZh && zh) {
    return (
      <h3>
        {title}
        <span className="text-zh"> {zh}</span>
      </h3>
    );
  }
  if (showZh && zh) return <h3>{zh}</h3>;
  return <h3>{title}</h3>;
}

function SectionBody({
  text,
  textZh,
  lang,
  figureIds,
  figures,
  paperId,
  preview,
  onOpenFigure,
}: {
  text: string;
  textZh?: string;
  lang: LangMode;
  figureIds: string[];
  figures: Figure[];
  paperId: string | null;
  preview: boolean;
  onOpenFigure: (figure: Figure) => void;
}) {
  const byId = new Map(figures.map((figure) => [figure.figure_id, figure]));
  const { blocks, used } = attachFigures(splitProseAndTables(text), figures);
  const zhBlocks = textZh ? attachFigures(splitProseAndTables(textZh), figures).blocks : [];
  const zhParagraphs = textZh ? splitParagraphs(textZh) : [];
  let zhParaIndex = 0;
  const leftover = figureIds.filter((figureId) => {
    if (used.has(figureId)) return false;
    const figure = byId.get(figureId);
    if (!figure) return false;
    return figure.kind !== "table_snapshot" && figure.kind !== "formula";
  });
  let zhCursor = 0;

  const nextZhParagraph = () => {
    if (zhParaIndex >= zhParagraphs.length) return undefined;
    const paragraph = zhParagraphs[zhParaIndex];
    zhParaIndex += 1;
    return paragraph;
  };

  const nextZh = (type: ProseBlock["type"]) => {
    while (zhCursor < zhBlocks.length) {
      const block = zhBlocks[zhCursor];
      zhCursor += 1;
      if (block.type === "figure") continue;
      if (block.type === type) return block;
    }
    return null;
  };

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
          const zhBlock = lang.showZh ? nextZh("table") : null;
          return (
            <BilingualTable
              key={`t-${index}`}
              block={block}
              zhBlock={zhBlock?.type === "table" ? zhBlock : null}
              lang={lang}
            />
          );
        }
        if (block.type === "list") {
          const zhBlock = lang.showZh ? nextZh("list") : null;
          return (
            <BilingualList
              key={`l-${index}`}
              block={block}
              zhBlock={zhBlock?.type === "list" ? zhBlock : null}
              lang={lang}
            />
          );
        }
        const enParagraphs = splitParagraphs(block.text);
        if (enParagraphs.length === 0) return null;
        return (
          <div key={`p-${index}`} className="bilingual-paragraphs">
            {enParagraphs.map((enPara, paraIndex) => (
              <BilingualText
                key={`p-${index}-${paraIndex}`}
                en={enPara}
                zh={lang.showZh ? nextZhParagraph() : undefined}
                lang={lang}
              />
            ))}
          </div>
        );
      })}
      {leftover.map((figureId) => {
        const figure = byId.get(figureId);
        if (!figure) return null;
        return (
          <FigureCard
            key={`tail-${figureId}`}
            figure={figure}
            paperId={paperId}
            preview={preview}
            onOpen={onOpenFigure}
          />
        );
      })}
    </div>
  );
}

function usableZh(en: string | undefined, zh: string | undefined): string | undefined {
  if (!zh || !zh.trim()) return undefined;
  if (en && zh.trim() === en.trim()) return undefined;
  // 没有汉字就不当作中文译文（失败回退 / 模型抄原文）
  if (!/[\u4e00-\u9fff]/.test(zh)) return undefined;
  return zh;
}

function splitParagraphs(text: string): string[] {
  return text
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);
}

function BilingualText({
  en,
  zh,
  lang,
}: {
  en: string;
  zh?: string;
  lang: LangMode;
}) {
  const zhText = usableZh(en, zh);
  if (lang.showEn && lang.showZh && zhText) {
    const enSegs = segmentByDisplayMath(en);
    const zhSegs = segmentByDisplayMath(zhText);
    const count = Math.max(enSegs.length, zhSegs.length);
    return (
      <div className="bilingual-segments">
        {Array.from({ length: count }, (_, index) => {
          const enSeg = enSegs[index] ?? { prose: "", display: null };
          const zhSeg = zhSegs[index] ?? { prose: "", display: null };
          const segZh = usableZh(enSeg.prose, zhSeg.prose);
          const display = enSeg.display ?? zhSeg.display;
          if (!segZh && !enSeg.prose.trim() && !display) return null;
          return (
            <div key={index} className="bilingual-segment">
              {enSeg.prose.trim() ? <MathText text={enSeg.prose} /> : null}
              {segZh ? <MathText text={zhSeg.prose} className="text-zh" /> : null}
              {display ? <MathDisplay value={display} /> : null}
            </div>
          );
        })}
      </div>
    );
  }
  if (lang.showZh && zhText) return <MathText text={zhText} className="text-zh" />;
  if (lang.showEn || (lang.showZh && !zhText)) return <MathText text={en} />;
  return null;
}

function BilingualList({
  block,
  zhBlock,
  lang,
}: {
  block: Extract<ProseBlock, { type: "list" }>;
  zhBlock: Extract<ProseBlock, { type: "list" }> | null;
  lang: LangMode;
}) {
  const Tag = block.ordered ? "ol" : "ul";
  const zhItems =
    zhBlock?.items.map((item, index) => usableZh(block.items[index], item)).filter(Boolean) ?? [];
  const hasZh = zhItems.length > 0 && zhBlock != null;

  if (lang.showEn && lang.showZh && hasZh && zhBlock) {
    return (
      <div className="bilingual-block">
        <Tag className="section-list">
          {block.items.map((item, itemIndex) => (
            <li key={itemIndex}>
              <MathText text={item} inline />
            </li>
          ))}
        </Tag>
        <Tag className="section-list text-zh">
          {zhBlock.items.map((item, itemIndex) => {
            const zh = usableZh(block.items[itemIndex], item);
            return (
              <li key={itemIndex}>
                <MathText text={zh || item} inline />
              </li>
            );
          })}
        </Tag>
      </div>
    );
  }
  if (lang.showZh && hasZh && zhBlock) {
    const ZhTag = zhBlock.ordered ? "ol" : "ul";
    return (
      <ZhTag className="section-list text-zh">
        {zhBlock.items.map((item, itemIndex) => (
          <li key={itemIndex}>
            <MathText text={usableZh(block.items[itemIndex], item) || item} inline />
          </li>
        ))}
      </ZhTag>
    );
  }
  if (lang.showEn || lang.showZh) {
    return (
      <Tag className="section-list">
        {block.items.map((item, itemIndex) => (
          <li key={itemIndex}>
            <MathText text={item} inline />
          </li>
        ))}
      </Tag>
    );
  }
  return null;
}

function BilingualTable({
  block,
  zhBlock,
  lang,
}: {
  block: Extract<ProseBlock, { type: "table" }>;
  zhBlock: Extract<ProseBlock, { type: "table" }> | null;
  lang: LangMode;
}) {
  const renderTable = (
    headers: string[],
    rows: string[][],
    caption: string | null,
    className?: string,
  ) => (
    <figure className={["md-table-wrap", className].filter(Boolean).join(" ")}>
      <div className="md-table-scroll">
        <table className="md-table">
          <thead>
            <tr>
              {headers.map((cell, cellIndex) => (
                <th key={cellIndex}>{cell ? <MathText text={cell} inline /> : "\u00a0"}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex}>{cell ? <MathText text={cell} inline /> : "\u00a0"}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {caption ? (
        <figcaption>
          <MathText text={caption} inline />
        </figcaption>
      ) : null}
    </figure>
  );

  const zhUsable =
    zhBlock &&
    usableZh(
      [block.caption, ...block.headers, ...block.rows.flat()].filter(Boolean).join("\n"),
      [zhBlock.caption, ...zhBlock.headers, ...zhBlock.rows.flat()].filter(Boolean).join("\n"),
    );

  if (lang.showEn && lang.showZh && zhBlock && zhUsable) {
    return (
      <div className="bilingual-block">
        {renderTable(block.headers, block.rows, block.caption)}
        {renderTable(zhBlock.headers, zhBlock.rows, zhBlock.caption, "text-zh")}
      </div>
    );
  }
  if (lang.showZh && zhBlock && zhUsable) {
    return renderTable(zhBlock.headers, zhBlock.rows, zhBlock.caption, "text-zh");
  }
  if (lang.showEn || lang.showZh) {
    return renderTable(block.headers, block.rows, block.caption);
  }
  return null;
}

function ReferencesBody({
  text,
  lang,
  textZh,
}: {
  text: string;
  lang: LangMode;
  textZh?: string;
}) {
  const entries = splitReferenceEntries(text);
  const zhEntries = textZh ? splitReferenceEntries(textZh) : [];
  if (!entries.length) return null;
  return (
    <ol className="ref-list">
      {entries.map((entry, index) => (
        <li key={`${index}-${entry.slice(0, 24)}`}>
          <span className="ref-n">[{index + 1}]</span>
          <span className="ref-body">
            {lang.showEn ? <MathText text={entry} inline cites={false} /> : null}
            {lang.showZh && usableZh(entry, zhEntries[index]) ? (
              <MathText text={zhEntries[index]} inline cites={false} className="text-zh" />
            ) : lang.showZh && !lang.showEn ? (
              <MathText text={entry} inline cites={false} />
            ) : null}
          </span>
        </li>
      ))}
    </ol>
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
