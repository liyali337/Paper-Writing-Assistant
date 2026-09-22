import type { Figure, Section, SectionTranslation } from "../api/types";
import {
  collectLabeledTables,
  layoutPaperMedia,
  normalizeTableLabel,
  splitProseAndTables,
  splitReferenceEntries,
  tablesFromText,
  type ProseBlock,
} from "../lib/mdTable";
import { FigureCard } from "./FigureViews";
import { dropLeadingZhDebris, dropDuplicateLatexProse, isMathOnlyText, pairBilingualProse, peelHanFromLatex, prepareZhMath, realignWhereSegments, stripDisplayMath, stripZhMathDebris } from "../lib/bilingual";
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
  onStopTranslate?: () => void;
  onJump: (page: number, sectionId?: string) => void;
  onOpenFigure: (figure: Figure) => void;
};

const AUTHOR_HINT =
  /@|university|universit[aä]t|institute|laboratory|\blab\b|corresponding author|department of|学院|大学|e-?mail\s*:|\bare with\b|\bis with\b|ieee/i;
const WATERMARK_HINT =
  /watermark|ieee xplore|identical to the accepted version|preprint|arxiv\.org/i;
const BODY_KINDS = new Set(["abstract", "intro", "related", "conclusion", "references"]);

export function isFrontMatterSection(section: Section): boolean {
  return section.title.trim().toLowerCase() === "front matter";
}

export function isPaperTitleSection(section: Section, paperTitle?: string | null): boolean {
  if (isFrontMatterSection(section)) return false;
  if (BODY_KINDS.has(section.kind) || section.level !== 1 || section.page_start > 1) return false;
  const title = section.title.trim();
  if (!title || /^(\d+(?:\.\d+)*|[IVXLC]{1,6})[.\s:-]/i.test(title)) return false;
  if ((section.kind === "method" || section.kind === "experiment") && title.length < 60) return false;
  if (isUsablePaperTitle(paperTitle) && title === paperTitle?.trim()) return true;
  return AUTHOR_HINT.test(section.text);
}

export function isMetaSection(section: Section, paperTitle?: string | null): boolean {
  return isFrontMatterSection(section) || isPaperTitleSection(section, paperTitle);
}

function looksLikePdfFilename(name: string): boolean {
  return /\.pdf$/i.test(name.trim());
}

function isUsablePaperTitle(title?: string | null): boolean {
  if (!title?.trim()) return false;
  return !looksLikePdfFilename(title);
}

function resolvePaperTitle(paperTitle?: string | null, titleSection?: Section): string | undefined {
  if (isUsablePaperTitle(paperTitle)) return paperTitle!.trim();
  if (titleSection?.title.trim()) return titleSection.title.trim();
  if (paperTitle?.trim() && !looksLikePdfFilename(paperTitle)) return paperTitle.trim();
  return undefined;
}

function resolveAuthorIntro(
  frontMatter?: Section,
  titleSection?: Section,
  paperAuthors?: string[],
): string | undefined {
  if (titleSection?.text.trim()) return titleSection.text.trim();
  if (frontMatter?.text.trim()) {
    const kept = splitParagraphs(frontMatter.text).filter((part) => !WATERMARK_HINT.test(part));
    if (kept.length) return kept.join("\n\n");
  }
  if (paperAuthors?.length) return paperAuthors.join(" · ");
  return undefined;
}

function splitRunInAbstract(text: string): { authors: string; abstract: string } | null {
  const match = text.match(/(?:^|\n\n)\s*(?:Abstract|摘要)\s*[-—–:.\u2013\u2014]*\s*/i);
  if (!match || match.index == null) return null;
  const authors = text.slice(0, match.index).trim();
  const abstract = text.slice(match.index + match[0].length).trim();
  if (!abstract) return null;
  return { authors, abstract };
}

function peelAffiliationParagraphs(text: string): { body: string; affiliations: string[] } {
  const parts = splitParagraphs(text);
  const affiliations: string[] = [];
  const kept: string[] = [];
  for (const part of parts) {
    if (part.length <= 700 && /(?:\b(?:are|is|was)\s+with\b|e-?mail\s*:)/i.test(part)) {
      affiliations.push(part);
    } else {
      kept.push(part);
    }
  }
  return { body: kept.join("\n\n"), affiliations };
}

function fixDropCap(text: string): string {
  return text.replace(/^([A-Z])\s+([A-Z][a-z]{2,})\b/, (_all, letter: string, rest: string) => {
    if (/^(the|this|that|these|novel|recent|traditional|however|existing)$/i.test(rest)) {
      return `${letter} ${rest}`;
    }
    return `${letter}${rest[0].toLowerCase()}${rest.slice(1)}`;
  });
}

function prepareReadSections(
  sections: Section[],
  paperTitle?: string | null,
  paperAuthors?: string[],
): { authorText?: string; bodySections: Section[] } {
  const frontMatter = sections.find(isFrontMatterSection);
  const titleSection = sections.find((section) => isPaperTitleSection(section, paperTitle));
  let authorText = resolveAuthorIntro(frontMatter, titleSection, paperAuthors);
  const body: Section[] = [];
  const hasAbstract = sections.some((section) => section.kind === "abstract" && !isMetaSection(section, paperTitle));

  if (titleSection && !hasAbstract) {
    const split = splitRunInAbstract(titleSection.text);
    if (split) {
      authorText = split.authors || authorText;
      body.push({
        ...titleSection,
        section_id: `${titleSection.section_id}-abstract`,
        title: "Abstract",
        kind: "abstract",
        text: split.abstract,
        figure_ids: [],
      });
    }
  }

  const affiliations: string[] = [];
  for (const section of sections) {
    if (isMetaSection(section, paperTitle)) continue;
    if (section.kind === "intro") {
      const peeled = peelAffiliationParagraphs(section.text);
      affiliations.push(...peeled.affiliations);
      body.push({ ...section, text: fixDropCap(peeled.body) });
      continue;
    }
    body.push(section);
  }

  if (affiliations.length) {
    const extra = affiliations.join("\n\n");
    if (!authorText) authorText = extra;
    else if (!authorText.includes(extra.slice(0, 32))) authorText = `${authorText}\n\n${extra}`;
  }

  return { authorText, bodySections: body };
}

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
  onStopTranslate,
  onJump,
  onOpenFigure,
}: Props) {
  const mode = lang ?? { showEn: true, showZh: false };
  const frontMatter = sections.find(isFrontMatterSection);
  const titleSection = sections.find((section) => isPaperTitleSection(section, paperTitle));
  const headerTitle = resolvePaperTitle(paperTitle, titleSection) || heading;
  const headerTitleZh =
    paperTitleZh ||
    (titleSection ? translations?.get(titleSection.section_id)?.title_zh : undefined);
  const prepared = prepareReadSections(sections, paperTitle, paperAuthors);
  const authorText = prepared.authorText;
  const bodySections = prepared.bodySections;
  const headerPage = titleSection?.page_start ?? frontMatter?.page_start;
  const blocksBySection = layoutPaperMedia(
    bodySections.filter((section) => section.kind !== "references"),
    figures,
  );
  const zhTablesByLabel = collectLabeledTables(
    mode.showZh
      ? bodySections.map((section) => translations?.get(section.section_id)?.text_zh)
      : [],
  );

  if (sections.length === 0) {
    return (
      <div className="empty">
        {headerTitle ? (
          <PaperHeader
            title={headerTitle}
            titleZh={headerTitleZh}
            authorText={authorText}
            lang={mode}
          />
        ) : null}
        {note ? <p>{note}</p> : null}
      </div>
    );
  }

  return (
    <article className="prose paper-read">
      {mode.showZh && translating ? (
        <p className="translate-hint">
          <span>
            正在按章节翻译
            {translateProgress ? `（${translateProgress.done}/${translateProgress.total}）` : ""}
            ，长章节会逐段更新译文，英文原文仍可阅读…
            {translateHint ? ` ${translateHint}` : ""}
          </span>
          {onStopTranslate ? (
            <button className="ghost-btn is-stop" type="button" onClick={onStopTranslate}>
              停止翻译
            </button>
          ) : null}
        </p>
      ) : null}
      {mode.showZh && !translating && translateHint ? (
        <p className="translate-hint">{translateHint}</p>
      ) : null}
      {mode.showZh && translateError ? (
        <p className="translate-hint is-bad">{translateError}</p>
      ) : null}
      {headerTitle || authorText ? (
        <PaperHeader
          title={headerTitle || ""}
          titleZh={headerTitleZh}
          authorText={authorText}
          lang={mode}
          page={headerPage}
          onJump={headerPage ? () => onJump(headerPage, titleSection?.section_id) : undefined}
        />
      ) : null}
      {note ? <p className="lead">{note}</p> : null}
      {bodySections.map((section) => {
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
                titleZh={zh?.title_zh}
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
                  blocks={blocksBySection.get(section.section_id) ?? []}
                  textZh={zh?.text_zh}
                  zhTablesByLabel={zhTablesByLabel}
                  lang={mode}
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
  authorText,
  lang,
  page,
  onJump,
}: {
  title: string;
  titleZh?: string | null;
  authorText?: string;
  lang: LangMode;
  page?: number;
  onJump?: () => void;
}) {
  const zh = usableZh(title, titleZh ?? undefined);
  const showEn = Boolean(title) && (lang.showEn || !zh);
  const showZh = Boolean(lang.showZh && zh);
  const authorParas = authorText ? splitParagraphs(authorText) : [];
  if (!showEn && !showZh && authorParas.length === 0) return null;
  return (
    <header className="paper-read-head">
      {showEn || showZh ? (
        <div className="paper-read-title-row">
          <div className="paper-read-titles">
            {showEn ? <h2>{title}</h2> : null}
            {showZh ? (
              showEn ? (
                <p className="paper-read-title-zh">{zh}</p>
              ) : (
                <h2>{zh}</h2>
              )
            ) : null}
          </div>
          {page && onJump ? (
            <button className="cite" type="button" onClick={onJump}>
              p.{page}
            </button>
          ) : null}
        </div>
      ) : null}
      {authorParas.length ? (
        <div className="paper-read-authors">
          {authorParas.map((paragraph, index) => (
            <p key={`${index}-${paragraph.slice(0, 24)}`}>
              <MathText text={paragraph} />
            </p>
          ))}
        </div>
      ) : null}
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
  blocks,
  textZh,
  zhTablesByLabel,
  lang,
  figures,
  paperId,
  preview,
  onOpenFigure,
}: {
  blocks: ProseBlock[];
  textZh?: string;
  zhTablesByLabel: Map<string, Extract<ProseBlock, { type: "table" }>>;
  lang: LangMode;
  figures: Figure[];
  paperId: string | null;
  preview: boolean;
  onOpenFigure: (figure: Figure) => void;
}) {
  const byId = new Map(figures.map((figure) => [figure.figure_id, figure]));
  const zhBlocks = textZh ? splitProseAndTables(textZh) : [];
  const zhTextPool = zhBlocks
    .filter((block): block is Extract<ProseBlock, { type: "text" }> => block.type === "text")
    .filter((block) => !isMathOnlyText(block.text) && /[\u4e00-\u9fff]/.test(block.text))
    .map((block) => block.text);
  const enTextPool = blocks
    .filter((block): block is Extract<ProseBlock, { type: "text" }> => block.type === "text")
    .filter((block) => Boolean(block.text.trim()) && !isMathOnlyText(block.text))
    .map((block) => block.text);
  const pairedZh = lang.showZh ? pairBilingualProse(enTextPool, zhTextPool) : [];
  const zhTablesUnlabeled: Extract<ProseBlock, { type: "table" }>[] = [];
  if (lang.showZh) {
    for (const table of tablesFromText(textZh ?? "")) {
      if (!normalizeTableLabel(table.caption)) zhTablesUnlabeled.push(table);
    }
  }
  const zhLists = zhBlocks.filter((block): block is Extract<ProseBlock, { type: "list" }> => block.type === "list");
  let textPairIndex = 0;
  let unlabeledTableIndex = 0;
  let listIndex = 0;

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
              lang={lang}
            />
          );
        }
        if (block.type === "table") {
          const label = normalizeTableLabel(block.caption);
          const zhBlock = lang.showZh
            ? (label ? zhTablesByLabel.get(label) : null) ??
              zhTablesUnlabeled[unlabeledTableIndex++] ??
              null
            : null;
          return (
            <BilingualTable
              key={`t-${index}`}
              block={block}
              zhBlock={zhBlock}
              lang={lang}
            />
          );
        }
        if (block.type === "list") {
          const zhBlock = lang.showZh ? zhLists[listIndex++] ?? null : null;
          return (
            <BilingualList
              key={`l-${index}`}
              block={block}
              zhBlock={zhBlock}
              lang={lang}
            />
          );
        }
        if (!block.text.trim()) return null;
        const mathOnly = isMathOnlyText(block.text);
        const zhText = lang.showZh && !mathOnly ? pairedZh[textPairIndex++] : undefined;
        return (
          <div key={`p-${index}`} className="bilingual-paragraphs">
            <BilingualText en={block.text} zh={zhText} lang={lang} />
          </div>
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
  const zhText = usableZh(en, zh ? prepareZhMath(stripZhMathDebris(zh)) : zh);
  if (lang.showEn && lang.showZh && zhText) {
    const enSegs = segmentByDisplayMath(en);
    const zhSegs = realignWhereSegments(enSegs, dropLeadingZhDebris(segmentByDisplayMath(zhText)));
    const count = Math.max(enSegs.length, zhSegs.length);
    return (
      <div className="bilingual-segments">
        {Array.from({ length: count }, (_, index) => {
          const enSeg = enSegs[index] ?? { prose: "", display: null };
          const zhSeg = zhSegs[index] ?? { prose: "", display: null };
          const peeledZhDisplay = peelHanFromLatex(zhSeg.display);
          const peeledEnDisplay = peelHanFromLatex(enSeg.display);
          const display = peeledZhDisplay.math ?? peeledEnDisplay.math;
          const zhProse = prepareZhMath(
            dropDuplicateLatexProse(
              stripDisplayMath(
                [zhSeg.prose, peeledZhDisplay.zh, !enSeg.display ? peeledEnDisplay.zh : null]
                  .filter(Boolean)
                  .join("\n\n"),
              ),
            ),
          );
          const segZh = usableZh(enSeg.prose, zhProse);
          if (!segZh && !enSeg.prose.trim() && !display) return null;
          return (
            <div key={index} className="bilingual-segment">
              {enSeg.prose.trim() ? <MathText text={enSeg.prose} /> : null}
              {display ? <MathDisplay value={display} /> : null}
              {segZh ? <MathText text={segZh} className="text-zh" /> : null}
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
  const useZhGrid = Boolean(lang.showZh && zhBlock && !lang.showEn);
  const headers = useZhGrid && zhBlock ? zhBlock.headers : block.headers;
  const rows = useZhGrid && zhBlock ? zhBlock.rows : block.rows;
  const zhHeaders = lang.showZh && zhBlock && lang.showEn ? zhBlock.headers : undefined;
  const zhRows = lang.showZh && zhBlock && lang.showEn ? zhBlock.rows : undefined;

  return (
    <figure className="md-table-wrap">
      <BilingualCaption
        en={block.caption}
        zh={zhBlock?.caption ?? null}
        lang={lang}
        className="md-table-caption"
      />
      <div className="md-table-scroll">
        <table className="md-table">
          <thead>
            <tr>
              {headers.map((cell, cellIndex) => (
                <th key={cellIndex}>
                  <BilingualCell en={cell} zh={zhHeaders?.[cellIndex]} lang={lang} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex}>
                    <BilingualCell
                      en={cell}
                      zh={zhRows?.[rowIndex]?.[cellIndex]}
                      lang={lang}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <BilingualCaption
        en={block.note ?? null}
        zh={zhBlock?.note ?? null}
        lang={lang}
        className="md-table-note"
      />
    </figure>
  );
}

function BilingualCell({
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
    return (
      <>
        {en ? <MathText text={en} inline /> : "\u00a0"}
        <span className="text-zh">
          <MathText text={zhText} inline />
        </span>
      </>
    );
  }
  const text = lang.showZh && zhText ? zhText : en;
  return text ? <MathText text={text} inline /> : "\u00a0";
}

function BilingualCaption({
  en,
  zh,
  lang,
  className,
}: {
  en: string | null;
  zh: string | null;
  lang: LangMode;
  className?: string;
}) {
  const zhText = usableZh(en ?? "", zh ?? undefined);
  if (lang.showEn && lang.showZh && zhText) {
    return (
      <figcaption className={className}>
        {en ? <MathText text={en} inline /> : null}
        <span className="text-zh">
          <MathText text={zhText} inline />
        </span>
      </figcaption>
    );
  }
  const text = lang.showZh && zhText ? zhText : en;
  if (!text) return null;
  return (
    <figcaption className={className}>
      <MathText text={text} inline />
    </figcaption>
  );
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
