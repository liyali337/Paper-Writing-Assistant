export type ProseBlock =
  | { type: "text"; text: string }
  | { type: "table"; headers: string[]; rows: string[][]; caption: string | null; note?: string | null }
  | { type: "figure"; figureId: string }
  | { type: "list"; ordered: boolean; items: string[] };

export type FigureRef = {
  figure_id: string;
  kind: string;
  label: string | null;
  caption: string | null;
  section_id?: string | null;
};

export type SectionMediaInput = {
  section_id: string;
  text: string;
  figure_ids: string[];
};

const TABLE_NUMBER = "([IVXLCDM]+|[A-Z]?\\d+)";
const TABLE_CAPTION = new RegExp(`^(?:table|tab\\.?|表)\\s*\\.?\\s*${TABLE_NUMBER}\\b`, "i");
const FORMULA_MARK = /<!--fig:(fig-\d+)-->/g;
const FIGURE_MENTION = /\b(fig(?:ure)?|algorithm)\s*\.?\s*([A-Z]?\d+)\b/gi;
const MEDIA_MENTION = new RegExp(
  `\\b(fig(?:ure)?|algorithm|table|tab\\.?)\\s*\\.?\\s*${TABLE_NUMBER}\\b`,
  "gi",
);
const TABLE_MENTION_VERB =
  /^(?:compares?|shows?|presents?|reports?|summarizes?|lists?|illustrates?|demonstrates?|validates?|outlines?)\b/i;
const TABLE_NOTE_MARK = /(?:†|\*|‡|∗|\\ast|\\dagger|denote|indicate|applied|missing the)/i;
const LIST_NUMBERED = /^(?:\(\d+\)|\d+\.(?!\d))\s+(\S[\s\S]*)$/;
const LIST_BULLET = /^[-–•*]\s+(\S[\s\S]*)$/;
const VISUAL_FIGURE = new Set(["figure", "algorithm"]);

export function splitProseAndTables(text: string): ProseBlock[] {
  const blocks: ProseBlock[] = [];
  const source = text.replace(/\r\n/g, "\n");
  FORMULA_MARK.lastIndex = 0;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = FORMULA_MARK.exec(source))) {
    pushChunk(blocks, source.slice(last, match.index));
    blocks.push({ type: "figure", figureId: match[1] });
    last = match.index + match[0].length;
  }
  pushChunk(blocks, source.slice(last));
  return clusterLists(attachTableChrome(blocks));
}

export function attachFigures(
  blocks: ProseBlock[],
  figures: FigureRef[],
): { blocks: ProseBlock[]; used: Set<string> } {
  const byLabel = new Map<string, string>();
  for (const figure of figures) {
    if (figure.kind === "formula" || figure.kind === "table_snapshot") continue;
    const label = normalizeFigureLabel(figure.label) ?? normalizeFigureLabel(figure.caption);
    if (label && !byLabel.has(label)) byLabel.set(label, figure.figure_id);
  }
  const used = new Set<string>();
  for (const block of blocks) {
    if (block.type === "figure") used.add(block.figureId);
  }
  const out: ProseBlock[] = [];
  for (const block of blocks) {
    out.push(block);
    for (const figureId of mentionsIn(block, byLabel)) {
      if (used.has(figureId)) continue;
      used.add(figureId);
      out.push({ type: "figure", figureId });
    }
  }
  return { blocks: out, used };
}

export function normalizeFigureLabel(text: string | null | undefined): string | null {
  if (!text) return null;
  const match = text.match(/^(fig(?:ure)?|algorithm)\s*\.?\s*([A-Z]?\d+)\b/i);
  if (!match) return null;
  const kind = match[1].toLowerCase().startsWith("alg") ? "algorithm" : "figure";
  return `${kind} ${match[2].toLowerCase()}`;
}

export function normalizeTableLabel(text: string | null | undefined): string | null {
  if (!text) return null;
  const match = text.match(new RegExp(`^(?:table|tab\\.?|表)\\s*\\.?\\s*${TABLE_NUMBER}\\b`, "i"));
  if (!match) return null;
  const num = normalizeTableNumber(match[1]);
  return num ? `table ${num}` : null;
}

export function normalizeTableNumber(raw: string | null | undefined): string | null {
  if (!raw) return null;
  const token = raw.trim();
  if (/^[A-Z]?\d+$/i.test(token)) return token.toLowerCase();
  const roman = romanToInt(token);
  return roman ? String(roman) : null;
}

export function tablesFromText(text: string): Extract<ProseBlock, { type: "table" }>[] {
  return splitProseAndTables(text).filter(
    (block): block is Extract<ProseBlock, { type: "table" }> => block.type === "table",
  );
}

export function collectLabeledTables(
  texts: Iterable<string | null | undefined>,
): Map<string, Extract<ProseBlock, { type: "table" }>> {
  const map = new Map<string, Extract<ProseBlock, { type: "table" }>>();
  for (const text of texts) {
    if (!text) continue;
    for (const table of tablesFromText(text)) {
      const label = normalizeTableLabel(table.caption);
      if (label && !map.has(label)) map.set(label, table);
    }
  }
  return map;
}

export function layoutPaperMedia(
  sections: SectionMediaInput[],
  figures: FigureRef[],
): Map<string, ProseBlock[]> {
  const figureByLabel = new Map<string, string>();
  const figureById = new Map(figures.map((figure) => [figure.figure_id, figure]));
  for (const figure of figures) {
    if (!VISUAL_FIGURE.has(figure.kind)) continue;
    const label = normalizeFigureLabel(figure.label) ?? normalizeFigureLabel(figure.caption);
    if (label && !figureByLabel.has(label)) figureByLabel.set(label, figure.figure_id);
  }

  type Parsed = {
    prose: ProseBlock[];
    tables: Extract<ProseBlock, { type: "table" }>[];
  };
  const parsed = new Map<string, Parsed>();
  const tableByLabel = new Map<string, Extract<ProseBlock, { type: "table" }>>();
  const unlabeledBySection = new Map<string, Extract<ProseBlock, { type: "table" }>[]>();
  const tableOrigin = new Map<Extract<ProseBlock, { type: "table" }>, string>();

  for (const section of sections) {
    const blocks = splitProseAndTables(section.text);
    const tables: Extract<ProseBlock, { type: "table" }>[] = [];
    const prose: ProseBlock[] = [];
    for (const block of blocks) {
      if (block.type === "table") {
        tables.push(block);
        tableOrigin.set(block, section.section_id);
        const label = normalizeTableLabel(block.caption);
        if (label) {
          if (!tableByLabel.has(label)) tableByLabel.set(label, block);
        } else {
          const list = unlabeledBySection.get(section.section_id) ?? [];
          list.push(block);
          unlabeledBySection.set(section.section_id, list);
        }
        continue;
      }
      prose.push(block);
    }
    parsed.set(section.section_id, { prose, tables });
  }

  type MdTable = Extract<ProseBlock, { type: "table" }>;
  const usedFigures = new Set<string>();
  const usedTables = new Set<MdTable>();
  const usedTableLabels = new Set<string>();
  const unmatchedTableMentions: { sectionId: string; index: number }[] = [];
  const extras = new Map<string, Map<number, ProseBlock[]>>();

  const queueExtra = (sectionId: string, index: number, block: ProseBlock) => {
    let byIndex = extras.get(sectionId);
    if (!byIndex) {
      byIndex = new Map();
      extras.set(sectionId, byIndex);
    }
    const list = byIndex.get(index) ?? [];
    list.push(block);
    byIndex.set(index, list);
  };

  const markTableUsed = (table: MdTable) => {
    usedTables.add(table);
    const label = normalizeTableLabel(table.caption);
    if (label) usedTableLabels.add(label);
  };

  const tableAlreadyShown = (table: MdTable) => {
    if (usedTables.has(table)) return true;
    const label = normalizeTableLabel(table.caption);
    return Boolean(label && usedTableLabels.has(label));
  };

  for (const section of sections) {
    const prose = parsed.get(section.section_id)?.prose ?? [];
    for (const [index, block] of prose.entries()) {
      if (block.type === "figure") {
        usedFigures.add(block.figureId);
        continue;
      }
      for (const mention of scanMediaMentions(blockText(block))) {
        if (mention.kind === "figure") {
          const figureId = figureByLabel.get(mention.key);
          if (!figureId || usedFigures.has(figureId)) continue;
          usedFigures.add(figureId);
          queueExtra(section.section_id, index, { type: "figure", figureId });
          continue;
        }
        if (usedTableLabels.has(mention.key)) continue;
        const table = tableByLabel.get(mention.key);
        if (!table || usedTables.has(table)) {
          unmatchedTableMentions.push({ sectionId: section.section_id, index });
          continue;
        }
        markTableUsed(table);
        queueExtra(section.section_id, index, table);
      }
    }
  }

  for (const miss of unmatchedTableMentions) {
    const candidate = (parsed.get(miss.sectionId)?.tables ?? []).find(
      (table) => tableOrigin.get(table) === miss.sectionId && !tableAlreadyShown(table),
    );
    if (!candidate) continue;
    markTableUsed(candidate);
    queueExtra(miss.sectionId, miss.index, candidate);
  }

  const layout = new Map<string, ProseBlock[]>();
  for (const section of sections) {
    const prose = parsed.get(section.section_id)?.prose ?? [];
    const byIndex = extras.get(section.section_id);
    const out: ProseBlock[] = [];
    for (const [index, block] of prose.entries()) {
      out.push(block);
      const inserted = byIndex?.get(index);
      if (inserted) out.push(...inserted);
    }
    for (const figureId of section.figure_ids) {
      if (usedFigures.has(figureId)) continue;
      const figure = figureById.get(figureId);
      if (!figure || !VISUAL_FIGURE.has(figure.kind)) continue;
      usedFigures.add(figureId);
      out.push({ type: "figure", figureId });
    }
    for (const table of unlabeledBySection.get(section.section_id) ?? []) {
      if (tableAlreadyShown(table)) continue;
      markTableUsed(table);
      out.push(table);
    }
    for (const table of parsed.get(section.section_id)?.tables ?? []) {
      if (tableAlreadyShown(table)) continue;
      if (tableOrigin.get(table) !== section.section_id) continue;
      markTableUsed(table);
      out.push(table);
    }
    layout.set(section.section_id, out);
  }
  return layout;
}

function blockText(block: ProseBlock): string {
  if (block.type === "text") return block.text;
  if (block.type === "list") return block.items.join("\n");
  if (block.type === "table") {
    return [block.caption, ...block.headers].filter(Boolean).join(" ");
  }
  return "";
}

function mentionsIn(block: ProseBlock, byLabel: Map<string, string>): string[] {
  const text = blockText(block);
  if (!text) return [];
  const ids: string[] = [];
  FIGURE_MENTION.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = FIGURE_MENTION.exec(text))) {
    const kind = match[1].toLowerCase().startsWith("alg") ? "algorithm" : "figure";
    const label = `${kind} ${match[2].toLowerCase()}`;
    const figureId = byLabel.get(label);
    if (figureId && !ids.includes(figureId)) ids.push(figureId);
  }
  return ids;
}

function scanMediaMentions(text: string): { kind: "figure" | "table"; key: string }[] {
  if (!text) return [];
  const out: { kind: "figure" | "table"; key: string }[] = [];
  const seen = new Set<string>();
  MEDIA_MENTION.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = MEDIA_MENTION.exec(text))) {
    const word = match[1].toLowerCase();
    const raw = match[2];
    const kind = word.startsWith("tab") ? "table" : "figure";
    const num =
      kind === "table" ? (normalizeTableNumber(raw) ?? raw.toLowerCase()) : raw.toLowerCase();
    const key = word.startsWith("alg")
      ? `algorithm ${num}`
      : kind === "table"
        ? `table ${num}`
        : `figure ${num}`;
    const token = `${kind}:${key}`;
    if (seen.has(token)) continue;
    seen.add(token);
    out.push({ kind, key });
  }
  return out;
}

function pushChunk(blocks: ProseBlock[], chunk: string) {
  for (const block of splitTablesAndParagraphs(chunk)) {
    blocks.push(block);
  }
}

function splitTablesAndParagraphs(text: string): ProseBlock[] {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  type Chunk = { table: boolean; lines: string[] };
  const raw: Chunk[] = [];
  let inTable = false;

  const pushLine = (line: string, table: boolean) => {
    const last = raw[raw.length - 1];
    if (!last || last.table !== table) {
      raw.push({ table, lines: [line] });
    } else {
      last.lines.push(line);
    }
    inTable = table;
  };

  for (const line of lines) {
    if (isMdTableLine(line)) {
      pushLine(line, true);
    } else if (line.trim() === "" && inTable) {
      inTable = false;
    } else {
      pushLine(line, false);
    }
  }

  const blocks: ProseBlock[] = [];
  for (const chunk of raw) {
    const blob = chunk.lines.join("\n").trim();
    if (!blob) continue;
    if (chunk.table) {
      let added = false;
      for (const piece of splitStackedTableChunks(chunk.lines)) {
        const table = parseMdTable(piece);
        if (table) {
          blocks.push({ type: "table", ...table, caption: null });
          added = true;
        }
      }
      if (added) continue;
    }
    for (const para of toParagraphs(blob)) {
      if (isBareFigureCaption(para)) continue;
      const split = splitCaptionFromBody(para);
      if (split.caption && blocks.at(-1)?.type === "table") {
        const prev = blocks[blocks.length - 1];
        if (prev.type === "table") {
          const nextLabel = normalizeTableLabel(split.caption);
          const currentLabel = normalizeTableLabel(prev.caption);
          if (!prev.caption) {
            prev.caption = split.caption;
            if (split.rest) pushText(blocks, split.rest);
            continue;
          }
          if (!nextLabel || !currentLabel || nextLabel === currentLabel) {
            prev.caption = joinCaption(prev.caption, split.caption);
            if (split.rest) pushText(blocks, split.rest);
            continue;
          }
        }
      }
      pushText(blocks, para);
    }
  }
  return blocks;
}

function toParagraphs(blob: string): string[] {
  return blob
    .split(/\n{2,}/)
    .flatMap((para) => {
      const lines = para
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);
      if (lines.length <= 1) return lines;
      if (isBareFigureCaption(lines[0])) {
        return [lines.slice(1).join("\n")].filter(Boolean);
      }
      if (TABLE_CAPTION.test(lines[0]) && !TABLE_CAPTION.test(lines[1])) {
        return [lines[0], lines.slice(1).join("\n")];
      }
      if (lines.some((line) => parseListItem(line))) {
        return [lines.join("\n")];
      }
      if (lines.some((line) => line.includes("$$"))) {
        return [lines.join("\n")];
      }
      return [lines.join(" ")];
    })
    .filter(Boolean);
}

function isBareFigureCaption(para: string): boolean {
  const text = para.trim();
  const match = text.match(/^(fig(?:ure)?|algorithm|图|算法)\s*\.?\s*[a-z]?\d+\s*[.:：]?\s*(.*)$/i);
  if (!match) return false;
  const rest = (match[2] || "").trim();
  if (!rest) return true;
  if (
    /^(?:illustrate|show|present|depict|display|visualize|compare|provide|give|summarize|report)s?\b/i.test(
      rest,
    ) ||
    /^(?:比较|展示|显示|给出|验证|说明|描绘)/.test(rest)
  ) {
    return false;
  }
  if (/\b(as shown|see fig|shown in|we |this (paper|section|work)|in this)\b/i.test(text)) {
    return false;
  }
  return rest.split(/[.!?。]/).filter((part) => part.trim().length > 20).length <= 1;
}

function splitCaptionFromBody(para: string): { caption: string | null; rest: string | null } {
  const text = para.trim();
  if (!isTableCaptionPara(text)) return { caption: null, rest: text };
  const bodyStart = text.search(
    /\.\s+(where |let |as shown |we |the final |thus |therefore |hence )/i,
  );
  if (bodyStart >= 0) {
    return {
      caption: text.slice(0, bodyStart + 1).trim(),
      rest: text.slice(bodyStart + 1).trim() || null,
    };
  }
  return { caption: text.replace(/\s+/g, " "), rest: null };
}

type MdTable = Extract<ProseBlock, { type: "table" }>;

function attachTableChrome(blocks: ProseBlock[]): ProseBlock[] {
  const out: ProseBlock[] = [];
  const source = extractTrailingTableTitles(blocks);
  for (let index = 0; index < source.length; index += 1) {
    const block = source[index];
    if (block.type !== "table") {
      out.push(block);
      continue;
    }
    const table: MdTable = { ...block };
    const prev = out.at(-1);
    if (prev?.type === "text" && isTableCaptionPara(prev.text)) {
      table.caption = joinCaption(prev.text, table.caption);
      out.pop();
    }
    while (index + 1 < source.length && source[index + 1]?.type === "text") {
      const next = (source[index + 1] as Extract<ProseBlock, { type: "text" }>).text;
      if (isTableCaptionPara(next)) {
        const nextLabel = normalizeTableLabel(next);
        const currentLabel = normalizeTableLabel(table.caption);
        if (table.caption && nextLabel && currentLabel && nextLabel !== currentLabel) break;
        table.caption = joinCaption(table.caption, next);
        index += 1;
        continue;
      }
      if (isTableNotePara(next) || isCaptionContinuation(table.caption, next)) {
        table.caption = joinCaption(table.caption, next);
        index += 1;
        continue;
      }
      break;
    }
    applyPeeledChrome(table);
    out.push(table);
  }
  return adoptOrphanTableTitles(out);
}

function extractTrailingTableTitles(blocks: ProseBlock[]): ProseBlock[] {
  const out: ProseBlock[] = [];
  for (const block of blocks) {
    if (block.type !== "text") {
      out.push(block);
      continue;
    }
    const peeled = peelTrailingTableTitle(block.text);
    if (!peeled.title) {
      out.push(block);
      continue;
    }
    if (peeled.rest) out.push({ type: "text", text: peeled.rest });
    out.push({ type: "text", text: peeled.title });
  }
  return out;
}

function peelTrailingTableTitle(text: string): { rest: string; title: string | null } {
  const blob = text.trim();
  const match = blob.match(
    /(?:^|[\s-])([A-Z][A-Z0-9,.\s'’()/-]{18,}\((?:[∗*†‡]|\\ast|\\dagger)[\s\S]{8,}\))\s*$/,
  );
  if (!match || !TABLE_NOTE_MARK.test(match[1])) return { rest: blob, title: null };
  const title = collapseWs(match[1]);
  let rest = blob.slice(0, match.index).trim();
  if (blob[match.index] === "-" || /-$/.test(rest)) {
    rest = rest.replace(/-?[a-z]+-?$/i, "").trim();
  }
  return { rest, title };
}

function adoptOrphanTableTitles(blocks: ProseBlock[]): ProseBlock[] {
  const orphans: { index: number; text: string }[] = [];
  const shorts: MdTable[] = [];
  blocks.forEach((block, index) => {
    if (block.type === "text" && isOrphanTableTitle(block.text)) {
      orphans.push({ index, text: block.text });
    }
    if (block.type === "table" && isShortTableCaption(block.caption)) {
      shorts.push(block);
    }
  });
  if (!orphans.length || !shorts.length) return blocks;
  const drop = new Set<number>();
  const count = Math.min(orphans.length, shorts.length);
  for (let index = 0; index < count; index += 1) {
    const table = shorts[index];
    table.caption = joinCaption(table.caption, orphans[index].text);
    applyPeeledChrome(table);
    drop.add(orphans[index].index);
  }
  return blocks.filter((_, index) => !drop.has(index));
}

function applyPeeledChrome(table: MdTable) {
  const peeled = peelTableNote([table.caption, table.note].filter(Boolean).join(" "));
  table.caption = peeled.caption;
  table.note = peeled.note;
}

function peelTableNote(text: string): { caption: string | null; note: string | null } {
  let blob = collapseWs(text);
  if (!blob) return { caption: null, note: null };
  const notes: string[] = [];
  const mx = blob.match(/^(.*?)(?:\s+)(['"“]M\s*\(\s*X\s*\)['"”]?[\s\S]*)$/i);
  if (mx) {
    blob = mx[1].trim();
    notes.unshift(collapseWs(mx[2]));
  }
  const foot = blob.match(
    /^(.*?[.。])\s*(\((?:\\+ast|†|\*|‡|∗|\\dagger)[\s\S]*)$/i,
  );
  if (foot && /^(?:table|tab\.?|表)\b/i.test(foot[1])) {
    blob = foot[1].trim();
    notes.unshift(collapseWs(foot[2]));
  } else {
    const loose = blob.match(/^(.*?)\s+(\((?:\\+ast|†|\*|‡|∗|\\dagger)[\s\S]*)$/i);
    if (loose && /^(?:table|tab\.?|表)\b/i.test(loose[1]) && loose[1].length > 8) {
      blob = loose[1].trim();
      notes.unshift(collapseWs(loose[2]));
    }
  }
  return { caption: blob || null, note: notes.join(" ") || null };
}

function isTableCaptionPara(text: string): boolean {
  const blob = collapseWs(text);
  const match = blob.match(new RegExp(`^(?:table|tab\\.?|表)\\s*\\.?\\s*${TABLE_NUMBER}\\b(.*)$`, "i"));
  if (!match) return false;
  const rest = (match[2] || "").replace(/^[.:：]\s*/, "").trim();
  if (!rest) return true;
  if (TABLE_MENTION_VERB.test(rest)) return false;
  if (/^(?:比较|展示|显示|给出|列出|报告)/.test(rest)) return false;
  return true;
}

function isTableNotePara(text: string): boolean {
  const blob = collapseWs(text);
  if (!blob || blob.length > 700 || isTableCaptionPara(blob)) return false;
  if (/^(?:note|notes)[:：]/i.test(blob)) return true;
  if (/^[-–•*†‡§]\s+\S/.test(blob)) return true;
  if (/^\([^)]{8,}\)\s*$/.test(blob)) return true;
  if (/^['"“]M\s*\(\s*X\s*\)/i.test(blob)) return true;
  if (/^(?:FEATURES ARE|CONCATENATED AS|REPRESENTATION\.)/i.test(blob)) return true;
  if (TABLE_NOTE_MARK.test(blob) && /[)†*‡∗]/.test(blob) && !/[a-z]{5,}/.test(blob.replace(/\$[^$]*\$/g, ""))) {
    return true;
  }
  return false;
}

function isCaptionContinuation(caption: string | null | undefined, next: string): boolean {
  if (!caption) return false;
  const open = (caption.match(/\(/g) ?? []).length + (caption.match(/\\left\(/g) ?? []).length;
  const close = (caption.match(/\)/g) ?? []).length;
  if (open > close) return true;
  if (/[,;:-]$/.test(caption.trim())) return true;
  if (/^(?:FEATURES ARE|CONCATENATED|REPRESENTATION)/i.test(next.trim())) return true;
  return false;
}

function isOrphanTableTitle(text: string): boolean {
  if (isTableCaptionPara(text)) return false;
  const blob = collapseWs(text);
  if (blob.length < 20 || blob.length > 600) return false;
  const core = blob.replace(/^[a-z]{1,16}(?:\s*-\s*)?/, "");
  const letters = core.replace(/[^A-Za-z]/g, "");
  const caps = core.replace(/[^A-Z]/g, "");
  if (letters.length < 12 || caps.length / letters.length < 0.7) return false;
  return TABLE_NOTE_MARK.test(blob);
}

function isShortTableCaption(caption: string | null | undefined): boolean {
  if (!caption) return false;
  return new RegExp(`^(?:table|tab\\.?|表)\\s*\\.?\\s*${TABLE_NUMBER}\\.?$`, "i").test(caption.trim());
}

function joinCaption(left: string | null | undefined, right: string | null | undefined): string | null {
  const parts = [left, right].map((part) => collapseWs(part ?? "")).filter(Boolean);
  return parts.length ? parts.join(" ") : null;
}

function collapseWs(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function romanToInt(text: string): number | null {
  const token = text.toUpperCase();
  if (!/^[IVXLCDM]+$/.test(token) || token.length > 6) return null;
  const value: Record<string, number> = { I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1000 };
  let total = 0;
  for (let index = 0; index < token.length; index += 1) {
    const current = value[token[index]] ?? 0;
    const next = value[token[index + 1]] ?? 0;
    total += current < next ? -current : current;
  }
  if (total < 1 || total > 20) return null;
  return total;
}

function pushText(blocks: ProseBlock[], text: string): void {
  const trimmed = text.trim();
  if (!trimmed) return;
  const last = blocks.at(-1);
  if (last?.type === "text" && last.text === trimmed) return;
  blocks.push({ type: "text", text: trimmed });
}

function clusterLists(blocks: ProseBlock[]): ProseBlock[] {
  const out: ProseBlock[] = [];
  for (const block of blocks) {
    if (block.type !== "text") {
      out.push(block);
      continue;
    }
    for (const part of explodeListParagraph(block.text)) {
      const prev = out.at(-1);
      if (part.type === "list" && prev?.type === "list" && prev.ordered === part.ordered) {
        prev.items.push(...part.items);
      } else if (part.type === "text" && prev?.type === "text" && prev.text === part.text) {
        continue;
      } else {
        out.push(part);
      }
    }
  }
  return out;
}

function explodeListParagraph(text: string): ProseBlock[] {
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length) return [];
  if (!lines.some((line) => parseListItem(line))) {
    const joined = lines.some((line) => line.includes("$$")) ? lines.join("\n") : lines.join(" ");
    return [{ type: "text", text: joined }];
  }
  const out: ProseBlock[] = [];
  let buf: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;

  const flushText = () => {
    if (!buf.length) return;
    const joined = buf.some((line) => line.includes("$$")) ? buf.join("\n") : buf.join(" ");
    out.push({ type: "text", text: joined });
    buf = [];
  };
  const flushList = () => {
    if (list) out.push({ type: "list", ...list });
    list = null;
  };

  for (const line of lines) {
    const item = parseListItem(line);
    if (item) {
      flushText();
      if (list && list.ordered === item.ordered) {
        list.items.push(item.body);
      } else {
        flushList();
        list = { ordered: item.ordered, items: [item.body] };
      }
    } else {
      flushList();
      buf.push(line);
    }
  }
  flushText();
  flushList();
  return out;
}

const REF_PREFIX = /^(?:\[(\d+)\]|\((\d+)\)|(\d+)\.)\s*/;

export function splitReferenceEntries(text: string): string[] {
  let source = text.replace(/\r\n/g, "\n").trim();
  if (!source) return [];
  source = source.replace(/(^|\n)\s*((?:\[\d+\]|\(\d+\)|\d+\.)\s*)\n+/g, "$1$2");
  source = source.replace(/(^|\n)\s*(\d{1,3})\s*\n+(?=\S)/g, "$1[$2] ");
  const entries: string[] = [];
  for (const chunk of source.split(/\n\s*\n+/)) {
    const pieces = chunk
      .split(/(?:\n\s*)+(?=\[\d+\]|\(\d+\)|\d+\.\s+\S)/)
      .flatMap((part) => part.split(/(?<=\.)\s+(?=\[\d+\])/));
    for (const piece of pieces) {
      const body = piece.replace(REF_PREFIX, "").replace(/\s+/g, " ").trim();
      if (body.length >= 8) entries.push(body);
    }
  }
  if (!entries.length) {
    const body = source.replace(REF_PREFIX, "").replace(/\s+/g, " ").trim();
    return body ? [body] : [];
  }
  return entries;
}

export function parseListItem(line: string): { ordered: boolean; body: string } | null {
  const numbered = line.match(LIST_NUMBERED);
  if (numbered) return { ordered: true, body: numbered[1].trim() };
  const bullet = line.match(LIST_BULLET);
  if (bullet) return { ordered: false, body: bullet[1].trim() };
  return null;
}

function isMdTableLine(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed.startsWith("|")) return false;
  return trimmed.indexOf("|", 1) !== -1;
}

function splitStackedTableChunks(lines: string[]): string[] {
  const trimmed = lines.map((line) => line.trim()).filter(Boolean);
  if (!trimmed.length) return [];
  const groups: string[][] = [];
  let current: string[] = [];
  let separators = 0;
  for (const line of trimmed) {
    if (isSeparator(line)) {
      separators += 1;
      if (separators > 1 && current.length >= 2) {
        const header = current.pop()!;
        groups.push(current);
        current = [header, line];
        separators = 1;
        continue;
      }
    }
    current.push(line);
  }
  if (current.length) groups.push(current);
  return groups.map((group) => group.join("\n"));
}

function parseMdTable(blob: string): { headers: string[]; rows: string[][] } | null {
  const rows = blob
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => isMdTableLine(line) && !isSeparator(line))
    .map(splitCells);
  if (rows.length < 2) return null;
  const width = Math.max(...rows.map((row) => row.length));
  const padded = rows.map((row) => {
    const next = row.slice();
    while (next.length < width) next.push("");
    return next;
  });
  return { headers: padded[0], rows: padded.slice(1) };
}

function isSeparator(line: string): boolean {
  const trimmed = line.replace(/\|/g, "").replace(/[:\s-]/g, "");
  return trimmed.length === 0 && /-/.test(line);
}

function splitCells(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((cell) => cell.trim());
}
