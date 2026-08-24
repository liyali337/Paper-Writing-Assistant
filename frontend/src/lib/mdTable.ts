export type ProseBlock =
  | { type: "text"; text: string }
  | { type: "table"; headers: string[]; rows: string[][]; caption: string | null }
  | { type: "figure"; figureId: string }
  | { type: "list"; ordered: boolean; items: string[] };

export type FigureRef = {
  figure_id: string;
  kind: string;
  label: string | null;
  caption: string | null;
};

const TABLE_CAPTION = /^(table|tab\.?)\s*[a-z]?\d+/i;
const FIGURE_CAPTION = /^(fig(?:ure)?|algorithm)\s*\.?\s*[a-z]?\d+/i;
const FORMULA_MARK = /<!--fig:(fig-\d+)-->/g;
const FIGURE_MENTION = /\b(fig(?:ure)?|algorithm)\s*\.?\s*([A-Z]?\d+)\b/gi;
const LIST_NUMBERED = /^(?:\(\d+\)|\d+\.(?!\d))\s+(\S[\s\S]*)$/;
const LIST_BULLET = /^[-–•*]\s+(\S[\s\S]*)$/;

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
  return clusterLists(blocks);
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

function mentionsIn(block: ProseBlock, byLabel: Map<string, string>): string[] {
  const text =
    block.type === "text"
      ? block.text
      : block.type === "list"
        ? block.items.join("\n")
        : block.type === "table"
          ? [block.caption, ...block.headers].filter(Boolean).join(" ")
          : "";
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
      const table = parseMdTable(blob);
      if (table) {
        blocks.push({ type: "table", ...table, caption: null });
        continue;
      }
    }
    for (const para of toParagraphs(blob)) {
      if (FIGURE_CAPTION.test(para.trim())) continue;
      const split = splitCaptionFromBody(para);
      if (split.caption && blocks.at(-1)?.type === "table") {
        const prev = blocks[blocks.length - 1];
        if (prev.type === "table" && !prev.caption) prev.caption = split.caption;
        if (split.rest) pushText(blocks, split.rest);
        continue;
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
      if (FIGURE_CAPTION.test(lines[0])) {
        return [lines.slice(1).join("\n")].filter(Boolean);
      }
      if (TABLE_CAPTION.test(lines[0]) && !TABLE_CAPTION.test(lines[1])) {
        return [lines[0], lines.slice(1).join("\n")];
      }
      if (lines.some((line) => parseListItem(line))) {
        return [lines.join("\n")];
      }
      return [lines.join(" ")];
    })
    .filter(Boolean);
}

function splitCaptionFromBody(para: string): { caption: string | null; rest: string | null } {
  const text = para.trim();
  if (!TABLE_CAPTION.test(text)) return { caption: null, rest: text };
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
    return [{ type: "text", text: lines.join(" ") }];
  }
  const out: ProseBlock[] = [];
  let buf: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;

  const flushText = () => {
    if (!buf.length) return;
    out.push({ type: "text", text: buf.join(" ") });
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
