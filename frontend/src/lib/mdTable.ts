export type ProseBlock =
  | { type: "text"; text: string }
  | { type: "table"; headers: string[]; rows: string[][]; caption: string | null }
  | { type: "figure"; figureId: string };

const TABLE_CAPTION = /^(table|tab\.?)\s*[a-z]?\d+/i;
const FIGURE_CAPTION = /^(fig(?:ure)?|algorithm)\s*\.?\s*[a-z]?\d+/i;
const FORMULA_MARK = /<!--fig:(fig-\d+)-->/g;

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
  return blocks;
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
      return [lines.join("\n")];
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
