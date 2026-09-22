import katex from "katex";
import "katex/dist/katex.min.css";

import { isAskHeading, isMarkerOnlyLatex, splitAskBold, splitAskParagraphs } from "../lib/askProse";

type Piece =
  | { type: "text"; value: string }
  | { type: "inline"; value: string }
  | { type: "display"; value: string };

const BEGIN_ENV = /^\\begin\{(align\*?|equation\*?|gather\*?|multline\*?|eqnarray\*?|aligned)\}/;

const ACCENT_CMD: Record<string, string> = {
  "¨": "ddot",
  "ˆ": "hat",
  "˜": "tilde",
  "¯": "bar",
  "´": "acute",
  "˙": "dot",
  "`": "grave",
  "˚": "mathring",
};

/** PDF 常把 \ddot{x} 抽成孤立 ¨ 再跟 $x...$。先并回数学体，再切分。 */
function mergeOrphanAccents(text: string): string {
  return text.replace(
    /([¨ˆ˜¯´˙`˚])\s*\$([A-Za-z])([^$]*)\$/g,
    (all, mark: string, base: string, rest: string) => {
      const cmd = ACCENT_CMD[mark];
      return cmd ? `$\\${cmd}{${base}}${rest}$` : all;
    },
  );
}

export type DisplaySegment = { prose: string; display: string | null };

/** 按独立公式（$$…$$）切段，便于中英对照时「原文 → 译文 → 公式」。 */
export function segmentByDisplayMath(text: string): DisplaySegment[] {
  const pieces = splitMath(mergeOrphanAccents(text));
  const segments: DisplaySegment[] = [];
  let prose = "";

  const push = (display: string | null) => {
    segments.push({ prose, display });
    prose = "";
  };

  for (const piece of pieces) {
    if (piece.type === "display") {
      push(piece.value);
    } else if (piece.type === "inline") {
      prose += `$${piece.value}$`;
    } else {
      prose += piece.value;
    }
  }
  if (prose.trim() || segments.length === 0) {
    push(null);
  }
  return segments;
}

export function MathDisplay({ value }: { value: string }) {
  return <RenderedMath value={value} display />;
}

export function MathText({
  text,
  inline = false,
  cites = true,
  layout = "prose",
  className,
}: {
  text: string;
  inline?: boolean;
  cites?: boolean;
  layout?: "prose" | "ask";
  className?: string;
}) {
  const pieces = splitMath(mergeOrphanAccents(text));
  const blocks: Piece[][] = [];
  let current: Piece[] = [];
  for (const piece of pieces) {
    if (piece.type === "display") {
      if (current.length) {
        blocks.push(current);
        current = [];
      }
      blocks.push([piece]);
    } else {
      current.push(piece);
    }
  }
  if (current.length) blocks.push(current);

  const nodes = blocks.map((block, index) => {
        const first = block[0];
        if (block.length === 1 && first.type === "display") {
          return <RenderedMath key={index} value={first.value} display />;
        }
        const hasCopy = block.some((piece) => {
          if (piece.type !== "text" && isMarkerOnlyLatex(piece.value)) return false;
          return Boolean(piece.value.trim());
        });
        if (!hasCopy) return null;
        const Tag = inline ? "span" : "p";
        if (layout === "ask" && !inline) {
          return (
            <AskProseBlock
              key={index}
              block={block}
              cites={cites}
            />
          );
        }
        return (
          <Tag key={index}>
            {block.map((piece, pieceIndex) => renderPiece(piece, pieceIndex, cites))}
          </Tag>
        );
      });

  if (!className) return <>{nodes}</>;
  const Wrap = inline ? "span" : "div";
  return <Wrap className={className}>{nodes}</Wrap>;
}

function renderPiece(piece: Piece, pieceIndex: number, cites: boolean) {
  if (piece.type === "text") {
    if (!piece.value) return null;
    return cites ? (
      <TextWithCites key={pieceIndex} value={piece.value} />
    ) : (
      <span key={pieceIndex}>{piece.value}</span>
    );
  }
  if (isMarkerOnlyLatex(piece.value)) return null;
  return <RenderedMath key={pieceIndex} value={piece.value} />;
}

function AskProseBlock({ block, cites }: { block: Piece[]; cites: boolean }) {
  const rows: Piece[][] = [[]];
  for (const piece of block) {
    if (piece.type === "text") {
      const parts = splitAskParagraphs(piece.value);
      parts.forEach((part, index) => {
        if (index > 0) rows.push([]);
        if (part) rows[rows.length - 1].push({ type: "text", value: part });
      });
      continue;
    }
    if (piece.type !== "display" && isMarkerOnlyLatex(piece.value)) continue;
    rows[rows.length - 1].push(piece);
  }
  return (
    <>
      {rows.map((row, rowIndex) => {
        if (!row.some((piece) => piece.value.trim())) return null;
        const plain = row
          .filter((piece) => piece.type === "text")
          .map((piece) => piece.value)
          .join("");
        const heading = row.every((piece) => piece.type === "text") && isAskHeading(plain.trim());
        return (
          <p key={rowIndex} className={heading ? "ask-heading" : undefined}>
            {row.map((piece, pieceIndex) =>
              piece.type === "text" ? (
                <AskInlineText key={pieceIndex} value={piece.value} cites={cites} />
              ) : (
                renderPiece(piece, pieceIndex, cites)
              ),
            )}
          </p>
        );
      })}
    </>
  );
}

function AskInlineText({ value, cites }: { value: string; cites: boolean }) {
  return (
    <>
      {splitAskBold(value).map((part, index) => {
        const body = cites ? <TextWithCites value={part.value} /> : part.value;
        if (part.type === "bold") {
          return <strong key={index}>{body}</strong>;
        }
        return <span key={index}>{body}</span>;
      })}
    </>
  );
}

const CITE_NUM = /\[(\d+(?:\s*[,;–-]\s*\d+)*)\]/g;

function TextWithCites({ value }: { value: string }) {
  const parts: Array<{ type: "text" | "cite"; value: string }> = [];
  CITE_NUM.lastIndex = 0;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = CITE_NUM.exec(value))) {
    if (match.index > last) {
      parts.push({ type: "text", value: value.slice(last, match.index) });
    }
    parts.push({ type: "cite", value: match[1].replace(/\s+/g, "") });
    last = match.index + match[0].length;
  }
  if (last < value.length) parts.push({ type: "text", value: value.slice(last) });
  if (!parts.some((part) => part.type === "cite")) {
    return <span>{value}</span>;
  }
  return (
    <span>
      {parts.map((part, index) =>
        part.type === "cite" ? (
          <sup className="cite-ref" key={index}>
            {part.value}
          </sup>
        ) : (
          <span key={index}>{part.value}</span>
        ),
      )}
    </span>
  );
}

function RenderedMath({ value, display = false }: { value: string; display?: boolean }) {
  const prepared = prepareKatex(value);
  // 行内公式即使被 prepareKatex 标成 display，也不能输出 <div>（会拆掉 <p> 后半句）
  const displayMode = display;
  try {
    const html = katex.renderToString(prepared.body, {
      displayMode,
      throwOnError: false,
      output: "html",
      strict: "ignore",
    });
    const Tag = displayMode ? "div" : "span";
    return (
      <Tag
        className={displayMode ? "math-display" : "math-inline"}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  } catch {
    return displayMode ? <div className="math-display">{value}</div> : <span>{value}</span>;
  }
}

const DOUBLE_SUP =
  /([A-Za-z](?:_\{[^{}]+\}|_[A-Za-z0-9]+)?\s*\^\{[^{}]+\})\s+\^\{([^{}]+)\}/g;
const ORPHAN_SUP = /(^|,\s*)(?:\{\})?\^\{([^{}]+)\}/g;

/** PDF 丢掉矩阵字母后会留下 _{r}h_{r}、z^{v} ^{Qv}，KaTeX 无法解析。 */
export function repairPdfScripts(body: string): string {
  let cleaned = body.replace(/\\ /g, " ");
  cleaned = cleaned.replace(DOUBLE_SUP, "$1 W^{$2}");
  cleaned = cleaned.replace(ORPHAN_SUP, "$1 W^{$2}");
  cleaned = repairLeadingDictTerms(cleaned);
  cleaned = cleaned.replace(/=([A-Za-z])/g, "= $1");
  cleaned = cleaned.replace(/\s{2,}/g, " ").trim();
  return cleaned;
}

const LEAD_DICT_TERM =
  /(?<![A-Za-z0-9])(?:\{\})?_\{([A-Za-z0-9]+)\}\s*([A-Za-z])_\{([A-Za-z0-9]+)\}/;

function repairLeadingDictTerms(body: string): string {
  if (!LEAD_DICT_TERM.test(body)) return body;

  const repairRow = (row: string) => {
    let dictLetter: string | null = null;
    const out = row.replace(
      new RegExp(LEAD_DICT_TERM.source, "g"),
      (_all, sub: string, base: string, hsub: string) => {
        if (!dictLetter) dictLetter = sub[0].toUpperCase();
        return `${dictLetter}_{${sub}} ${base}_{${hsub}}`;
      },
    );
    return out.replace(/([a-z]_\{[^{}]+\})\s+([A-Z]_\{)/g, "$1 + $2");
  };

  const parts: string[] = [];
  let buf = "";
  let depth = 0;
  for (const char of body) {
    if (char === "{") depth += 1;
    else if (char === "}") depth = Math.max(0, depth - 1);
    if (char === "," && depth === 0) {
      parts.push(buf);
      buf = "";
      continue;
    }
    buf += char;
  }
  parts.push(buf);
  const hits = parts.filter((part) =>
    /(?<![A-Za-z0-9])(?:\{\})?_\{[A-Za-z0-9]+\}\s*[A-Za-z]_\{[A-Za-z0-9]+\}/.test(part),
  ).length;
  if (hits >= 2) {
    return parts
      .map((part) => part.trim())
      .filter(Boolean)
      .map(repairRow)
      .join(", ");
  }
  return repairRow(body);
}

/** Docling / KaTeX：去掉误插入的 $，align 片段包进 aligned。 */
export function prepareKatex(value: string): { body: string; forceDisplay: boolean } {
  let body = value.replace(/\$/g, "").replace(/\uFFFD/g, "").trim();
  body = repairPdfScripts(body);
  body = body.replace(/α/g, "\\alpha ");
  body = body.replace(/β/g, "\\beta ");
  body = body.replace(/γ/g, "\\gamma ");
  body = body.replace(/θ/g, "\\theta ");
  body = body.replace(/Δ/g, "\\Delta ");
  body = body.replace(/λ/g, "\\lambda ");
  body = body.replace(/μ/g, "\\mu ");
  body = body.replace(/π/g, "\\pi ");
  body = body.replace(/σ/g, "\\sigma ");
  body = body.replace(/φ/g, "\\phi ");
  body = body.replace(/ω/g, "\\omega ");
  body = body.replace(/⊕/g, "\\oplus ");
  body = body.replace(/∈/g, "\\in ");
  body = body.replace(/ℝ/g, "\\mathbb{R}");
  body = body.replace(/×/g, "\\times ");
  body = body.replace(/\\([A-Za-z]+)\s+\{/g, "\\$1{");
  body = body.replace(/([_^])\{([^{}\\]*)\\\}/g, "$1{$2}");
  body = body.replace(/\^\{\{([^{}]+)\}\}/g, "^{$1}");
  body = body.replace(/([A-Za-z])\}([a-z]{1,4})\s*\^\{((?:\\times|\\cdot|\\otimes)\s*)/g, "$1_$2 $3");
  body = body.replace(/(?<!mathbb\{)(?<![A-Za-z\\])R\^\{/g, "\\mathbb{R}^{");
  body = body.replace(
    /(?<!mathbb\{)(?<![A-Za-z\\])R(\d+)\s*(?:\\times|×)\s*([A-Za-z](?:_\{[^{}]+\}|_[A-Za-z0-9]+)?)/g,
    "\\mathbb{R}^{$1 \\times $2}",
  );
  body = body.replace(/(?<![A-Za-z\\])([WFTX])([SPRNIT])_\{?([A-Za-z0-9]+)\}?/g, "$1^{$2}_{$3}");
  body = body.replace(/(?<![A-Za-z\\])R([A-Z])\s*(?:\\times|×)\s*([A-Z])/g, "\\mathbb{R}^{$1 \\times $2}");
  body = body.replace(
    /(?:\\mathbb\{R\}|(?<!mathbb\{)(?<![A-Za-z\\])R)_\{([A-Za-z])\}\^\s*\{([^{}]*?\\times\s*)([A-Za-z])\}/g,
    "\\mathbb{R}^{$2$3_$1}",
  );
  body = body.replace(
    /(?:\\mathbb\{R\}|(?<!mathbb\{)(?<![A-Za-z\\])R)_([A-Za-z])\^\s*\{([^{}]*?\\times\s*)([A-Za-z])\}/g,
    "\\mathbb{R}^{$2$3_$1}",
  );
  body = body.replace(
    /(?:\\mathbb\{R\})\^\s*\{([^{}]*?\\times\s*)([A-Za-z])\}_\{([A-Za-z])\}/g,
    "\\mathbb{R}^{$1$2_$3}",
  );
  body = body.replace(
    /(?:\\mathbb\{R\})\^\s*\{([^{}]*?\\times\s*)([A-Za-z])\}_([A-Za-z])(?![A-Za-z0-9{])/g,
    "\\mathbb{R}^{$1$2_$3}",
  );
  body = body.replace(
    /(?<![A-Za-z\\])([QFHWXZPK])([A-Z])([A-Z])\s*\\in\s*(\\mathbb\{R\})(?=(?:_\{?[A-Za-z]\}?)?\s*\^\{[^{}]{0,80}\\times)/g,
    (full, base, sup, sub, bb) => {
      const token = `${base}${sup}${sub}`;
      const skip = new Set([
        "RNN",
        "CNN",
        "GAN",
        "MLP",
        "SVD",
        "PCA",
        "RGB",
        "NLP",
        "NMS",
        "ROI",
        "FPS",
        "GPU",
        "CPU",
        "BERT",
        "LSTM",
        "VAE",
        "GNN",
        "ViT",
      ]);
      if (skip.has(token) || [...token].every((ch) => "QKV".includes(ch))) return full;
      return `${base}_{${sub}}^{${sup}} \\in ${bb}`;
    },
  );
  body = body.replace(/(\}_{[A-Za-z0-9]+}),(?=[A-Z])/g, "$1, ");
  body = body.replace(/(\\[A-Za-z]+)\s+([_^])/g, "$1$2");
  if (body.endsWith(".") && !body.endsWith("\\ldots")) body = body.slice(0, -1);
  const tagged = body.match(/[.,]?\s*\((\d+[a-z]?)\)\s*$/);
  if (tagged) {
    body = `${body.slice(0, tagged.index).replace(/[ .,]+$/, "")} \\tag{${tagged[1]}}`;
  }
  if (/\\begin\{(?:aligned|align\*?|gather|split|cases|array|equation\*?|multline)\}/.test(body)) {
    return { body, forceDisplay: true };
  }
  if (body.includes("&=")) {
    return { body: `\\begin{aligned}${body}\\end{aligned}`, forceDisplay: true };
  }
  return { body, forceDisplay: false };
}

export function splitMath(text: string): Piece[] {
  const source = text.replace(/\r\n/g, "\n");
  const pieces: Piece[] = [];
  let buf = "";
  let index = 0;

  const flushText = () => {
    if (buf) {
      pieces.push({ type: "text", value: buf });
      buf = "";
    }
  };

  const skipExtraDollars = (pos: number) => {
    while (pos < source.length) {
      let cursor = pos;
      while (cursor < source.length && " \t\n".includes(source[cursor])) cursor += 1;
      if (source.startsWith("$$", cursor)) {
        pos = cursor + 2;
        continue;
      }
      break;
    }
    return pos;
  };

  const skipEmptyTrailers = (pos: number) => {
    while (pos < source.length) {
      let cursor = pos;
      while (cursor < source.length && " \t\n".includes(source[cursor])) cursor += 1;
      if (!source.startsWith("$$", cursor)) break;
      let next = cursor + 2;
      while (next < source.length && " \t\n".includes(source[next])) next += 1;
      if (next >= source.length || source.startsWith("$$", next)) {
        pos = cursor + 2;
        continue;
      }
      break;
    }
    return pos;
  };

  while (index < source.length) {
    if (source.startsWith("\\$", index)) {
      buf += "\\$";
      index += 2;
      continue;
    }
    const env = source.slice(index).match(BEGIN_ENV);
    if (env) {
      const close = `\\end{${env[1]}}`;
      const end = source.indexOf(close, index + env[0].length);
      if (end >= 0) {
        flushText();
        pieces.push({ type: "display", value: source.slice(index, end + close.length).trim() });
        index = end + close.length;
        continue;
      }
    }
    if (source.startsWith("\\[", index)) {
      const end = source.indexOf("\\]", index + 2);
      if (end >= 0) {
        flushText();
        pieces.push({ type: "display", value: source.slice(index + 2, end).trim() });
        index = end + 2;
        continue;
      }
    }
    if (source.startsWith("\\(", index)) {
      const end = source.indexOf("\\)", index + 2);
      if (end >= 0) {
        flushText();
        pieces.push({ type: "inline", value: source.slice(index + 2, end).trim() });
        index = end + 2;
        continue;
      }
    }
    if (source.startsWith("$$", index)) {
      const start = skipExtraDollars(index + 2);
      const close = source.indexOf("$$", start);
      if (close < 0) {
        buf += source[index];
        index += 1;
        continue;
      }
      flushText();
      pieces.push({ type: "display", value: source.slice(start, close).trim() });
      index = skipEmptyTrailers(close + 2);
      continue;
    }
    if (source[index] === "$") {
      const close = source.indexOf("$", index + 1);
      if (close >= 0) {
        const body = source.slice(index + 1, close);
        if (/[\u4e00-\u9fff]/.test(body)) {
          buf += body;
          index = close + 1;
          continue;
        }
        flushText();
        pieces.push({ type: "inline", value: body.trim() });
        index = close + 1;
        continue;
      }
    }
    buf += source[index];
    index += 1;
  }
  flushText();
  return glueShatteredMath(pieces.length ? pieces : [{ type: "text", value: source }]);
}

const ASSIGN_START =
  /[A-Za-z](?:_\{[^{}]+\}|\^\{\{?[^{}]+\}\})+\s*=/;

function isMostlyLatexAssignment(text: string): boolean {
  if (/[\u4e00-\u9fff]|其中|式中|\bwhere\b/i.test(text)) return false;
  const trimmed = text.replace(/\$/g, "").trim();
  if (!trimmed || !ASSIGN_START.test(trimmed)) return false;
  const plain = trimmed
    .replace(/\\[A-Za-z]+/g, " ")
    .replace(/\{[^{}]*\}/g, " ")
    .replace(/[_^\\]/g, " ");
  return (plain.match(/\b[A-Za-z]{5,}\b/g) ?? []).length === 0;
}

function isAssignmentTail(piece: Piece): boolean {
  const value = piece.value.trim();
  if (!value) return true;
  if (/[\u4e00-\u9fff]|其中|式中|\bwhere\b/i.test(value)) return false;
  if (piece.type === "display") {
    return /^\s*(?:\\in|∈)/.test(value) || /\\tag\{/.test(value) || /^\\times/.test(value);
  }
  if (piece.type === "inline") {
    if (/\\mathbb|\s\\in\s|\\in\s*\\mathbb/.test(value)) return false;
    return (
      /^(?:\\(?:alpha|theta|times|cdot|in|oplus|mathrm)|[αθ×∈])/.test(value) ||
      (value.length <= 24 && /[_^\\]/.test(value) && !/\s/.test(value))
    );
  }
  return /[_^]\{/.test(value) || /^[(),;:\s]+$/.test(value);
}

/** PDF 常把展示公式拆成「赋值正文 + $\alpha$ + $$\\in$$」。拼回一块给 KaTeX。 */
function glueShatteredMath(pieces: Piece[]): Piece[] {
  const out: Piece[] = [];
  let index = 0;
  while (index < pieces.length) {
    const piece = pieces[index];
    if (piece.type === "text" && isMostlyLatexAssignment(piece.value)) {
      const chunk: Piece[] = [piece];
      let cursor = index + 1;
      while (cursor < pieces.length && isAssignmentTail(pieces[cursor])) {
        chunk.push(pieces[cursor]);
        cursor += 1;
      }
      const body = chunk
        .map((part) => part.value.replace(/\$/g, "").trim())
        .filter(Boolean)
        .join(" ");
      out.push({ type: "display", value: body });
      index = cursor;
      continue;
    }
    out.push(piece);
    index += 1;
  }
  return out;
}
