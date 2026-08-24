import katex from "katex";
import "katex/dist/katex.min.css";

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

export function MathText({
  text,
  inline = false,
  cites = true,
  className,
}: {
  text: string;
  inline?: boolean;
  cites?: boolean;
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
        const hasCopy = block.some((piece) => piece.value.trim());
        if (!hasCopy) return null;
        const Tag = inline ? "span" : "p";
        return (
          <Tag key={index}>
            {block.map((piece, pieceIndex) =>
              piece.type === "text" ? (
                piece.value ? (
                  cites ? (
                    <TextWithCites key={pieceIndex} value={piece.value} />
                  ) : (
                    <span key={pieceIndex}>{piece.value}</span>
                  )
                ) : null
              ) : (
                <RenderedMath key={pieceIndex} value={piece.value} />
              ),
            )}
          </Tag>
        );
      });

  if (!className) return <>{nodes}</>;
  const Wrap = inline ? "span" : "div";
  return <Wrap className={className}>{nodes}</Wrap>;
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
  const displayMode = display || prepared.forceDisplay;
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

/** Docling / KaTeX：去掉误插入的 $，align 片段包进 aligned。 */
export function prepareKatex(value: string): { body: string; forceDisplay: boolean } {
  let body = value.replace(/\$/g, "").replace(/\uFFFD/g, "").trim();
  body = body.replace(/\\([A-Za-z]+)\s+\{/g, "\\$1{");
  body = body.replace(/([_^])\{([^{}\\]*)\\\}/g, "$1{$2}");
  body = body.replace(/([A-Za-z])\}([a-z]{1,4})\s*\^\{((?:\\times|\\cdot|\\otimes)\s*)/g, "$1_$2 $3");
  body = body.replace(/(?<!mathbb\{)(?<![A-Za-z\\])R\^\{/g, "\\mathbb{R}^{");
  body = body.replace(/(?<![A-Za-z\\])([WFTX])([SPRNIT])_\{?([A-Za-z0-9]+)\}?/g, "$1^{$2}_{$3}");
  body = body.replace(/(?<![A-Za-z\\])R([A-Z])\s*(?:\\times|×)\s*([A-Z])/g, "\\mathbb{R}^{$1 \\times $2}");
  body = body.replace(/(\}_{[A-Za-z0-9]+}),(?=[A-Z])/g, "$1, ");
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
        buf += source.slice(index);
        break;
      }
      flushText();
      pieces.push({ type: "display", value: source.slice(start, close).trim() });
      index = skipExtraDollars(close + 2);
      continue;
    }
    if (source[index] === "$") {
      const close = source.indexOf("$", index + 1);
      if (close >= 0 && !source.startsWith("$$", close)) {
        flushText();
        pieces.push({ type: "inline", value: source.slice(index + 1, close).trim() });
        index = close + 1;
        continue;
      }
    }
    buf += source[index];
    index += 1;
  }
  flushText();
  return pieces.length ? pieces : [{ type: "text", value: source }];
}
