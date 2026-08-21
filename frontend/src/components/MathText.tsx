import katex from "katex";
import "katex/dist/katex.min.css";

type Piece =
  | { type: "text"; value: string }
  | { type: "inline"; value: string }
  | { type: "display"; value: string };

export function MathText({ text }: { text: string }) {
  const pieces = splitMath(text);
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

  return (
    <>
      {blocks.map((block, index) => {
        const first = block[0];
        if (block.length === 1 && first.type === "display") {
          return <RenderedMath key={index} value={first.value} display />;
        }
        const hasCopy = block.some((piece) => piece.value.trim());
        if (!hasCopy) return null;
        return (
          <p key={index}>
            {block.map((piece, pieceIndex) =>
              piece.type === "text" ? (
                piece.value ? <span key={pieceIndex}>{piece.value}</span> : null
              ) : (
                <RenderedMath key={pieceIndex} value={piece.value} />
              ),
            )}
          </p>
        );
      })}
    </>
  );
}

function RenderedMath({ value, display = false }: { value: string; display?: boolean }) {
  try {
    const html = katex.renderToString(value, {
      displayMode: display,
      throwOnError: false,
      output: "html",
    });
    const Tag = display ? "div" : "span";
    return (
      <Tag
        className={display ? "math-display" : "math-inline"}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  } catch {
    return display ? <div className="math-display">{value}</div> : <span>{value}</span>;
  }
}

export function splitMath(text: string): Piece[] {
  const pieces: Piece[] = [];
  const source = text.replace(/\r\n/g, "\n");
  const pattern = /\$\$([\s\S]+?)\$\$|\$([^$\n]+)\$/g;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(source))) {
    if (match.index > last) {
      pieces.push({ type: "text", value: source.slice(last, match.index) });
    }
    if (match[1] != null) {
      pieces.push({ type: "display", value: match[1].trim() });
    } else if (match[2] != null) {
      pieces.push({ type: "inline", value: match[2].trim() });
    }
    last = match.index + match[0].length;
  }
  if (last < source.length) {
    pieces.push({ type: "text", value: source.slice(last) });
  }
  return pieces.length ? pieces : [{ type: "text", value: source }];
}
