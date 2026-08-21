import katex from "katex";
import "katex/dist/katex.min.css";

type Piece =
  | { type: "text"; value: string }
  | { type: "inline"; value: string }
  | { type: "display"; value: string };

export function MathText({ text }: { text: string }) {
  const pieces = splitMath(text);
  return (
    <>
      {pieces.map((piece, index) => {
        if (piece.type === "text") {
          return piece.value ? <span key={index}>{piece.value}</span> : null;
        }
        try {
          const html = katex.renderToString(piece.value, {
            displayMode: piece.type === "display",
            throwOnError: false,
            output: "html",
          });
          return (
            <span
              key={index}
              className={piece.type === "display" ? "math-display" : "math-inline"}
              dangerouslySetInnerHTML={{ __html: html }}
            />
          );
        } catch {
          return <span key={index}>{piece.value}</span>;
        }
      })}
    </>
  );
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
