export function splitAskParagraphs(text: string): string[] {
  let normalized = (text || "").replace(/\r\n/g, "\n");
  normalized = normalized.replace(/([。！？])[ \t]*(?=\*\*[一二三四五六七八九十0-9])/g, "$1\n\n");
  normalized = normalized.replace(
    /(\*\*[一二三四五六七八九十0-9]+[、.．][^*]*\*\*)[ \t]+(?=\S)/g,
    "$1\n\n",
  );
  normalized = normalized.replace(/\n{3,}/g, "\n\n");
  return normalized.split(/\n\n+/).filter((part) => part.trim().length > 0);
}

export type AskInline =
  | { type: "text"; value: string }
  | { type: "bold"; value: string };

export function splitAskBold(text: string): AskInline[] {
  const parts: AskInline[] = [];
  const pattern = /\*\*([^*]+)\*\*/g;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text))) {
    if (match.index > last) {
      parts.push({ type: "text", value: text.slice(last, match.index) });
    }
    parts.push({ type: "bold", value: match[1] });
    last = match.index + match[0].length;
  }
  if (last < text.length) {
    parts.push({ type: "text", value: text.slice(last) });
  }
  return parts.length ? parts : [{ type: "text", value: text }];
}

export function isAskHeading(text: string): boolean {
  const trimmed = text.trim();
  return /^\*\*[^*]+\*\*$/.test(trimmed) || /^[一二三四五六七八九十0-9]+[、.．]/.test(trimmed);
}

export function isMarkerOnlyLatex(value: string): boolean {
  const body = value.replace(/\$/g, "").trim();
  return /^(?:\^\{?(?:\\circ|circ|\\ast|\*|\\dagger|\\ddagger|\\bullet)\}?|\\circ|\\bullet|\\ast|\\dagger|\\ddagger)$/i.test(
    body,
  );
}
