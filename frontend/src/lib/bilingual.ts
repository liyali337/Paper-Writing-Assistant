const HAN = /[\u4e00-\u9fff]/;

/** 中文夹着的裸 LaTeX（没包 $）收成行内公式。先整包集合，再包标识符，避免拆开 `\in \{\mathrm{rgb},…\}`。 */
const BARE_INLINE_PATTERNS: RegExp[] = [
  /(?<![$\\])((?:[A-Za-z]\s+)?\\in\s*\\\{(?:[^{}]|\{[^{}]{0,40}\})*\\\})/g,
  /(?<![$\\])(\\frac\{[^{}]{1,40}\}\{[^{}]{1,40}\})/g,
  /(?<![$\\])(\([A-Za-z](?:[_^](?:\{(?:[^{}]|\{[^{}]{0,40}\})+\}|[A-Za-z0-9]+))(?:,\s*[A-Za-z](?:[_^](?:\{(?:[^{}]|\{[^{}]{0,40}\})+\}|[A-Za-z0-9]+)))+\))/g,
  /(?<![$\\])([A-Za-z](?:[_^](?:\{(?:[^{}]|\{[^{}]{0,40}\})+\}|[A-Za-z0-9'\\]+))+)/g,
  /(?<![$\\])(\\\{(?:[^{}]|\{[^{}]{0,40}\})*\\\})/g,
  /(?<![$\\])(\\[A-Za-z]+(?:\s*\{[^{}]{0,80}\})+(?:[_^](?:\{[^{}]{0,40}\}|[A-Za-z0-9]+))*)/g,
  /(?<![$\\])(\\(?:in|cdot|times|mid|oplus|otimes|leq|geq|neq|pm|infty|ldots|dots)(?![A-Za-z]))/g,
];

/** 只剩独立公式、没有正文。这类块不该消耗一段中文译文。 */
export function isMathOnlyText(text: string): boolean {
  const stripped = text
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/\$[^$\n]+\$/g, " ")
    .replace(/\\begin\{[a-zA-Z*]+\}[\s\S]*?\\end\{[a-zA-Z*]+\}/g, " ")
    .replace(/\\[A-Za-z]+/g, " ")
    .replace(/[\s$.,;:(){}[\]\\|&^_+\-−×·≤≥≠=]+/g, "");
  return stripped.length === 0;
}

/** 去掉译文开头被切断的公式尾巴，例如 `|T)。`；含 `\\in` / `\\mathbb` 的完整行内公式要保留。 */
export function stripZhMathDebris(text: string): string {
  return unwrapHanMath(
    text
      .replace(/^\s*(?:\$\$[\s\S]{0,48}?\$\$\s*)+/g, "")
      .replace(/^\s*(?:\$(?!\\)[^$\\\n\u4e00-\u9fff]{1,16}\$\s*)+/g, "")
      .replace(/^\s*(?:\\?\|[A-Za-z0-9]{1,8}\\?\)[。.]?\s*)+/g, "")
      .trimStart(),
  );
}

/** 译文里的独立公式已用英文渲染，中文段再带公式源码会重复出现。 */
export function stripDisplayMath(text: string): string {
  const withoutBlocks = text.replace(/\$\$[\s\S]*?\$\$/g, " ").replace(/[ \t]+\n/g, "\n").trim();
  return dropDuplicateLatexProse(withoutBlocks);
}

/** 公式源码 +「其中…」拆开；源码丢掉，只留中文解释。 */
export function dropDuplicateLatexProse(text: string): string {
  const split = splitZhFormulaAndWhere(text);
  if (split.where) return [split.rest, split.where].filter(Boolean).join("\n\n");
  if (split.formula && !HAN.test(split.rest)) return "";
  return split.rest || text.trim();
}

export function splitZhFormulaAndWhere(text: string): {
  formula: string | null;
  where: string | null;
  rest: string;
} {
  const trimmed = (text || "").trim();
  if (!trimmed) return { formula: null, where: null, rest: "" };
  const peeled = peelTrailingWhereZh(trimmed);
  if (peeled.where) {
    const lead = peeled.lead.trim();
    if (!lead) return { formula: null, where: peeled.where, rest: "" };
    if (isLatexAssignmentProse(lead)) {
      return { formula: lead, where: peeled.where, rest: "" };
    }
    return { formula: null, where: peeled.where, rest: lead };
  }
  if (isLatexAssignmentProse(trimmed) && !HAN.test(trimmed)) {
    return { formula: trimmed, where: null, rest: "" };
  }
  return { formula: null, where: null, rest: trimmed };
}

function isLatexAssignmentProse(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) return false;
  if (/\\tag\{/.test(trimmed)) return true;
  const cmds = trimmed.match(/\\[A-Za-z]+/g)?.length ?? 0;
  if (cmds < 2) return false;
  const han = (trimmed.match(/[\u4e00-\u9fff]/g) ?? []).length;
  return han === 0 || (cmds >= 3 && han < 4);
}

/** 先跳过 $$…$$，再处理行内 $…$，避免把独立公式的收尾 `$$` 拆成行内 `$`。 */
function mapDollarSpans(
  text: string,
  onInline: (body: string) => string,
  onText: (body: string) => string = (value) => value,
): string {
  if (!text) return text;
  let out = "";
  let index = 0;
  while (index < text.length) {
    if (text.startsWith("\\$", index)) {
      out += "\\$";
      index += 2;
      continue;
    }
    if (text.startsWith("$$", index)) {
      const close = text.indexOf("$$", index + 2);
      if (close < 0) {
        out += onText(text.slice(index));
        break;
      }
      out += text.slice(index, close + 2);
      index = close + 2;
      continue;
    }
    if (text[index] === "$") {
      const close = text.indexOf("$", index + 1);
      if (close < 0) {
        out += onText(text[index]);
        index += 1;
        continue;
      }
      out += onInline(text.slice(index + 1, close));
      index = close + 1;
      continue;
    }
    let next = text.indexOf("$", index);
    if (next < 0) next = text.length;
    out += onText(text.slice(index, next));
    index = next;
  }
  return out;
}

function inMathSpan(text: string, index: number): boolean {
  const before = text.slice(0, index);
  if ((before.split("$$").length - 1) % 2 === 1) return true;
  return (before.replace(/\$\$/g, "").split("$").length - 1) % 2 === 1;
}

function wrapBarePlain(text: string): string {
  let out = text;
  for (const pattern of BARE_INLINE_PATTERNS) {
    pattern.lastIndex = 0;
    out = out.replace(pattern, (...args: (string | number)[]) => {
      const all = String(args[0]);
      const input = String(args[args.length - 1]);
      const offset = Number(args[args.length - 2]);
      if (inMathSpan(input, offset)) return all;
      return `$${all}$`;
    });
  }
  return out;
}

/** 把 `P_{\mathrm{pos}}^{ij}`、`\frac{N_1}{N_0}` 这类裸公式包进 `$`。 */
export function wrapBareInlineMath(text: string): string {
  return mapDollarSpans(
    text,
    (body) => (HAN.test(body) ? wrapBarePlain(body) : `$${body}$`),
    wrapBarePlain,
  );
}

/** `$其中…$` 不是公式，拆出汉字，避免整句进 KaTeX。 */
export function unwrapHanMath(text: string): string {
  return mapDollarSpans(text, (body) => (HAN.test(body) ? body : `$${body}$`));
}

/** 译文展示前：保住 $$ 后的行内公式，并把裸 LaTeX 包上 `$`。 */
export function prepareZhMath(text: string): string {
  return wrapBareInlineMath(unwrapHanMath(text));
}

/** 公式 LaTeX 里误带「其中…」时撕开，公式只留数学。 */
export function peelHanFromLatex(text: string | null): { math: string | null; zh: string | null } {
  if (!text) return { math: null, zh: null };
  const trimmed = text.trim();
  if (!HAN.test(trimmed)) return { math: trimmed, zh: null };
  const match = trimmed.match(
    /^(?<math>[\s\S]*?)(?:\s+)(?<zh>(?:其中(?!的)|式中)[\s\S]+)$/,
  );
  if (match?.groups) {
    return {
      math: match.groups.math.replace(/\$/g, "").trim() || null,
      zh: prepareZhMath(match.groups.zh.trim()),
    };
  }
  const mixed = trimmed.match(
    /^(?<math>[^是表其][\s\S]*?)(?:\s+)(?<zh>[是表][\s\S]*[\u4e00-\u9fff][\s\S]*)$/,
  );
  if (mixed?.groups?.math && mixed.groups.zh) {
    return {
      math: mixed.groups.math.replace(/\$/g, "").trim() || null,
      zh: prepareZhMath(mixed.groups.zh.trim()),
    };
  }
  return { math: null, zh: prepareZhMath(trimmed) };
}

export function isMathDebrisProse(prose: string): boolean {
  const cleaned = stripZhMathDebris(prose).trim();
  if (!cleaned) return true;
  if (HAN.test(cleaned)) return false;
  return cleaned.length <= 24 && !/[A-Za-z]{8,}/.test(cleaned);
}

export type ZhSegment = { prose: string; display: string | null };

const WHERE_EN = /^\s*(where|in which|here,)\b/i;
const WHERE_ZH = /(?:其中(?!的)|式中)/;

export function isWhereProse(text: string): boolean {
  return WHERE_EN.test(text.trim());
}

/** 把误放在公式前/公式后的「其中/式中」撕下来。 */
export function peelTrailingWhereZh(zh: string): { lead: string; where: string | null } {
  const text = zh.trim();
  const patterns = [
    /^(?<lead>[\s\S]*?\\tag\{[^}]+\})\s*(?<where>(?:其中(?!的)|式中)[\s\S]+)$/,
    /^(?<lead>[\s\S]*?\\(?:mathcal|mathbb|frac|log|left|sum)[\s\S]*?)\s+(?<where>(?:其中(?!的)|式中)[\s\S]+)$/,
    /^(?<lead>[\s\S]+?)(?:\n\n+|(?<=[：:。])\s+)(?<where>(?:其中(?!的)|式中)[\s\S]+)$/,
    /^(?<lead>[\s\S]+?[：:])\s*(?<where>(?:其中(?!的)|式中)[\s\S]+)$/,
  ];
  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (!match?.groups) continue;
    const lead = match.groups.lead.trim();
    const where = match.groups.where.trim();
    if (where.length < 4) continue;
    if (!lead) return { lead: "", where };
    if (isLatexAssignmentProse(lead)) return { lead, where };
    if (lead.length < 8) continue;
    if (isWhereProse(lead) || WHERE_ZH.test(lead.slice(0, 8))) continue;
    return { lead, where };
  }
  return { lead: zh, where: null };
}

function nextWhereSegmentIndex<T extends ZhSegment>(enSegs: T[], from: number): number {
  for (let index = from + 1; index < enSegs.length; index += 1) {
    if (isWhereProse(enSegs[index].prose)) return index;
    if (enSegs[index].prose.trim() && !isMathDebrisProse(enSegs[index].prose)) return -1;
  }
  return -1;
}

function startsWithWhereZh(text: string): boolean {
  return /^(?:其中(?!的)|式中)/.test(text.trim());
}

/**
 * 英文是「导语 + 公式（可连续多条）+ where」，中文常把「其中…」写到公式前。
 * 跳过中间的纯公式段，把解释句接到英文 where 上。
 */
export function realignWhereSegments<T extends ZhSegment>(enSegs: T[], zhSegs: T[]): T[] {
  const zh = zhSegs.map((seg) => ({ ...seg }));
  for (let index = 0; index < enSegs.length; index += 1) {
    if (!enSegs[index]?.display) continue;
    const whereAt = nextWhereSegmentIndex(enSegs, index);
    if (whereAt < 0) continue;
    const current = zh[index];
    if (!current) continue;
    const fromLatex = peelHanFromLatex(current.display);
    if (fromLatex.zh) {
      current.display = fromLatex.math;
      current.prose = current.prose.trim()
        ? `${current.prose.trim()}\n\n${fromLatex.zh}`
        : fromLatex.zh;
    }
    const attachedWhere = startsWithWhereZh(current.prose) ? current.prose.trim() : null;
    const peeled = attachedWhere ? { lead: "", where: attachedWhere } : peelTrailingWhereZh(current.prose);
    if (!peeled.where) continue;
    const keepLead = peeled.lead.trim() && !isLatexAssignmentProse(peeled.lead) ? peeled.lead : "";
    zh[index] = { ...current, prose: keepLead };
    if (zh[whereAt]) {
      const rest = zh[whereAt].prose.trim();
      zh[whereAt] = {
        ...zh[whereAt],
        prose: rest && !startsWithWhereZh(rest) ? `${peeled.where}\n\n${rest}` : rest || peeled.where,
      };
    } else {
      while (zh.length < whereAt) zh.push({ prose: "", display: null } as T);
      zh[whereAt] = { prose: peeled.where, display: null } as T;
    }
  }
  return zh;
}

/** 正文块配对：where ↔ 其中 优先，再按顺序，最后从上一句撕「其中」。 */
export function pairBilingualProse(enTexts: string[], zhTexts: string[]): (string | undefined)[] {
  const assigned: (string | undefined)[] = enTexts.map(() => undefined);
  const used = new Set<number>();

  for (let index = 0; index < enTexts.length; index += 1) {
    if (!isWhereProse(enTexts[index])) continue;
    const found = zhTexts.findIndex((text, zhIndex) => !used.has(zhIndex) && startsWithWhereZh(text));
    if (found < 0) continue;
    used.add(found);
    assigned[index] = zhTexts[found];
  }

  let cursor = 0;
  for (let index = 0; index < enTexts.length; index += 1) {
    if (assigned[index] !== undefined) continue;
    while (cursor < zhTexts.length && used.has(cursor)) cursor += 1;
    if (cursor >= zhTexts.length) break;
    used.add(cursor);
    assigned[index] = zhTexts[cursor];
    cursor += 1;
  }

  for (let index = 0; index < enTexts.length; index += 1) {
    if (!isWhereProse(enTexts[index]) || assigned[index]) continue;
    for (let prev = index - 1; prev >= 0; prev -= 1) {
      const source = assigned[prev];
      if (!source) continue;
      const peeled = peelTrailingWhereZh(source);
      if (!peeled.where) continue;
      assigned[prev] = peeled.lead;
      assigned[index] = peeled.where;
      break;
    }
  }
  return assigned;
}

/** 丢掉译文开头没有汉字的公式碎片段，避免和英文公式错位。 */
export function dropLeadingZhDebris<T extends ZhSegment>(segments: T[]): T[] {
  let index = 0;
  while (
    index < segments.length &&
    !segments[index].display &&
    isMathDebrisProse(segments[index].prose)
  ) {
    index += 1;
  }
  if (index >= segments.length) return segments.slice(0, 0);
  return segments.slice(index).map((seg) => {
    const cleaned = stripZhMathDebris(seg.prose);
    return cleaned === seg.prose ? seg : { ...seg, prose: cleaned };
  });
}
