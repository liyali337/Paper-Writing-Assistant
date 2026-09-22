import assert from "node:assert/strict";
import test from "node:test";

import { isAskHeading, isMarkerOnlyLatex, splitAskBold, splitAskParagraphs } from "./askProse.ts";

test("splits section headings after a Chinese period", () => {
  const parts = splitAskParagraphs("结论如下。 **一、与 SOTA 对比** 在车辆集上提升 3.0%。");
  assert.equal(parts.length, 3);
  assert.equal(parts[0].trim(), "结论如下。");
  assert.equal(parts[1].trim(), "**一、与 SOTA 对比**");
  assert.match(parts[2], /在车辆集上提升/);
});

test("parses bold markers", () => {
  const parts = splitAskBold("前 **一、消融** 后");
  assert.deepEqual(parts, [
    { type: "text", value: "前 " },
    { type: "bold", value: "一、消融" },
    { type: "text", value: " 后" },
  ]);
});

test("detects heading-only paragraphs", () => {
  assert.equal(isAskHeading("**二、消融实验（4.4 节）**"), true);
  assert.equal(isAskHeading("普通一句。"), false);
});

test("skips table ranking markers disguised as latex", () => {
  assert.equal(isMarkerOnlyLatex("\\circ"), true);
  assert.equal(isMarkerOnlyLatex("^{\\circ}"), true);
  assert.equal(isMarkerOnlyLatex("\\lambda"), false);
});
