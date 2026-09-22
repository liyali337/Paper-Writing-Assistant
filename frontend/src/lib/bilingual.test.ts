import assert from "node:assert/strict";
import test from "node:test";

import {
  dropLeadingZhDebris,
  dropDuplicateLatexProse,
  isMathOnlyText,
  pairBilingualProse,
  peelHanFromLatex,
  peelTrailingWhereZh,
  prepareZhMath,
  realignWhereSegments,
  stripDisplayMath,
  stripZhMathDebris,
  unwrapHanMath,
} from "./bilingual.ts";

test("unwraps Chinese accidentally wrapped as inline math", () => {
  assert.equal(unwrapHanMath("$其中 Φ 表示文本嵌入。$"), "其中 Φ 表示文本嵌入。");
  assert.equal(unwrapHanMath("其中 $Q_T^T$ 表示文本。"), "其中 $Q_T^T$ 表示文本。");
});

test("does not strip inline math after a display formula's closing $$", () => {
  const raw = [
    "$$",
    "\\mathcal{L}_{i2j} = x, \\tag{5}",
    "$$ 其中 $P_{\\mathrm{pos}}^{ij}$ 表示正样本对，（$\\in \\{0,1\\}$）。该损失引入 $\\frac{N_1}{N_0}$。",
  ].join("\n");
  const out = prepareZhMath(raw);
  assert.match(out, /\$P_\{\\mathrm\{pos\}\}\^\{ij\}\$/);
  assert.match(out, /\$\\in \\\{0,1\\\}\$/);
  assert.match(out, /\$\\frac\{N_1\}\{N_0\}\$/);
  assert.equal((out.match(/\$P_\{\\mathrm\{pos\}\}/g) ?? []).length, 1);
});

test("does not rewrap mathrm tokens already inside an inline set", () => {
  const raw =
    "其中 $X_{T,i}$ 和 $X_{T',i}$（$i \\in \\{\\mathrm{rgb}, \\mathrm{nir}, \\mathrm{tir}\\}$）分别表示三种模态。";
  const out = prepareZhMath(raw);
  assert.match(out, /\$i \\in \\{\\mathrm\{rgb\}, \\mathrm\{nir\}, \\mathrm\{tir\}\\\}/);
  assert.doesNotMatch(out, /\$\\in\$/);
  assert.doesNotMatch(out, /\$\\mathrm\{rgb\}\$/);
});

test("wraps bare latex tokens in a Chinese where clause", () => {
  const raw =
    "其中 P_{\\mathrm{pos}}^{ij} 表示正样本对，样本对 (F_i, F_j) 的概率（\\in \\{0,1\\}）。间隔 \\frac{N_1}{N_0}。";
  const out = prepareZhMath(raw);
  assert.match(out, /\$P_\{\\mathrm\{pos\}\}\^\{ij\}\$/);
  assert.match(out, /\$\(F_i, F_j\)\$/);
  assert.match(out, /\$\\in \\\{0,1\\\}\$/);
  assert.match(out, /\$\\frac\{N_1\}\{N_0\}\$/);
});

test("strips duplicate display math from Chinese prose", () => {
  assert.match(
    stripDisplayMath("$$\nQ=W+b\n$$\n其中 $\\Phi$ 表示文本嵌入。"),
    /^其中/,
  );
});

test("peels 其中 after a raw latex formula with \\tag", () => {
  const raw =
    "\\mathcal{L}_{i2j} = -\\log \\left( \\mathbb{E} h \\right), \\tag{5} 其中 P ij pos 表示正样本对，P ij neg 表示负样本对。";
  const peeled = peelTrailingWhereZh(raw);
  assert.match(peeled.where ?? "", /^其中 P ij pos/);
  assert.match(peeled.lead, /\\tag\{5\}/);
  assert.equal(dropDuplicateLatexProse(raw), peeled.where);
});

test("moves 其中 out of an unparsed latex Chinese segment", () => {
  const aligned = realignWhereSegments(
    [
      { prose: "", display: "L=1" },
      { prose: "where P pos represents positive pairs.", display: null },
    ],
    [
      {
        prose:
          "\\mathcal{L}_{i2j} = -\\log h, \\tag{5} 其中 P ij pos 表示正样本对。",
        display: null,
      },
    ],
  );
  assert.equal((aligned[0]?.prose ?? "").trim(), "");
  assert.match(aligned[1]?.prose ?? "", /^其中/);
  assert.doesNotMatch(aligned[1]?.prose ?? "", /\\mathcal/);
});

test("peels 其中 out of a latex display body", () => {
  const peeled = peelHanFromLatex(
    "Q_T^T = \\Phi(X), \\tag{2} 其中 $\\Phi$ 表示由冻结的 CLIP 生成的嵌入。",
  );
  assert.match(peeled.math ?? "", /Q_T\^T/);
  assert.doesNotMatch(peeled.math ?? "", /其中/);
  assert.match(peeled.zh ?? "", /^其中/);
});

test("moves 其中 that starts the formula segment onto the English where slot", () => {
  const aligned = realignWhereSegments(
    [
      { prose: "", display: "Q=W+b" },
      { prose: "where $\\Phi$ denotes the text embedding.", display: null },
    ],
    [{ prose: "其中 $\\Phi$ 表示文本嵌入。", display: "Q=W+b" }],
  );
  assert.equal((aligned[0]?.prose ?? "").trim(), "");
  assert.match(aligned[1]?.prose ?? "", /^其中/);
});

test("formula-only blocks do not count as prose", () => {
  assert.equal(isMathOnlyText("$$\n\\mathcal{L}_{i2j}=x\n$$"), true);
  assert.equal(isMathOnlyText("$$ E = mc^2 $$"), true);
  assert.equal(isMathOnlyText("where $x$ is the query."), false);
  assert.equal(isMathOnlyText("|T)。在我们的对比学习设置中"), false);
});

test("strips a split formula tail from Chinese prose", () => {
  assert.equal(
    stripZhMathDebris("|T)。在我们的对比学习设置中，我们将正样本记为 N1。"),
    "在我们的对比学习设置中，我们将正样本记为 N1。",
  );
  assert.equal(stripZhMathDebris("$F_i$ 在我们的对比学习设置中"), "在我们的对比学习设置中");
});

test("does not strip a complete where-clause definition", () => {
  const raw = "$ZN \\in \\mathbb{R}^{N \\times C_v}$ 是视觉 patch 序列。";
  assert.match(stripZhMathDebris(raw), /\$ZN \\in \\mathbb\{R\}/);
  assert.match(stripZhMathDebris(raw), /是视觉/);
});

test("drops leading math-only Chinese segments", () => {
  const segs = dropLeadingZhDebris([
    { prose: "", display: "E=mc^2" },
    { prose: "|T)。在我们的对比学习设置中", display: "L=1" },
  ]);
  assert.equal(segs.length, 2);
  assert.equal(segs[0]?.display, "E=mc^2");
  assert.equal(segs[1]?.prose, "在我们的对比学习设置中");
  assert.equal(segs[1]?.display, "L=1");
});

test("drops leading debris that is not a formula slot", () => {
  const segs = dropLeadingZhDebris([
    { prose: "|T)。", display: null },
    { prose: "在我们的对比学习设置中", display: null },
  ]);
  assert.equal(segs.length, 1);
  assert.match(segs[0]?.prose ?? "", /^在我们/);
});

test("keeps empty Chinese slot that only holds a display formula", () => {
  const segs = dropLeadingZhDebris([
    { prose: "", display: "Q=W+b" },
    { prose: "其中 $\\Phi$ 表示文本嵌入。", display: null },
  ]);
  assert.equal(segs.length, 2);
  assert.equal(segs[0]?.display, "Q=W+b");
  assert.match(segs[1]?.prose ?? "", /^其中/);
});

test("moves 其中 clause to the English where segment after the formula", () => {
  const zh = peelTrailingWhereZh(
    "我们将其形式化表示为：\n其中 $\\Phi$ 表示由冻结的 CLIP 文本编码器生成的嵌入。",
  );
  assert.equal(zh.lead, "我们将其形式化表示为：");
  assert.match(zh.where ?? "", /^其中/);
  const aligned = realignWhereSegments(
    [
      { prose: "formally expressed as:", display: "Q=W+b" },
      { prose: "where $\\Phi$ denotes the text embedding.", display: null },
    ],
    [{ prose: "我们将其形式化表示为：其中 $\\Phi$ 表示文本嵌入。", display: "Q=W+b" }],
  );
  assert.equal(aligned[0]?.prose, "我们将其形式化表示为：");
  assert.match(aligned[1]?.prose ?? "", /^其中/);
});

test("pairs where-clause across two display formulas", () => {
  const aligned = realignWhereSegments(
    [
      { prose: "form the Transformer input:", display: "R=1" },
      { prose: "", display: "F=2" },
      { prose: "where $Z_N$ is the visual patch sequence.", display: null },
    ],
    [
      {
        prose: "构成 Transformer 层的输入：\n其中 $Z_N$ 是视觉 patch 序列。",
        display: "R=1",
      },
    ],
  );
  assert.match(aligned[0]?.prose ?? "", /输入：/);
  assert.doesNotMatch(aligned[0]?.prose ?? "", /其中/);
  assert.match(aligned[2]?.prose ?? "", /^其中/);
});

test("pairBilingualProse attaches 其中 to the English where paragraph", () => {
  const paired = pairBilingualProse(
    [
      "The register tokens form the input:",
      "where $Z_N$ is the visual patch sequence.",
    ],
    ["寄存器 token 构成输入：其中 $Z_N$ 是视觉 patch 序列。"],
  );
  assert.equal(paired[0], "寄存器 token 构成输入：");
  assert.match(paired[1] ?? "", /^其中/);
});
