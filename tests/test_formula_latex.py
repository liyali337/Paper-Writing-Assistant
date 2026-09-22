from dl_agent.knowledge.formula_latex import (
    is_display_formula,
    is_renderable_latex,
    latex_looks_incomplete,
    latex_needs_vision,
    looks_like_latex,
    normalize_latex,
    spans_to_latex,
)
from dl_agent.knowledge.layout import LayoutItem, assemble, tidy_math_prose, tidy_translation_math
from tests.helpers import make_png


def test_spans_to_latex_builds_sub_and_super() -> None:
    spans = [
        {"text": "x", "size": 11, "bbox": (10, 20, 16, 32)},
        {"text": "c", "size": 7.5, "bbox": (16, 16, 21, 24)},
        {"text": "m", "size": 7.5, "bbox": (16, 28, 22, 36)},
    ]
    assert spans_to_latex(spans) == "x^{c}_{m}"


def test_normalize_strips_dollar_wrappers() -> None:
    assert normalize_latex("$$ F_m = x $$") == "F_m = x"
    assert looks_like_latex(r"F_{m}=\{x_m^c\}")


def test_display_latex_becomes_block_math() -> None:
    items = [
        LayoutItem(kind="heading", page=3, text="3.1. Feature Extraction", level=2),
        LayoutItem(kind="text", page=3, text="The encoder extracts features:"),
        LayoutItem(
            kind="formula",
            page=3,
            text=r"F_m=\{x_m^c,x_m^p\}",
            bbox=(60, 200, 280, 218),
        ),
        LayoutItem(kind="text", page=3, text="Here the CLS token is"),
        LayoutItem(
            kind="formula",
            page=3,
            text=r"x_m^c",
            bbox=(90, 240, 120, 252),
        ),
        LayoutItem(kind="text", page=3, text="in the sentence."),
    ]
    result = assemble("p1", items)
    text = result.sections[0].text
    assert "$$\nF_m=\\{x_m^c,x_m^p\\}\n$$" in text
    assert "$x_m^c$" in text
    assert result.figures == []


def test_tidy_does_not_break_latex_dollars() -> None:
    raw = "Here, $x^{c}_{m}$ denotes the CLS token, while x p m stays."
    out = tidy_math_prose(raw)
    assert "$x^{c}_{m}$" in out
    assert "$x_m^p$" in out


def test_tidy_folds_spaced_letters_with_operators() -> None:
    out = tidy_math_prose("The patch tokens have shape L p × D after pooling.")
    assert r"$L_{p} \times D$" in out or r"$L_{p}\times D$" in out
    assert "shape $" in out
    prose = tidy_math_prose("the CLS token remains readable")
    assert "$" not in prose
    raw = (
        "denotes the CLS token, which summarizes the global semantic knowledge "
        "of the image, while x_m^p = {x_m^1, x_m^2, ..., x_m^n} represents the "
        "patch tokens that capture local spatial details."
    )
    out = tidy_math_prose(raw)
    assert "$x_m^p =" in out
    assert r"\ldots" in out
    assert r"\{x_m^1" in out
    assert "while $" in out
    assert "$ represents" in out


def test_looks_like_latex_rejects_replacement_chars() -> None:
    assert not looks_like_latex("\ufffd_m^c")
    assert looks_like_latex(r"F_m=\{x_m^c\}")


def test_renderable_latex_rejects_broken_pdf_fragments() -> None:
    assert is_renderable_latex(r"F_m=\{x_m^c,x_m^p\}")
    assert is_renderable_latex(r"\mathcal{L} = \ell_{ce} + \ell_{tri}")
    assert is_renderable_latex(r"x_{i}^{j}")
    assert is_renderable_latex(r"\int f(x)\,dx")
    assert not is_renderable_latex(r"\in \mathbb{R}^{B\times } ^{\times Cv}, ⊕Q^{V}")
    assert not is_renderable_latex(r"L_{C} F_{i}, F_{j}, Q^{T} L_{i} _{j}. i,j\in")
    assert not is_renderable_latex(r"\mathbb{R}^{N\timesC}_{v}")
    assert not is_renderable_latex("")


def test_incomplete_juxtaposed_losses_need_vision() -> None:
    assert latex_looks_incomplete(r"L_g L_{CE} L_{Tri}")
    assert latex_needs_vision(r"L_g L_{CE} L_{Tri}")
    assert latex_looks_incomplete(r"L = L_g, F^{T'} F^T L_C F^V L_g F^T")
    assert not latex_looks_incomplete(r"F_m=\{x_m^c,x_m^p\}")
    assert not latex_looks_incomplete(r"\sum_{i,j} L_{C}(F_i, F_j)")
    assert not latex_needs_vision(r"\mathcal{L} = \ell_{ce} + \ell_{tri}")


def test_tidy_strips_garbled_math_and_unwraps_italic_words() -> None:
    raw = (
        "respectively: $$\ufffd$_{m}$ $x_{$x_{m}$p}$\n\n"
        "$$\nF_m = \\{x_m^c, x_m^p\\}, \\quad m \\in \\{R, N, T\\}. (1)\n$$\n\n"
        "Here, x c m denotes the CLS token. adaptively $f$use the global "
        "structure. Mul$t$imodal features. Proj$ec^{t}$ion."
    )
    out = tidy_math_prose(raw)
    assert "\ufffd" not in out
    assert "$f$use" not in out
    assert "fuse" in out
    assert "Multimodal" in out
    assert "Projection" in out
    assert "CLS token" in out
    assert "$$" in out
    assert "F_m" in out
    assert "x_m^c" in out


def test_tidy_keeps_prose_trapped_between_broken_dollars() -> None:
    trapped = "Here the CLS token is important. $\ufffd_{m}$ more prose."
    out = tidy_math_prose(f"lead-in.\n$$\n{trapped}\n$$")
    assert "CLS token" in out
    assert "\ufffd" not in out


def test_splice_does_not_break_english_words() -> None:
    from dl_agent.knowledge.formula_latex import _splice_latex

    assert _splice_latex("adaptively fuse the global", ["f"], "f") is None
    assert "$x_m^c$" in (_splice_latex("token x denotes", ["x"], "x_m^c") or "")


def test_formula_image_still_used_without_latex() -> None:
    items = [
        LayoutItem(kind="heading", page=1, text="1. Method", level=1),
        LayoutItem(
            kind="formula",
            page=1,
            image_bytes=make_png(240, 48),
            width_px=240,
            height_px=48,
            source="page_clip",
        ),
    ]
    result = assemble("p1", items)
    assert result.figures[0].kind == "formula"
    assert "<!--fig:fig-001-->" in result.sections[0].text


def test_display_vs_inline_by_bbox() -> None:
    wide = LayoutItem(kind="formula", page=1, bbox=(80, 100, 260, 118))
    tiny = LayoutItem(kind="formula", page=1, bbox=(90, 100, 118, 112))
    assert is_display_formula(wide)
    assert not is_display_formula(tiny)


def test_tidy_repairs_nested_dollars_and_unicode_hats() -> None:
    raw = (
        "The resulting x̂_m retains the most discriminative local details.\n\n"
        "α-fusion. The global feature $x_m^c$ and the local feature \\hat{x}_m "
        "are fused with a weighting factor:\n\n"
        "$$\n$$\n"
        r"\hat {f}_m = \alpha \cdot x_m^$c+(1$- \alpha) \cdot \hat {x}_m, (9)"
        "\n$$\n$$"
    )
    out = tidy_math_prose(raw)
    assert r"$\hat{x}_m$" in out
    assert "$x_m^c$" in out
    assert "x_m^$c" not in out
    assert "(1$-" not in out
    assert out.count("$$") == 2
    assert r"\hat{f}_m" in out
    assert r"\cdot" in out
    assert r"\tag{9}" in out
    twice = tidy_math_prose(out)
    assert r"\hat{f}_m" in twice
    assert twice.count("$$") == 2


def test_tidy_does_not_wrap_plus_fragments_inside_latex() -> None:
    raw = r"The fused map is \hat{f}_m = \alpha \cdot x_m^c+(1- \alpha) \cdot \hat{x}_m."
    out = tidy_math_prose(raw)
    assert "$c+(1$" not in out
    assert "$x_m^c$" in out


def test_tidy_wraps_inline_symbols_even_when_line_has_tex() -> None:
    raw = (
        "where n is the number of patch tokens, x_m^i denotes the original "
        "patch feature, x_f^i the low-pass filtered patch feature, and the "
        "absolute difference|x_m^i - x_f^i|represents the residual. "
        r"The terms \mu and \sigma denote the mean and standard deviation "
        r"of all residuals, respectively. We then select the top "
        r"K = ⌊ $n\times r$⌋ (r denotes the selection ratio of tokens) patches."
    )
    out = tidy_math_prose(raw)
    assert "$x_m^i$" in out
    assert "$x_f^i$" in out
    assert r"$|x_m^i - x_f^i|$" in out
    assert r"$\mu$" in out
    assert r"$\sigma$" in out
    assert r"\lfloor n\times r \rfloor" in out
    assert "⌊ $" not in out


def test_tidy_repairs_linear_weight_matrix_inline() -> None:
    raw = (
        "where Linear S m and Linear P m denote the shared and private "
        "linear transformation layers for modality m, respectively, and "
        r"their corresponding weight matrices are $WS_{m},WP_{m}\in RD\times D.$"
    )
    out = tidy_math_prose(raw)
    assert "Linear S m" not in out
    assert r"\mathrm{Linear}_m^{S}" in out
    assert r"\mathrm{Linear}_m^{P}" in out
    assert r"W^{S}_{m}" in out
    assert r"W^{P}_{m}" in out
    assert r"\mathbb{R}^{D \times D}" in out or r"\mathbb{R}^{D\times D}" in out
    assert r"RD\times D" not in out


def test_tidy_repairs_hyphen_word_and_leaked_subscript() -> None:
    raw = (
        r"To obtain a unified multi$-mo^{d}$ al representation, we concatenate. "
        r"A set of shared latent tokens Z inv $\in R^{B\times L}s^{\times D}$ "
        r"is initialized. The number of shared latent variables L s is 24. "
        r"Similarly F cls $\in R^{n\times P}q^{\times D}$ and K p tokens."
    )
    out = tidy_math_prose(raw)
    assert "multi-modal" in out
    assert "$mo^{d}$" not in out
    assert r"Z_{\mathrm{inv}}" in out
    assert r"F_{\mathrm{cls}}" in out
    assert r"$L_s$" in out
    assert r"$K_p$" in out
    assert r"\mathbb{R}^{B\times L_s \times D}" in out
    assert r"P_q" in out
    assert r"L}s^{" not in out
    assert "Z inv" not in out


def test_tidy_merges_orphan_accents_into_math() -> None:
    raw = (
        r"based on the patch tokens ¨ $x_m^p$, the shared tokens ˜ $y_i$, "
        "the spaced form ¨ x m, and the combining form x\u0308_m. "
        r"Bernhard Sch¨ olkopf, and Olivier Bachem."
    )
    out = tidy_math_prose(raw)
    assert r"\ddot{x}_m^p" in out
    assert "¨" not in out
    assert r"\tilde{y}_i" in out
    assert r"\ddot{x}_m" in out
    assert "Schölkopf" in out
    assert r"\ddot{o}" not in out


def test_sanitize_closes_escaped_script_braces() -> None:
    raw = (
        "$$\n"
        r"\mathbf{x}_{m}^{(l+1)\}=\text{SPBF}(\mathbf{x}_{m}^{(l)}) \tag{21}"
        "\n$$"
    )
    out = tidy_math_prose(raw)
    assert r"^{(l+1)}" in out
    assert r"^{(l+1)\}" not in out


def test_tidy_peels_english_off_display_and_collapses_shattered_ops() -> None:
    raw = (
        "The register tokens form the Transformer input:\n\n"
        r"$$\n"
        r"R_{\mathrm{in}}^{(l)}=[Z_N^{(l)}\oplus r^{(l)}\oplus Q_T^{V(l)}]"
        r"\in\mathbb{R}^{B\times(1+n+N)\times C_v}"
        "\n$$\n\n"
        r"F_{out}^{{(l)}}=\alpha (R_{in}^{{(l)}};\theta_{blk}^{{(l)}})\in"
        "\n$\\times$\n"
        "$(1+n+N)$\n"
        "$\\times$\n"
        r"$, \tag{4} where ZN\in \mathbb{R}^{N\timesC}_{v}$ is the visual "
        "patch sequence (N denotes the number of image patches), "
        r"$l(1$\n$\leq$\n$l$\n$\leq$\n$L)$ indexes the Transformer layers."
    )
    out = tidy_math_prose(raw)
    assert r"\tag{4}" in out
    assert "$$" in out
    assert r"N\timesC" not in out
    assert "where" in out
    assert "visual patch sequence" in out
    assert out.count(r"$\times$") <= 1
    assert "\n$\\leq$\n" not in out
    assert r"^{{(l)}}" not in out


def test_tidy_wraps_loss_names_as_mathcal() -> None:
    out = tidy_math_prose(
        "Here, LCE and L Tri denote label smoothing losses, and L C denotes the CT-CMC loss."
    )
    assert r"$\mathcal{L}_{CE}$" in out
    assert r"$\mathcal{L}_{Tri}$" in out
    assert r"\mathcal{L}_{C}$" in out
    assert "LCE" not in out
    assert "L Tri" not in out


def test_tidy_wraps_leaked_inline_set_and_leq() -> None:
    raw = (
        r"where ZN\in \mathbb{R}^{N \times C}_{v} is the visual patch sequence "
        r"(N denotes the number of image patches), "
        r"(1\leq l\leq L) indexes the Transformer layers."
    )
    out = tidy_math_prose(raw)
    assert r"ZN\in \mathbb{R}" not in out
    assert r"Z_{N}" in out
    assert r"\mathbb{R}^{N \times C}_{v}" not in out
    assert r"C_v" in out
    assert r"$(1\leq l\leq L)$" in out
    assert "$$" not in out
    assert "visual patch sequence" in out
    assert "indexes the Transformer layers" in out


def test_tidy_merges_split_display_assignment() -> None:
    raw = (
        "The register tokens form the Transformer input:\n\n"
        r"$$\n"
        r"R_{\mathrm{in}}^{(l)}=[Z_N^{(l)}\oplus r^{(l)}\oplus Q_T^{V(l)}]"
        r"\in\mathbb{R}^{B\times(1+n+N)\times C_v}"
        "\n$$\n\n"
        r"F_{out}^{{(l)}}=\alpha (R_{in}^{{(l)}}; \theta_{blk}^{{(l)}})"
        "\n\n"
        r"$$\n"
        r"\in \mathbb{R}^{B\times(1+n+N)\times C_v},"
        "\n$$"
    )
    out = tidy_math_prose(raw)
    assert r"F_{out}^{{(l)}}" not in out
    assert r"$\alpha$" not in out
    assert r"$\theta$" not in out
    assert r"F_{out}^{(l)}=\alpha" in out or r"F_{\mathrm{out}}^{(l)}" in out
    assert r"\theta_{blk}^{(l)}" in out
    assert out.count(r"\in\mathbb{R}^{B\times(1+n+N)\times C_v}") + out.count(
        r"\in \mathbb{R}^{B\times(1+n+N)\times C_v}"
    ) >= 2
    assert "$$" in out
    body = out[out.find(r"F_{out}") :]
    assert r"\in" in body[:200]


def test_tidy_merges_double_brace_assignment_split_by_inline_dollars() -> None:
    raw = (
        "Transformer layers:\n\n"
        "$$\n"
        r"R_{\mathrm{in}}^{(l)}=[Z_N^{(l)}\oplus r^{(l)}\oplus Q_T^{V(l)}]"
        r"\in\mathbb{R}^{B\times(1+n+N)\times C_v} \tag{3}"
        "\n$$\n"
        r"F_{out}^{{(l)}}=$\alpha$ (R_{in}^{{(l)}}; $\theta$_{blk}^{{(l)}})"
        "\n$$\n"
        r"\in \mathbb{R}^{B\times(1+n+N)\times C_v}, \tag{4}"
        "\n$$"
    )
    out = tidy_math_prose(raw)
    assert r"F_{out}^{{(l)}}" not in out
    assert r"$\alpha$" not in out
    assert r"$\theta$" not in out
    assert r"F_{out}^{(l)}" in out
    assert r"\alpha" in out
    assert r"\theta_{blk}^{(l)}" in out
    assert r"\tag{4}" in out
    start = out.find(r"F_{out}^{(l)}")
    end = out.find("$$", start)
    chunk = out[start:end] if end > start else out[start : start + 240]
    assert r"\in" in chunk


def test_tidy_repairs_real_space_digits_and_log_frac() -> None:
    dim = tidy_math_prose(
        r"and $W_{proj} \in R512 \times C_v$ is the projection weight matrix."
    )
    assert r"\mathbb{R}^{512 \times C_v}" in dim or r"\mathbb{R}^{512\times C_v}" in dim
    assert "R512" not in dim
    prose = tidy_math_prose("and W_proj ∈ R512 × C_v is the projection weight matrix.")
    assert r"\mathbb{R}^{512" in prose
    assert "R512" not in prose
    assert r"C_v" in prose
    assert r"C}$_v" not in prose
    log = tidy_math_prose(r"where n: $\Delta = \log (N_N^1 0)$.")
    assert r"\frac{N_1}{N_0}" in log
    assert r"N_N^1" not in log
    log_prose = tidy_math_prose("n: Δ = log (N_N^1 0).")
    assert r"\frac{N_1}{N_0}" in log_prose
    assert "$$" not in log_prose


def test_tidy_repairs_glued_tensor_and_stray_real_sub() -> None:
    inline = tidy_math_prose(
        r"and $QVT \in \mathbb{R}_v^{B\times C}$ is the query."
    )
    assert "QVT" not in inline
    assert r"Q_{T}^{V}" in inline
    assert r"C_v" in inline
    assert r"\mathbb{R}_v" not in inline
    compact = inline.replace(" ", "")
    assert r"\mathbb{R}^{B\times C_v}".replace(" ", "") in compact
    unicode_form = tidy_math_prose("and QVT ∈ ℝ_v^{B×C} is the query.")
    assert "QVT" not in unicode_form
    assert r"Q_{T}^{V}" in unicode_form
    assert r"C_v" in unicode_form
    already = tidy_math_prose(
        r"and $Q_{T}^{V} \in \mathbb{R}^{B\times C_v}$ is the query."
    )
    assert r"Q_{T}^{V}" in already
    assert r"C_v" in already
    cnn = tidy_math_prose(r"the map $CNN \in \mathbb{R}^{d\times d}$")
    assert "CNN" in cnn
    qkv = tidy_math_prose(r"attention $QKV \in \mathbb{R}^{B\times d}$")
    assert "QKV" in qkv
    pos = tidy_math_prose(r"$x \in \mathbb{R}_+^{n}$")
    assert r"\mathbb{R}_+" in pos


def test_translation_tidy_keeps_conditional_prob_in_same_paragraph() -> None:
    raw = (
        "为阐明该方法：负样本对遵循边缘分布的乘积 p neg（$F_i$|$T$）\\cdot p$ neg（$F_j$|T）。"
        "在我们的对比学习设置中，我们将正样本对的数量记为 N 1。\n\n"
        "带上下文的保护性间隔目标。Li 等人提供了下界 I ($F_i$, $F_{j}|T$)。"
    )
    out = tidy_translation_math(raw)
    paras = [part.strip() for part in out.split("\n\n") if part.strip()]
    assert paras[0].startswith("为阐明")
    assert "在我们的对比学习设置中" in paras[0]
    assert not any(part.startswith("|T") for part in paras)
    assert "$$" not in out
    assert "p$ neg" not in out
    assert "$F_j" in out


def test_translation_tidy_peels_where_zh_off_display_math() -> None:
    raw = (
        "形式化表示为：\n"
        "$$\n"
        r"Q_T^T = \Phi(X), \tag{2} 其中 \Phi 表示文本嵌入。"
        "\n$$"
    )
    out = tidy_translation_math(raw)
    assert "其中" not in (out.split("$$")[1] if "$$" in out else out)
    assert "其中" in out
    assert out.count("$$") >= 2


def test_translation_tidy_wraps_bare_latex_in_where_zh() -> None:
    raw = (
        "$$\n"
        r"\mathcal{L}_{i2j} = x, \tag{5}"
        "\n"
        r"$$ 其中 $P_{\mathrm{pos}}^{ij}$ 表示正样本对，（$\in \{0,1\}$）。"
        r"该损失引入了保护性间隔 $\frac{N_1}{N_0}$。"
    )
    out = tidy_translation_math(raw)
    assert r"$P_{\mathrm{pos}}^{ij}$" in out
    assert r"$\frac{N_1}{N_0}$" in out
    display_body = out.split("$$")[1]
    assert "其中" not in display_body

    bare = r"其中 P_{\mathrm{pos}}^{ij} 表示正样本对，间隔 \frac{N_1}{N_0}。"
    wrapped = tidy_translation_math(bare)
    assert r"$P_{\mathrm{pos}}^{ij}$" in wrapped
    assert r"$\frac{N_1}{N_0}$" in wrapped


def test_translation_tidy_keeps_in_set_as_one_inline() -> None:
    raw = (
        r"其中 $X_{T,i}$ 和 $X_{T',i}$（$i \in \{\mathrm{rgb}, \mathrm{nir}, \mathrm{tir}\}$）"
        "分别表示三种模态的 CoT 推理文本。"
    )
    out = tidy_translation_math(raw)
    assert r"$i \in \{\mathrm{rgb}, \mathrm{nir}, \mathrm{tir}\}$" in out
    assert r"$\in$" not in out
    assert r"$\mathrm{rgb}$" not in out


def test_tidy_repairs_leading_dictionary_subscripts() -> None:
    raw = (
        "$$\n"
        r"_{r}h_{r} _{rn}h_{rn} _{rt}h_{rt} _{s}h_{rnt}, "
        r"_{n}h_{n} _{rn}h_{rn} _{nt}h_{nt} _{s}h_{rnt}, "
        r"_{t}h_{t} _{rt}h_{rt} _{nt}h_{nt} _{s}h_{rnt},"
        "\n$$"
    )
    out = tidy_math_prose(raw)
    assert r"_{r}h_{r}" not in out
    assert r"RR_{r}" not in out
    assert r"R_{r} h_{r}" in out
    assert r"R_{rn} h_{rn}" in out
    assert r"R_{s} h_{rnt}" in out
    assert r"N_{n} h_{n}" in out
    assert r"T_{t} h_{t}" in out
    assert " + " in out
    assert tidy_math_prose(out).count(r"R_{r}") == out.count(r"R_{r}")


def test_tidy_repairs_orphan_weight_superscripts() -> None:
    raw = "$$\n" r"q_{v} =z^{v} ^{Qv}, q_{a} =z^{a} ^{Qa}, ^{K}, ^{V}, q_{v} q_{a}" "\n$$"
    out = tidy_math_prose(raw)
    assert r"z^{v} W^{Qv}" in out
    assert r"z^{a} W^{Qa}" in out
    assert r"W^{K}" in out
    assert r"W^{V}" in out
    assert r"^{Qv}" not in out.replace(r"W^{Qv}", "")
    assert r"q_{v} = z^{v}" in out
    assert r"qV_{v}" not in out
    assert r"q_{v} q_{a}" in out
    assert tidy_math_prose(out) == out


def test_translation_tidy_repairs_empty_group_subscripts() -> None:
    raw = (
        "$$\n"
        r"{}_{r}h_{r}\ {}_{rn}h_{rn}\ {}_{rt}h_{rt}\ {}_{s}h_{rnt},"
        r"\ {}_{n}h_{n}\ {}_{rn}h_{rn}\ {}_{nt}h_{nt}\ {}_{s}h_{rnt},"
        r"\ {}_{t}h_{t}\ {}_{rt}h_{rt}\ {}_{nt}h_{nt}\ {}_{s}h_{rnt},"
        "\n$$"
    )
    out = tidy_translation_math(raw)
    assert r"{}_{r}h_{r}" not in out
    assert r"RR_{r}" not in out
    assert r"R_{r} h_{r}" in out
    assert r"N_{n} h_{n}" in out
    assert r"T_{t} h_{t}" in out
    assert tidy_translation_math(out).count(r"R_{r}") == out.count(r"R_{r}")
