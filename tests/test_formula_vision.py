from dl_agent.config import Settings
from dl_agent.knowledge.formula_vision import (
    repair_formula_items,
    resolve_formula_vision_model,
)
from dl_agent.knowledge.layout import LayoutItem
from tests.helpers import make_png


def test_resolve_vision_model_defaults_qwen_vl_on_dashscope() -> None:
    settings = Settings(
        openai_api_key="sk-test",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen3.8-max",
    )
    assert resolve_formula_vision_model(settings) == "qwen-vl-plus"


def test_resolve_vision_model_uses_explicit_and_gemini() -> None:
    explicit = Settings(
        openai_api_key="sk-test",
        openai_base_url="https://api.deepseek.com/v1",
        model_name="deepseek-chat",
        formula_vision_model="qwen-vl-max",
    )
    assert resolve_formula_vision_model(explicit) == "qwen-vl-max"
    gemini = Settings(
        openai_api_key="AQ.test",
        openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model_name="gemini-3.6-flash",
    )
    assert resolve_formula_vision_model(gemini) == "gemini-3.6-flash"
    deepseek = Settings(
        openai_api_key="sk-test",
        openai_base_url="https://api.deepseek.com/v1",
        model_name="deepseek-chat",
    )
    assert resolve_formula_vision_model(deepseek) == "deepseek-flash"


def test_repair_formula_items_transcribes_broken_latex() -> None:
    captured: dict = {}

    def fake_chat(messages, **_kwargs):
        captured["messages"] = messages
        return '{"latex": "Q^{R} \\\\in \\\\mathbb{R}^{B \\\\times C_v}"}'

    items = [
        LayoutItem(kind="heading", page=1, text="3. Method", level=2),
        LayoutItem(
            kind="text",
            page=1,
            text="We concatenate memory tokens with visual patches.",
        ),
        LayoutItem(
            kind="formula",
            page=1,
            text=r"\in \mathbb{R}^{B\times } ^{\times Cv}, ⊕Q^{V}",
            image_bytes=make_png(240, 48),
            width_px=240,
            height_px=48,
            bbox=(80, 200, 320, 230),
        ),
        LayoutItem(
            kind="formula",
            page=1,
            text=r"F_m=\{x_m^c,x_m^p\}",
            image_bytes=make_png(180, 40),
            width_px=180,
            height_px=40,
        ),
    ]
    settings = Settings(
        data_dir=".",
        openai_api_key="sk-test",
        formula_vision_model="qwen-vl-plus",
    )
    out = repair_formula_items(items, settings=settings, chat_fn=fake_chat)
    assert out[2].text == r"Q^{R} \in \mathbb{R}^{B \times C_v}"
    assert out[3].text == r"F_m=\{x_m^c,x_m^p\}"
    user = captured["messages"][1]["content"]
    assert user[0]["type"] == "image_url"
    assert str(user[0]["image_url"]["url"]).startswith("data:image/png;base64,")
    assert "Broken OCR latex" in user[1]["text"]
    assert "visual patches" in user[1]["text"]


def test_repair_formula_items_rejects_invalid_model_output() -> None:
    def fake_chat(messages, **_kwargs):
        return '{"latex": "L_{i} _{j}"}'

    items = [
        LayoutItem(
            kind="formula",
            page=1,
            text=r"L_{i} _{j}",
            image_bytes=make_png(120, 36),
            width_px=120,
            height_px=36,
        )
    ]
    settings = Settings(
        openai_api_key="sk-test",
        formula_vision_model="qwen-vl-plus",
    )
    out = repair_formula_items(items, settings=settings, chat_fn=fake_chat)
    assert out[0].text == r"L_{i} _{j}"


def test_repair_skips_when_disabled() -> None:
    called = {"n": 0}

    def fake_chat(messages, **_kwargs):
        called["n"] += 1
        return r'{"latex": "a=b"}'

    items = [
        LayoutItem(
            kind="formula",
            page=1,
            text=r"L_{i} _{j}",
            image_bytes=make_png(120, 36),
        )
    ]
    settings = Settings(formula_vision_enabled=False, openai_api_key="sk-test")
    repair_formula_items(items, settings=settings, chat_fn=fake_chat)
    assert called["n"] == 0
    assert items[0].text == r"L_{i} _{j}"


def test_repair_strips_leaked_latex_from_following_text() -> None:
    def fake_chat(messages, **_kwargs):
        return r'{"latex": "F_{\\mathrm{out}}^{(l)}=\\alpha(R_{\\mathrm{in}}^{(l)}) \\tag{4}"}'

    items = [
        LayoutItem(
            kind="formula",
            page=1,
            text=r"F_{out}=\alpha",
            image_bytes=make_png(240, 48),
            width_px=240,
            height_px=48,
        ),
        LayoutItem(
            kind="text",
            page=1,
            text=r"F_{out}^{{(l)}}=\alpha\in $\times$ $, \tag{4} where ZN\in \mathbb{R}^{N\times C_v}$ is the visual patch sequence.",
        ),
    ]
    settings = Settings(
        openai_api_key="sk-test",
        formula_vision_model="deepseek-flash",
    )
    out = repair_formula_items(items, settings=settings, chat_fn=fake_chat)
    assert r"\tag{4}" in out[0].text
    assert r"\tag{4}" not in out[1].text
    assert "visual patch sequence" in out[1].text
    assert r"\mathbb{R}" in out[1].text


def test_repair_rewrites_incomplete_juxtaposed_formula() -> None:
    def fake_chat(messages, **_kwargs):
        return (
            r'{"latex": "\\mathcal{L}_g(F)=\\mathcal{L}_{CE}(F)+\\mathcal{L}_{Tri}(F) \\tag{7}"}'
        )

    items = [
        LayoutItem(
            kind="formula",
            page=1,
            text=r"L_g L_{CE} L_{Tri}",
            image_bytes=make_png(240, 48),
            width_px=240,
            height_px=48,
            bbox=(80, 200, 360, 228),
        )
    ]
    settings = Settings(
        openai_api_key="sk-test",
        formula_vision_model="deepseek-flash",
    )
    out = repair_formula_items(items, settings=settings, chat_fn=fake_chat)
    assert r"\mathcal{L}_g(F)" in out[0].text
    assert r"\mathcal{L}_{CE}(F)" in out[0].text
    assert "+" in out[0].text


def test_repair_rejects_incomplete_vision_output() -> None:
    def fake_chat(messages, **_kwargs):
        return r'{"latex": "L_g L_{CE} L_{Tri}"}'

    items = [
        LayoutItem(
            kind="formula",
            page=1,
            text=r"L_g L_{CE} L_{Tri}",
            image_bytes=make_png(240, 48),
            width_px=240,
            height_px=48,
            bbox=(80, 200, 360, 228),
        )
    ]
    settings = Settings(
        openai_api_key="sk-test",
        formula_vision_model="deepseek-flash",
    )
    out = repair_formula_items(items, settings=settings, chat_fn=fake_chat)
    assert out[0].text == r"L_g L_{CE} L_{Tri}"


def test_repair_sends_complete_display_formula() -> None:
    called = {"n": 0}

    def fake_chat(messages, **_kwargs):
        called["n"] += 1
        return r'{"latex": "Q^{T}(W_{T_{u,i}} Q^{V})=Q^{T}"}'

    items = [
        LayoutItem(
            kind="formula",
            page=1,
            text=r"Q^{T} \cdot W X_{Tu,i} Q^{V} = Q^{T}",
            image_bytes=make_png(240, 48),
            width_px=240,
            height_px=48,
            bbox=(80, 200, 360, 228),
        )
    ]
    settings = Settings(
        openai_api_key="sk-test",
        formula_vision_model="deepseek-flash",
    )
    repair_formula_items(items, settings=settings, chat_fn=fake_chat)
    assert called["n"] == 1
    assert items[0].text == r"Q^{T}(W_{T_{u,i}} Q^{V})=Q^{T}"


def test_repair_skips_complete_inline_formula() -> None:
    called = {"n": 0}

    def fake_chat(messages, **_kwargs):
        called["n"] += 1
        return r'{"latex": "should-not-run"}'

    items = [
        LayoutItem(
            kind="formula",
            page=1,
            text=r"x_m^c",
            image_bytes=make_png(40, 18),
            width_px=40,
            height_px=18,
            bbox=(90, 240, 120, 252),
        )
    ]
    settings = Settings(
        openai_api_key="sk-test",
        formula_vision_model="deepseek-flash",
    )
    repair_formula_items(items, settings=settings, chat_fn=fake_chat)
    assert called["n"] == 0
    assert items[0].text == r"x_m^c"
