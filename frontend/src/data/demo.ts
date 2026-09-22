import type {
  Figure,
  FigureTranslation,
  MethodExplain,
  Paper,
  PaperAnswer,
  PaperIntro,
  Section,
  SectionTranslation,
} from "../api/types";

const PAPER_ID = "demo";

export const demoPaper: Paper = {
  paper_id: PAPER_ID,
  sha256: "demo",
  filename: "HieraAlign-preview.pdf",
  status: "ready",
  parser: "docling",
  page_count: 8,
  figure_count: 2,
  title: "HieraAlign: Hierarchical Cross-Modal Alignment for Vision-Language Models",
  authors: ["Ada Chen", "Ben Li", "Cara Wu"],
  abstract:
    "We present HieraAlign, a hierarchical alignment framework that matches image patches, regions, and global tokens with noun phrases, sentences, and captions.",
  language: "en",
  intro_status: "ready",
  method_status: "ready",
  translate_status: "ready",
  index_status: "ready",
};

export const demoSections: Section[] = [
  {
    section_id: "sec-abstract",
    paper_id: PAPER_ID,
    title: "Abstract",
    kind: "abstract",
    level: 1,
    page_start: 1,
    page_end: 1,
    text: "Vision-language models often align only global image and text embeddings. We propose HieraAlign to align tokens across three granularities and show gains on retrieval and grounding.",
    parent_id: null,
    figure_ids: [],
  },
  {
    section_id: "sec-intro",
    paper_id: PAPER_ID,
    title: "1 Introduction",
    kind: "intro",
    level: 1,
    page_start: 1,
    page_end: 2,
    text: "Cross-modal retrieval and visual grounding benefit from fine-grained correspondence. Existing dual-encoder models typically project a single image CLS and a caption embedding into a shared space, discarding region and phrase structure. HieraAlign keeps three levels of tokens on both modalities and aligns them with a shared projection.",
    parent_id: null,
    figure_ids: ["fig-overview"],
  },
  {
    section_id: "sec-related",
    paper_id: PAPER_ID,
    title: "2 Related Work",
    kind: "related",
    level: 1,
    page_start: 2,
    page_end: 3,
    text: "Prior work includes CLIP-style contrastive learning, region-text matching, and hierarchical transformers for vision. Our approach combines multi-scale visual tokens with linguistically motivated text spans.",
    parent_id: null,
    figure_ids: [],
  },
  {
    section_id: "sec-method",
    paper_id: PAPER_ID,
    title: "3 Method",
    kind: "method",
    level: 1,
    page_start: 3,
    page_end: 5,
    text: "We construct patch, region, and image tokens from a hierarchical vision backbone, and noun-phrase, sentence, and caption tokens from a text encoder. A shared projection maps all levels into one embedding space, followed by level-aware contrastive losses.",
    parent_id: null,
    figure_ids: ["fig-pipeline"],
  },
  {
    section_id: "sec-experiment",
    paper_id: PAPER_ID,
    title: "4 Experiments",
    kind: "experiment",
    level: 1,
    page_start: 5,
    page_end: 7,
    text: "We evaluate on Flickr30K, COCO, and RefCOCO. HieraAlign improves Recall@1 and grounding IoU over a dual-encoder baseline with similar compute.",
    parent_id: null,
    figure_ids: [],
  },
  {
    section_id: "sec-conclusion",
    paper_id: PAPER_ID,
    title: "5 Conclusion",
    kind: "conclusion",
    level: 1,
    page_start: 7,
    page_end: 8,
    text: "Hierarchical alignment is a simple drop-in improvement for vision-language dual encoders. Future work includes denser supervision and video extensions.",
    parent_id: null,
    figure_ids: [],
  },
];

export const demoFigures: Figure[] = [
  {
    figure_id: "fig-overview",
    paper_id: PAPER_ID,
    section_id: "sec-intro",
    page: 2,
    kind: "figure",
    label: "Figure 1",
    caption: "Overview of HieraAlign: three visual levels align with three textual levels.",
    storage_key: "demo",
    source: "demo",
    width_px: 960,
    height_px: 420,
  },
  {
    figure_id: "fig-pipeline",
    paper_id: PAPER_ID,
    section_id: "sec-method",
    page: 4,
    kind: "figure",
    label: "Figure 2",
    caption: "Token construction and shared projection.",
    storage_key: "demo",
    source: "demo",
    width_px: 960,
    height_px: 360,
  },
];

export const demoIntro: PaperIntro = {
  paper_id: PAPER_ID,
  title_zh: "HieraAlign：面向视觉语言模型的层次化跨模态对齐",
  one_sentence_zh: "用补丁 / 区域 / 整图与短语 / 句子 / 标题三层对齐，提升检索与定位。",
  problem_zh: "主流双塔模型只对齐全局图像与文本向量，丢掉了区域与短语级对应关系。",
  motivation_zh: "检索与视觉定位都需要更细粒度的对应；层次化 token 能在不改整体架构的前提下补上这一环。",
  contributions: [
    "提出三层视觉 token 与三层文本 token 的对齐框架",
    "用共享投影与层级对比损失统一训练",
    "在检索与 RefCOCO 定位上相对基线稳定提升",
  ],
  task_setting_zh: "图像–文本检索与指代表达定位；训练数据以成对图文为主。",
  reading_order_zh: ["1 Introduction", "3 Method", "4 Experiments"],
  overview_figure_id: "fig-overview",
  evidence: [
    {
      page: 1,
      section_title: "1 Introduction",
      quote:
        "Existing dual-encoder models typically project a single image CLS and a caption embedding into a shared space, discarding region and phrase structure.",
      sourced: true,
    },
  ],
  partial: false,
  section_match: "demo",
  model: "demo",
  prompt_version: "demo",
};

export const demoMethod: MethodExplain = {
  paper_id: PAPER_ID,
  overview_zh:
    "从层次视觉骨干提取补丁、区域、整图 token，从文本编码器提取短语、句子、标题 token，经共享投影后用层级对比损失对齐。",
  pipeline_steps: [
    {
      name: "视觉多尺度编码",
      what_zh: "层次骨干输出 Patch / Region / CLS 三类 token。",
      page: 3,
      section_title: "3 Method",
      quote: "We construct patch, region, and image tokens from a hierarchical vision backbone",
      sourced: true,
      figure_ids: ["fig-pipeline"],
    },
    {
      name: "文本跨度编码",
      what_zh: "对名词短语、句子与完整标题分别编码。",
      page: 3,
      section_title: "3 Method",
      quote: "noun-phrase, sentence, and caption tokens from a text encoder",
      sourced: true,
      figure_ids: [],
    },
    {
      name: "共享投影与对比",
      what_zh: "统一映射到同一空间，按层级做对比学习。",
      page: 4,
      section_title: "3 Method",
      quote: "A shared projection maps all levels into one embedding space",
      sourced: true,
      figure_ids: ["fig-pipeline"],
    },
  ],
  key_modules: [
    {
      name: "Shared projection",
      what_zh: "所有层级共用一个线性投影头，减少参数并促进跨层对齐。",
      page: 4,
      section_title: "3 Method",
      quote: null,
      sourced: false,
      figure_ids: ["fig-pipeline"],
    },
  ],
  vs_prior_zh: "相对只做全局对齐的 CLIP 式双塔，HieraAlign 显式保留区域与短语对应。",
  assumptions_zh: ["训练图文对可提供弱监督", "层次骨干已能产出可用的区域特征"],
  open_to_read_zh: ["损失权重如何在层级间分配", "区域提案是否依赖额外检测器"],
  figure_ids: ["fig-overview", "fig-pipeline"],
  partial: false,
  section_match: "demo",
  model: "demo",
  prompt_version: "demo",
};

export const demoTitleZh = "HieraAlign：面向视觉语言模型的层次化跨模态对齐";

export const demoAskSuggestions = [
  "这篇论文主要解决什么问题？",
  "方法流水线分哪几步？",
  "我之前研读过类似方向的论文吗？",
  "在 arXiv 上检索相关论文",
];

export const demoAskAnswers: Record<string, PaperAnswer> = {
  "这篇论文主要解决什么问题？": {
    paper_id: PAPER_ID,
    question: "这篇论文主要解决什么问题？",
    answer_zh:
      "主流双塔视觉语言模型往往只对齐全局图像与文本向量，丢掉了区域与短语级对应。HieraAlign 用补丁 / 区域 / 整图与短语 / 句子 / 标题三层对齐，以提升检索与定位。",
    citations: [
      {
        page: 1,
        section_title: "1 Introduction",
        section_id: "sec-intro",
        quote:
          "Existing dual-encoder models typically project a single image CLS and a caption embedding into a shared space, discarding region and phrase structure.",
        sourced: true,
      },
    ],
    figure_ids: ["fig-overview"],
    no_evidence: false,
    partial: false,
    model: "demo",
    prompt_version: "ask-demo",
    embedding_version: "demo",
  },
  "方法流水线分哪几步？": {
    paper_id: PAPER_ID,
    question: "方法流水线分哪几步？",
    answer_zh:
      "大致三步：先从层次视觉骨干得到 Patch / Region / CLS；再从文本侧编码短语、句子与标题；最后经共享投影做层级对比学习，损失为 $$\\mathcal{L} = \\lambda \\mathcal{L}_{\\mathrm{con}} + \\mathcal{L}_{\\mathrm{ce}}$$。",
    citations: [
      {
        page: 3,
        section_title: "3 Method",
        section_id: "sec-method",
        quote: "We construct patch, region, and image tokens from a hierarchical vision backbone",
        sourced: true,
      },
      {
        page: 4,
        section_title: "3 Method",
        section_id: "sec-method",
        quote: "A shared projection maps all levels into one embedding space",
        sourced: true,
      },
    ],
    figure_ids: ["fig-pipeline"],
    no_evidence: false,
    partial: false,
    model: "demo",
    prompt_version: "ask-demo",
    embedding_version: "demo",
  },
  "实验在哪些数据集上评估？": {
    paper_id: PAPER_ID,
    question: "实验在哪些数据集上评估？",
    answer_zh: "论文在 Flickr30K、COCO 与 RefCOCO 上评估，相对双塔基线提升了 Recall@1 与定位 IoU。",
    citations: [
      {
        page: 5,
        section_title: "4 Experiments",
        section_id: "sec-experiment",
        quote: "We evaluate on Flickr30K, COCO, and RefCOCO.",
        sourced: true,
      },
    ],
    figure_ids: [],
    no_evidence: false,
    partial: false,
    model: "demo",
    prompt_version: "ask-demo",
    embedding_version: "demo",
  },
  "我之前研读过类似方向的论文吗？": {
    paper_id: PAPER_ID,
    question: "我之前研读过类似方向的论文吗？",
    answer_zh: "本地库里有一篇层次视觉 Transformer 工作，和当前这篇的跨模态对齐相关。细节需打开该篇精读。",
    citations: [],
    figure_ids: [],
    library_hits: [
      {
        paper_id: "demo-hiera",
        title: "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles",
        filename: "hiera.pdf",
        authors: ["Chaitanya Ryali", "Yuan-Ting Hu", "Daniel Bolya"],
        abstract: "We introduce Hiera, a hierarchical vision transformer that is fast, powerful, and simple.",
        why: "摘要命中 hierarchical alignment",
        openable: true,
        index_status: "ready",
      },
    ],
    mode: "library",
    no_evidence: false,
    partial: false,
    model: "demo",
    prompt_version: "ask-demo",
    embedding_version: "demo",
  },
  "在 arXiv 上检索相关论文": {
    paper_id: PAPER_ID,
    question: "在 arXiv 上检索相关论文",
    answer_zh:
      "arXiv 上能检索到标题相近的层次视觉 Transformer 工作。这些是外部文献元数据，不是当前这篇 PDF 的原文证据。",
    citations: [],
    figure_ids: [],
    external_refs: [
      {
        source: "arxiv",
        title: "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles",
        url: "https://arxiv.org/abs/2306.00989",
        identifier: "arxiv:2306.00989",
        snippet: "We introduce Hiera, a hierarchical vision transformer that is fast, powerful, and simple.",
        year: 2023,
        authors: ["Chaitanya Ryali", "Yuan-Ting Hu", "Daniel Bolya"],
      },
    ],
    mode: "arxiv",
    no_evidence: false,
    partial: false,
    model: "demo",
    prompt_version: "ask-demo",
    embedding_version: "demo",
  },
};

export function demoAnswerFor(question: string): PaperAnswer {
  const hit = demoAskAnswers[question.trim()];
  if (hit) return hit;
  return {
    paper_id: PAPER_ID,
    question,
    answer_zh:
      "这是界面预览里的示意回答。正式接入检索问答后，会按当前论文证据作答，并附可跳页的引用。",
    citations: [
      {
        page: 3,
        section_title: "3 Method",
        section_id: "sec-method",
        quote: "A shared projection maps all levels into one embedding space",
        sourced: true,
      },
    ],
    figure_ids: [],
    no_evidence: false,
    partial: true,
    model: "demo",
    prompt_version: "ask-demo",
    embedding_version: "demo",
  };
}

export const demoFigureTranslations: FigureTranslation[] = [
  {
    figure_id: "fig-overview",
    caption_zh: "图 1. HieraAlign 总览：三个视觉层级与三个文本层级对齐。",
  },
  {
    figure_id: "fig-pipeline",
    caption_zh: "图 2. Token 构造与共享投影。",
  },
];

export const demoTranslations: SectionTranslation[] = [
  {
    section_id: "sec-abstract",
    title_zh: "摘要",
    text_zh:
      "视觉语言模型常常只对齐全局图像与文本嵌入。我们提出 HieraAlign，在三个粒度上对齐 token，并在检索与定位任务上取得提升。",
  },
  {
    section_id: "sec-intro",
    title_zh: "1 引言",
    text_zh:
      "跨模态检索与视觉定位受益于细粒度对应。现有双塔模型通常只把单个图像 CLS 与标题嵌入投到共享空间，丢掉了区域与短语结构。HieraAlign 在两侧保留三层 token，并用共享投影对齐。",
  },
  {
    section_id: "sec-related",
    title_zh: "2 相关工作",
    text_zh:
      "已有工作包括 CLIP 式对比学习、区域–文本匹配，以及面向视觉的层次 Transformer。我们的方法把多尺度视觉 token 与语言学启发的文本跨度结合起来。",
  },
  {
    section_id: "sec-method",
    title_zh: "3 方法",
    text_zh:
      "我们从层次视觉骨干构造补丁、区域与图像 token，从文本编码器构造名词短语、句子与标题 token。共享投影把各层映射到同一嵌入空间，再配合层级感知的对比损失。",
  },
  {
    section_id: "sec-experiment",
    title_zh: "4 实验",
    text_zh:
      "我们在 Flickr30K、COCO 与 RefCOCO 上评估。在相近算力下，HieraAlign 相对双塔基线提升了 Recall@1 与定位 IoU。",
  },
  {
    section_id: "sec-conclusion",
    title_zh: "5 结论",
    text_zh: "层次化对齐是视觉–语言双塔的简单增益。未来工作包括更密监督与视频扩展。",
  },
];
