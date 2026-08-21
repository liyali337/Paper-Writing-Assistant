import type { Figure, PaperIntro, Section } from "../api/types";
import { FigureCard } from "./FigureViews";
import { SectionReader } from "./SectionReader";

type Props = {
  intro: PaperIntro | null;
  sections?: Section[];
  figures: Figure[];
  paperId: string | null;
  preview: boolean;
  onJump: (page: number) => void;
  onOpenFigure: (figure: Figure) => void;
  onJumpSectionTitle?: (title: string) => void;
  focusSectionId?: string | null;
};

export function IntroPanel({
  intro,
  sections = [],
  figures,
  paperId,
  preview,
  onJump,
  onOpenFigure,
  onJumpSectionTitle,
  focusSectionId,
}: Props) {
  if (!intro) {
    return (
      <SectionReader
        sections={sections}
        figures={figures}
        paperId={paperId}
        preview={preview}
        heading="章节原文"
        note={
          sections.length
            ? "中文总体介绍将在下一阶段生成。下面是按标题切开的原文与抽出的图，可点页码跳到左侧 PDF。"
            : "解析完成后会按标题列出章节原文，并生成中文问题、动机与贡献。"
        }
        focusId={focusSectionId}
        onJump={onJump}
        onOpenFigure={onOpenFigure}
      />
    );
  }

  const overview = figures.find((figure) => figure.figure_id === intro.overview_figure_id);
  const firstEvidence = intro.evidence[0];

  return (
    <article className="prose">
      <h2>{intro.title_zh}</h2>
      <p className="lead">{intro.one_sentence_zh}</p>

      <section className="block">
        <h3>问题</h3>
        <p>
          {intro.problem_zh}
          {firstEvidence ? <Cite page={firstEvidence.page} onJump={onJump} /> : null}
        </p>
      </section>

      <section className="block">
        <h3>动机</h3>
        <p>{intro.motivation_zh}</p>
      </section>

      <section className="block">
        <h3>贡献</h3>
        <ol>
          {intro.contributions.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ol>
      </section>

      {intro.task_setting_zh ? (
        <section className="block">
          <h3>任务设定</h3>
          <p>{intro.task_setting_zh}</p>
        </section>
      ) : null}

      {intro.reading_order_zh.length ? (
        <section className="block">
          <h3>建议阅读顺序</h3>
          <div className="order-list">
            {intro.reading_order_zh.map((title) => (
              <button key={title} type="button" onClick={() => onJumpSectionTitle?.(title)}>
                {title}
              </button>
            ))}
          </div>
        </section>
      ) : null}

      {overview ? (
        <FigureCard
          figure={overview}
          paperId={paperId}
          preview={preview}
          onOpen={onOpenFigure}
        />
      ) : null}

      {intro.evidence.map((item) =>
        item.quote ? (
          <details className="quote" key={`${item.page}-${item.quote}`}>
            <summary>
              原文摘录{item.sourced ? " · 已核对" : " · 未核对"} · p.{item.page}
            </summary>
            <blockquote>{item.quote}</blockquote>
          </details>
        ) : null,
      )}
    </article>
  );
}

function Cite({ page, onJump }: { page: number; onJump: (page: number) => void }) {
  return (
    <button className="cite" type="button" onClick={() => onJump(page)}>
      p.{page}
    </button>
  );
}
