import type { Section } from "../api/types";

type Props = {
  sections: Section[];
  activePage: number;
  onJump: (page: number, sectionId?: string) => void;
};

export function Outline({ sections, activePage, onJump }: Props) {
  if (sections.length === 0) {
    return (
      <div className="outline-list">
        <div className="empty">
          <h3>章节</h3>
          <p>解析完成后，这里会按标题列出原文小节。</p>
        </div>
      </div>
    );
  }

  return (
    <div className="outline-list">
      {sections.map((section) => {
        const on = activePage >= section.page_start && activePage <= section.page_end;
        return (
          <button
            key={section.section_id}
            type="button"
            className={`sec${on ? " is-on" : ""}${section.level > 1 ? " is-child" : ""}`}
            onClick={() => onJump(section.page_start, section.section_id)}
          >
            <span className="sec-title">{section.title}</span>
          </button>
        );
      })}
    </div>
  );
}
