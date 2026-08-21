"""讲解层。M0 空实现，M2 生成总体介绍与方法详解。"""

from dl_agent.domain.models import MethodExplain, PaperIntro


def build_intro(paper_id: str) -> PaperIntro:
    raise NotImplementedError("understand.build_intro is scheduled for M2")


def build_method(paper_id: str) -> MethodExplain:
    raise NotImplementedError("understand.build_method is scheduled for M2")
