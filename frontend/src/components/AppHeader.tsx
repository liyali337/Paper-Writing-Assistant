import type { ReactNode } from "react";

import { LogoMark } from "./icons";

type Props = {
  onBrand: () => void;
  center?: ReactNode;
  actions?: ReactNode;
};

export function AppHeader({ onBrand, center, actions }: Props) {
  return (
    <header className="topbar">
      <button className="brand" type="button" onClick={onBrand}>
        <span className="brand-mark">
          <LogoMark />
        </span>
        <span className="brand-name">
          论文助手 <span className="brand-sub">Studio</span>
        </span>
      </button>
      <div className="topbar-file">{center}</div>
      {actions ? <div className="topbar-actions">{actions}</div> : null}
    </header>
  );
}
