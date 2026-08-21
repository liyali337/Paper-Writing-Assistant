type IconProps = {
  size?: number;
};

export function LogoMark({ size = 18 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M7 4.5h7.2L19 9.2V19a1.5 1.5 0 0 1-1.5 1.5h-10A1.5 1.5 0 0 1 6 19V6A1.5 1.5 0 0 1 7.5 4.5H7z"
        stroke="currentColor"
        strokeWidth="1.6"
      />
      <path d="M14.2 4.5V8.2a.8.8 0 0 0 .8.8H19" stroke="currentColor" strokeWidth="1.6" />
      <path d="M9 13h6M9 16.5h4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

export function PdfIcon({ size = 26 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M7 3.5h7l5 5V19a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V5.5a2 2 0 0 1 2-2z"
        stroke="currentColor"
        strokeWidth="1.6"
      />
      <path d="M14 3.5V9h5.2" stroke="currentColor" strokeWidth="1.6" />
      <path d="M8.5 13.5h7M8.5 17h5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

export function TreeIcon({ size = 16 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M7 6h13M7 12h13M7 18h13" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
      <circle cx="4" cy="6" r="1.2" fill="currentColor" />
      <circle cx="4" cy="12" r="1.2" fill="currentColor" />
      <circle cx="4" cy="18" r="1.2" fill="currentColor" />
    </svg>
  );
}

export function Chevron({ dir, size = 16 }: IconProps & { dir: "left" | "right" }) {
  const d = dir === "left" ? "M14 5l-7 7 7 7" : "M10 5l7 7-7 7";
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d={d} stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function RefreshIcon({ size = 14 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M20 12a8 8 0 1 1-2.2-5.5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <path d="M20 5v5h-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function SplitIcon({ size = 16 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden>
      <rect x="3.5" y="4.5" width="17" height="15" rx="2" stroke="currentColor" strokeWidth="1.6" />
      <path d="M12 5v14" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}
