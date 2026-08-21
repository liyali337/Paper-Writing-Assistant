type Props = {
  figureId: string;
};

export function DemoDiagram({ figureId }: Props) {
  if (figureId === "fig-pipeline") {
    return (
      <svg viewBox="0 0 960 360" role="img" aria-label="Token construction">
        <rect width="960" height="360" fill="#f8fafc" />
        {["Patch P", "Region R", "CLS I"].map((label, i) => (
          <g key={label} transform={`translate(${80 + i * 280} 70)`}>
            <rect width="200" height="88" rx="12" fill="#fff" stroke="#99f6e4" strokeWidth="2" />
            <text x="100" y="52" textAnchor="middle" fontSize="20" fill="#0f766e" fontFamily="sans-serif">
              {label}
            </text>
          </g>
        ))}
        <path d="M180 168v40h600v40" fill="none" stroke="#0d9488" strokeWidth="2.4" />
        <rect x="280" y="240" width="400" height="64" rx="12" fill="#0f766e" />
        <text x="480" y="280" textAnchor="middle" fontSize="20" fill="#fff" fontFamily="sans-serif">
          Shared projection
        </text>
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 960 420" role="img" aria-label="HieraAlign overview">
      <rect width="960" height="420" fill="#f8fafc" />
      <text x="48" y="42" fontSize="16" fill="#64748b" fontFamily="sans-serif">
        Image
      </text>
      <text x="620" y="42" fontSize="16" fill="#64748b" fontFamily="sans-serif">
        Text
      </text>
      {[
        ["Patch tokens", "Noun phrases", 90],
        ["Region tokens", "Sentences", 190],
        ["Image CLS", "Caption", 290],
      ].map(([left, right, y]) => (
        <g key={left}>
          <rect x="48" y={Number(y)} width="260" height="72" rx="12" fill="#fff" stroke="#cbd5e1" />
          <text x="178" y={Number(y) + 42} textAnchor="middle" fontSize="18" fill="#0f172a" fontFamily="sans-serif">
            {left}
          </text>
          <rect x="640" y={Number(y)} width="260" height="72" rx="12" fill="#fff" stroke="#99f6e4" strokeWidth="2" />
          <text x="770" y={Number(y) + 42} textAnchor="middle" fontSize="18" fill="#0f766e" fontFamily="sans-serif">
            {right}
          </text>
          <path
            d={`M318 ${Number(y) + 36} H630`}
            stroke="#0d9488"
            strokeWidth="2"
            strokeDasharray="6 6"
          />
        </g>
      ))}
    </svg>
  );
}
