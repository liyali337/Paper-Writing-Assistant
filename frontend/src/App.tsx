import { useState } from "react";

import { Landing } from "./components/Landing";
import { Workspace } from "./components/Workspace";

type Session =
  | { kind: "read-landing" }
  | { kind: "workspace"; file: File | null; preview: boolean; paperId?: string };

export default function App() {
  const [session, setSession] = useState<Session>({ kind: "read-landing" });

  function goHome() {
    setSession({ kind: "read-landing" });
  }

  if (session.kind === "read-landing") {
    return (
      <Landing
        onBrand={goHome}
        onPicked={(file) => setSession({ kind: "workspace", file, preview: false })}
        onOpen={(paperId) => setSession({ kind: "workspace", file: null, preview: false, paperId })}
        onPreview={() => setSession({ kind: "workspace", file: null, preview: true })}
      />
    );
  }

  return (
    <Workspace
      key={session.preview ? "preview" : session.paperId ?? session.file?.name ?? "live"}
      file={session.file}
      preview={session.preview}
      paperId={session.paperId}
      onReset={goHome}
      onLoadPreview={() =>
        setSession({ kind: "workspace", file: session.file, preview: true, paperId: session.paperId })
      }
    />
  );
}
