import { useState } from "react";

import { Landing } from "./components/Landing";
import { Workspace } from "./components/Workspace";

type Session =
  | { kind: "landing" }
  | { kind: "workspace"; file: File | null; preview: boolean };

export default function App() {
  const [session, setSession] = useState<Session>({ kind: "landing" });

  if (session.kind === "landing") {
    return (
      <Landing
        onPicked={(file) => setSession({ kind: "workspace", file, preview: false })}
        onPreview={() => setSession({ kind: "workspace", file: null, preview: true })}
      />
    );
  }

  return (
    <Workspace
      key={session.preview ? "preview" : "live"}
      file={session.file}
      preview={session.preview}
      onReset={() => setSession({ kind: "landing" })}
      onLoadPreview={() =>
        setSession({ kind: "workspace", file: session.file, preview: true })
      }
    />
  );
}
