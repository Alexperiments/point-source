import type { AgentStatus } from "@/pages/Index";
import { Loader2 } from "lucide-react";

const labels: Record<AgentStatus, string> = {
  idle: "",
  thinking: "Thinking…",
  streaming: "Generating response…",
};

const ThinkingIndicator = ({ status }: { status: AgentStatus }) => {
  if (status === "idle") return null;

  return (
    <div className="flex gap-3">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-accent text-accent-foreground">
        <Loader2 size={14} className="animate-spin" />
      </div>
      <div className="flex items-center pt-0.5">
        <span className="text-sm text-muted-foreground animate-pulse">
          {labels[status]}
        </span>
      </div>
    </div>
  );
};

export default ThinkingIndicator;
