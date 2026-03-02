import { useState, useRef, useEffect } from "react";
import { Check, Copy, RotateCcw, Send, Telescope } from "lucide-react";
import type { Conversation, AgentStatus } from "@/pages/Index";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import ThinkingIndicator from "@/components/ThinkingIndicator";
import { toast } from "sonner";

interface Props {
  conversation: Conversation | null;
  onSend: (content: string) => void;
  onRetry: (assistantMessageId: string) => void;
  agentStatus: AgentStatus;
}

const normalizeAssistantMarkdown = (content: string): string => {
  const lines = content.split("\n");
  let inFence = false;

  for (let i = 0; i < lines.length; i += 1) {
    const current = lines[i];
    if (current.trimStart().startsWith("```")) {
      inFence = !inFence;
      continue;
    }

    if (inFence || !current.trim().endsWith(":")) {
      continue;
    }

    let j = i + 1;
    while (j < lines.length && lines[j].trim() === "") {
      j += 1;
    }

    const blockIndexes: number[] = [];
    while (j < lines.length) {
      const candidate = lines[j];
      if (candidate.trim() === "") {
        break;
      }
      if (!candidate.startsWith("    ")) {
        break;
      }
      blockIndexes.push(j);
      j += 1;
    }

    if (blockIndexes.length < 2) {
      continue;
    }

    for (const idx of blockIndexes) {
      lines[idx] = `- ${lines[idx].trim()}`;
    }
  }

  return lines.join("\n");
};

const ChatArea = ({ conversation, onSend, onRetry, agentStatus }: Props) => {
  const [input, setInput] = useState("");
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation?.messages, agentStatus]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + "px";
    }
  }, [input]);

  const handleSubmit = () => {
    if (!input.trim() || agentStatus !== "idle") return;
    onSend(input.trim());
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const messages = conversation?.messages ?? [];
  const lastAssistantMessageId = [...messages].reverse().find((m) => m.role === "assistant")?.id;

  const handleCopy = async (messageId: string, content: string) => {
    if (!navigator.clipboard) {
      toast.error("Clipboard is not available.");
      return;
    }

    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageId(messageId);
      window.setTimeout(() => {
        setCopiedMessageId((current) => (current === messageId ? null : current));
      }, 1200);
    } catch {
      toast.error("Failed to copy response.");
    }
  };

  return (
    <div className="flex flex-1 flex-col min-w-0">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 && agentStatus === "idle" ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center space-y-3 px-4">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-accent">
                <Telescope size={22} className="text-primary/70" />
              </div>
              <h2 className="text-lg font-medium text-foreground">How can I help you today?</h2>
              <p className="text-sm text-muted-foreground max-w-sm">
                Start a conversation by typing a message below.
              </p>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-2xl px-4 py-6 space-y-6">
            {messages.map((msg) => (
              <div key={msg.id} className="flex gap-3">
                <div
                  className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-medium
                    ${msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-accent text-accent-foreground"
                    }`}
                >
                  {msg.role === "user" ? "Y" : "AI"}
                </div>
                <div className="flex-1 pt-0.5 min-w-0">
                  <p className="text-[13px] font-medium mb-1 text-muted-foreground">
                    {msg.role === "user" ? "You" : "Assistant"}
                  </p>
                  {msg.role === "assistant" ? (
                    <div>
                      <div className="max-w-none font-sans text-[15px] leading-7 text-foreground [&_p]:my-0 [&_p+*]:mt-4 [&_ul]:my-3 [&_ul]:list-disc [&_ul]:pl-6 [&_ol]:my-3 [&_ol]:list-decimal [&_ol]:pl-6 [&_li]:my-1 [&_strong]:font-semibold [&_h1]:my-4 [&_h1]:text-[15px] [&_h1]:font-semibold [&_h2]:my-4 [&_h2]:text-[15px] [&_h2]:font-semibold [&_h3]:my-4 [&_h3]:text-[15px] [&_h3]:font-semibold [&_pre]:my-4 [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-accent [&_pre]:p-3 [&_pre]:text-accent-foreground [&_code]:rounded [&_code]:bg-accent [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-[0.9em] [&_a]:text-primary [&_a]:underline [&_a]:underline-offset-2">
                        <ReactMarkdown
                          remarkPlugins={[remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                          components={{
                            a: ({ href, children }) => (
                              <a href={href} target="_blank" rel="noopener noreferrer" className="text-primary hover:opacity-70 transition-opacity">
                                {children}
                              </a>
                            ),
                          }}
                        >{normalizeAssistantMarkdown(msg.content)}</ReactMarkdown>
                      </div>
                      <div className="mt-2 flex items-center gap-2 text-xs">
                        <button
                          onClick={() => void handleCopy(msg.id, msg.content)}
                          className="flex items-center gap-1 rounded-md px-2 py-1 text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
                          aria-label="Copy assistant response"
                        >
                          {copiedMessageId === msg.id ? <Check size={13} /> : <Copy size={13} />}
                          {copiedMessageId === msg.id ? "Copied" : "Copy"}
                        </button>
                        {msg.id === lastAssistantMessageId && (
                          <button
                            onClick={() => onRetry(msg.id)}
                            disabled={agentStatus !== "idle"}
                            className="flex items-center gap-1 rounded-md px-2 py-1 text-muted-foreground hover:bg-accent hover:text-foreground transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            aria-label="Retry assistant response"
                          >
                            <RotateCcw size={13} />
                            Retry
                          </button>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm leading-relaxed text-foreground whitespace-pre-wrap">
                      {msg.content}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {agentStatus !== "idle" && <ThinkingIndicator status={agentStatus} />}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="border-t border-border px-4 py-3">
        <div className="mx-auto max-w-2xl">
          <div className="flex items-end gap-2 rounded-xl border border-input bg-card p-2 shadow-sm focus-within:ring-1 focus-within:ring-ring transition-shadow">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message..."
              rows={1}
              className="flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || agentStatus !== "idle"}
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground disabled:opacity-30 hover:opacity-90 transition-opacity"
            >
              <Send size={15} />
            </button>
          </div>
          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            AI can make mistakes. Verify important information.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChatArea;
