import { useState, useRef, useEffect } from "react";
import { Send, Sparkles } from "lucide-react";
import type { Conversation, AgentStatus } from "@/pages/Index";
import ReactMarkdown from "react-markdown";
import ThinkingIndicator from "@/components/ThinkingIndicator";

interface Props {
  conversation: Conversation | null;
  onSend: (content: string) => void;
  agentStatus: AgentStatus;
}

const ChatArea = ({ conversation, onSend, agentStatus }: Props) => {
  const [input, setInput] = useState("");
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

  return (
    <div className="flex flex-1 flex-col min-w-0">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 && agentStatus === "idle" ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center space-y-3 px-4">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-accent">
                <Sparkles size={22} className="text-muted-foreground" />
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
                    <div className="prose prose-sm prose-neutral max-w-none text-foreground [&_p]:leading-relaxed [&_pre]:bg-accent [&_pre]:text-accent-foreground [&_pre]:rounded-lg [&_pre]:p-3 [&_code]:bg-accent [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-sm [&_a]:text-primary [&_a]:underline [&_a]:underline-offset-2 [&_h2]:text-xs [&_h2]:uppercase [&_h2]:tracking-wide [&_h2]:text-muted-foreground [&_h2]:mt-4 [&_h2]:mb-2 [&_ol]:text-xs [&_ol]:text-muted-foreground">
                      <ReactMarkdown
                        components={{
                          a: ({ href, children }) => (
                            <a href={href} target="_blank" rel="noopener noreferrer" className="text-primary hover:opacity-70 transition-opacity">
                              {children}
                            </a>
                          ),
                        }}
                      >{msg.content}</ReactMarkdown>
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
