import { useState } from "react";
import ChatSidebar from "@/components/ChatSidebar";
import ChatArea from "@/components/ChatArea";
import { Menu } from "lucide-react";
import { streamChat } from "@/lib/StreamChat";
import { toast } from "sonner";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
}

export type AgentStatus = "idle" | "thinking" | "streaming";

const Index = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [agentStatus, setAgentStatus] = useState<AgentStatus>("idle");

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null;

  const createConversation = () => {
    const conv: Conversation = {
      id: crypto.randomUUID(),
      title: "New chat",
      messages: [],
      createdAt: new Date(),
    };
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
  };

  const sendMessage = async (content: string) => {
    let convId = activeId;
    let currentMessages: ChatMessage[] = [];

    if (!convId) {
      const conv: Conversation = {
        id: crypto.randomUUID(),
        title: content.slice(0, 40),
        messages: [],
        createdAt: new Date(),
      };
      setConversations((prev) => [conv, ...prev]);
      convId = conv.id;
      setActiveId(conv.id);
      currentMessages = [];
    } else {
      currentMessages = conversations.find((c) => c.id === convId)?.messages ?? [];
    }

    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content };

    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convId) return c;
        return {
          ...c,
          title: c.messages.length === 0 ? content.slice(0, 40) : c.title,
          messages: [...c.messages, userMsg],
        };
      })
    );

    setAgentStatus("thinking");

    const apiMessages = [...currentMessages, userMsg].map((m) => ({
      role: m.role,
      content: m.content,
    }));

    let assistantSoFar = "";
    const assistantId = crypto.randomUUID();

    try {
      await streamChat({
        messages: apiMessages,
        onDelta: (chunk) => {
          if (agentStatus !== "streaming") setAgentStatus("streaming");
          assistantSoFar += chunk;
          const updatedContent = assistantSoFar;
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== convId) return c;
              const last = c.messages[c.messages.length - 1];
              if (last?.id === assistantId) {
                return {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === assistantId ? { ...m, content: updatedContent } : m
                  ),
                };
              }
              return {
                ...c,
                messages: [
                  ...c.messages,
                  { id: assistantId, role: "assistant" as const, content: updatedContent },
                ],
              };
            })
          );
        },
        onDone: () => setAgentStatus("idle"),
      });
    } catch (e: any) {
      setAgentStatus("idle");
      toast.error(e.message || "Failed to get response");
    }
  };

  const deleteConversation = (id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (activeId === id) setActiveId(null);
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      {!sidebarOpen && (
        <button
          onClick={() => setSidebarOpen(true)}
          className="fixed left-3 top-3 z-50 rounded-md p-2 text-muted-foreground hover:bg-accent transition-colors"
        >
          <Menu size={20} />
        </button>
      )}

      <ChatSidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={createConversation}
        onDelete={deleteConversation}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <ChatArea
        conversation={activeConversation}
        onSend={sendMessage}
        agentStatus={agentStatus}
      />
    </div>
  );
};

export default Index;
