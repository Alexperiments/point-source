import { useEffect, useState } from "react";
import ChatSidebar from "@/components/ChatSidebar";
import ChatArea from "@/components/ChatArea";
import { Menu } from "lucide-react";
import { streamChat } from "@/lib/StreamChat";
import { toast } from "sonner";
import { useAuth } from "@/context/AuthContext";

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

export type AgentStatus = "idle" | "thinking" | "retrieving" | "streaming";

const Index = () => {
  const { user, isLoading } = useAuth();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [agentStatus, setAgentStatus] = useState<AgentStatus>("idle");
  const [loginPromptVersion, setLoginPromptVersion] = useState(0);

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null;

  useEffect(() => {
    if (isLoading || user) return;
    setConversations([]);
    setActiveId(null);
    setAgentStatus("idle");
  }, [isLoading, user]);

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

  const streamAssistantReply = async ({
    convId,
    apiMessages,
    assistantId,
  }: {
    convId: string;
    apiMessages: { role: "user" | "assistant"; content: string }[];
    assistantId: string;
  }) => {
    let assistantSoFar = "";

    await streamChat({
      messages: apiMessages,
      onStatus: (status) => {
        if (status === "retrieving_documents") {
          setAgentStatus("retrieving");
          return;
        }

        if (status === "retrieval_timeout" || status === "retrieval_failed") {
          setAgentStatus("thinking");
          return;
        }

        if (status === "thinking") {
          setAgentStatus("thinking");
        }
      },
      onDelta: (chunk) => {
        setAgentStatus("streaming");
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
  };

  const sendMessage = async (content: string) => {
    if (isLoading || agentStatus !== "idle") return;
    if (!user) {
      setLoginPromptVersion((prev) => prev + 1);
      toast.error("Please log in to start chatting.");
      return;
    }

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
    const assistantId = crypto.randomUUID();

    try {
      await streamAssistantReply({
        convId,
        apiMessages,
        assistantId,
      });
    } catch (e: any) {
      setAgentStatus("idle");
      toast.error(e.message || "Failed to get response");
    }
  };

  const retryAssistantMessage = async (assistantMessageId: string) => {
    if (isLoading || agentStatus !== "idle" || !activeConversation) return;
    if (!user) {
      setLoginPromptVersion((prev) => prev + 1);
      toast.error("Please log in to retry.");
      return;
    }

    const targetIndex = activeConversation.messages.findIndex((m) => m.id === assistantMessageId);
    if (targetIndex <= 0) return;

    const targetMessage = activeConversation.messages[targetIndex];
    const previousMessage = activeConversation.messages[targetIndex - 1];
    if (targetMessage?.role !== "assistant" || previousMessage?.role !== "user") return;

    const replayMessages = activeConversation.messages.slice(0, targetIndex);
    const convId = activeConversation.id;

    setConversations((prev) =>
      prev.map((c) => (c.id === convId ? { ...c, messages: replayMessages } : c))
    );
    setAgentStatus("thinking");

    const apiMessages = replayMessages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      await streamAssistantReply({
        convId,
        apiMessages,
        assistantId: assistantMessageId,
      });
    } catch (e: any) {
      setAgentStatus("idle");
      toast.error(e.message || "Failed to regenerate response");
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
        loginPromptVersion={loginPromptVersion}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <ChatArea
        conversation={activeConversation}
        onSend={sendMessage}
        onRetry={retryAssistantMessage}
        agentStatus={agentStatus}
      />
    </div>
  );
};

export default Index;
