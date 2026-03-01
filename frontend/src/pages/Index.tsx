import { useEffect, useState } from "react";
import ChatSidebar from "@/components/ChatSidebar";
import ChatArea from "@/components/ChatArea";
import { Menu } from "lucide-react";
import { streamChat } from "@/lib/StreamChat";
import { toast } from "sonner";
import { useAuth } from "@/context/AuthContext";
import {
  createThread,
  deleteThread,
  listThreads,
  type ThreadPayload,
} from "@/lib/ThreadsApi";

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
  updatedAt: Date;
}

export type AgentStatus = "idle" | "thinking" | "retrieving" | "streaming";

const toConversation = (thread: ThreadPayload): Conversation => ({
  id: thread.id,
  title: thread.title?.trim() || "New chat",
  createdAt: new Date(thread.created_at),
  updatedAt: new Date(thread.updated_at),
  messages: thread.messages
    .filter((message) => message.role === "user" || message.role === "assistant")
    .map((message) => ({
      id: message.id,
      role: message.role === "user" ? "user" : "assistant",
      content: message.content,
    })),
});

const Index = () => {
  const { user, isLoading } = useAuth();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [agentStatus, setAgentStatus] = useState<AgentStatus>("idle");
  const [loginPromptVersion, setLoginPromptVersion] = useState(0);

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null;

  useEffect(() => {
    if (isLoading) return;

    if (!user) {
      setConversations([]);
      setActiveId(null);
      setAgentStatus("idle");
      return;
    }

    let cancelled = false;

    const loadHistory = async () => {
      try {
        const threads = await listThreads();
        if (cancelled) return;

        const loadedConversations = threads.map(toConversation);
        setConversations(loadedConversations);
        setActiveId((prev) => {
          if (prev && loadedConversations.some((conversation) => conversation.id === prev)) {
            return prev;
          }
          return loadedConversations[0]?.id ?? null;
        });
      } catch (error: any) {
        if (!cancelled) {
          toast.error(error?.message || "Failed to load chat history.");
        }
      }
    };

    void loadHistory();

    return () => {
      cancelled = true;
    };
  }, [isLoading, user]);

  const createConversation = () => {
    setActiveId(null);
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
      threadId: convId,
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
                updatedAt: new Date(),
                messages: c.messages.map((m) =>
                  m.id === assistantId ? { ...m, content: updatedContent } : m
                ),
              };
            }
            return {
              ...c,
              updatedAt: new Date(),
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
    if (isLoading) return;
    if (!user) {
      setLoginPromptVersion((prev) => prev + 1);
      toast.error("Please log in to start chatting.");
      return;
    }
    if (agentStatus !== "idle") return;

    let convId = activeId;
    let currentMessages: ChatMessage[] = [];

    try {
      if (!convId) {
        const thread = await createThread(content.slice(0, 40));
        convId = thread.id;
        setActiveId(convId);
        setConversations((prev) => [toConversation(thread), ...prev]);
      } else {
        currentMessages = conversations.find((c) => c.id === convId)?.messages ?? [];
      }
    } catch (error: any) {
      toast.error(error?.message || "Failed to create chat.");
      return;
    }

    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content };

    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convId) return c;
        return {
          ...c,
          title: c.messages.length === 0 ? content.slice(0, 40) : c.title,
          updatedAt: new Date(),
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

    const targetIndex = activeConversation.messages.findIndex((m) => m.id === assistantMessageId);
    if (targetIndex <= 0) return;

    const previousMessage = activeConversation.messages[targetIndex - 1];
    if (previousMessage?.role !== "user") return;

    await sendMessage(previousMessage.content);
  };

  const deleteConversation = async (id: string) => {
    if (isLoading || !user) return;

    try {
      await deleteThread(id);
      setConversations((prev) => {
        const next = prev.filter((c) => c.id !== id);
        setActiveId((current) => (current === id ? (next[0]?.id ?? null) : current));
        return next;
      });
    } catch (error: any) {
      toast.error(error?.message || "Failed to delete chat.");
    }
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
