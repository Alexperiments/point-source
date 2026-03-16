import { useEffect, useState } from "react";
import ChatSidebar from "@/components/ChatSidebar";
import ChatArea from "@/components/ChatArea";
import { ChatStreamError, streamChat } from "@/lib/StreamChat";
import { toast } from "sonner";
import { useAuth } from "@/context/AuthContext";
import { useIsMobile } from "@/hooks/use-mobile";
import {
  createThread,
  deleteThread,
  listThreads,
  type ThreadPayload,
} from "@/lib/ThreadsApi";
import { AUTH_BASE_URL } from "@/lib/api";

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

type UsageSummary = {
  isPremium: boolean;
  requestsRemaining: number | null;
  resetInSeconds: number;
};

const parseUsageSummary = (payload: unknown): UsageSummary => {
  if (!payload || typeof payload !== "object") {
    throw new Error("Invalid usage payload from server.");
  }

  const raw = payload as Record<string, unknown>;
  const isPremium = raw.is_premium;
  const requestsRemaining = raw.requests_remaining;
  const resetInSeconds = raw.reset_in_seconds;

  if (
    typeof isPremium !== "boolean" ||
    (requestsRemaining !== null && typeof requestsRemaining !== "number") ||
    typeof resetInSeconds !== "number"
  ) {
    throw new Error("Unexpected usage payload from server.");
  }

  return {
    isPremium,
    requestsRemaining,
    resetInSeconds,
  };
};

const fetchUsageSummary = async (): Promise<UsageSummary> => {
  const response = await fetch(`${AUTH_BASE_URL}/users/me/usage`, {
    method: "GET",
    credentials: "include",
  });

  const text = await response.text();
  let payload: unknown = null;

  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }

  if (!response.ok) {
    throw new Error(`Failed to load usage with status ${response.status}.`);
  }

  return parseUsageSummary(payload);
};

const Index = () => {
  const { user, isLoading } = useAuth();
  const isMobile = useIsMobile();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(() =>
    typeof window !== "undefined" ? window.innerWidth >= 768 : true
  );
  const [agentStatus, setAgentStatus] = useState<AgentStatus>("idle");
  const [loginPromptVersion, setLoginPromptVersion] = useState(0);
  const [dailyQuotaRemainingSeconds, setDailyQuotaRemainingSeconds] = useState<number | null>(null);

  const activeConversation = conversations.find((c) => c.id === activeId) ?? null;

  useEffect(() => {
    if (typeof window === "undefined") return;
    setSidebarOpen(window.innerWidth >= 768);
  }, [isMobile]);

  useEffect(() => {
    if (isLoading) return;

    if (!user) {
      setConversations([]);
      setActiveId(null);
      setAgentStatus("idle");
      setDailyQuotaRemainingSeconds(null);
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

  useEffect(() => {
    if (isLoading || !user) return;

    let cancelled = false;

    const loadUsage = async () => {
      try {
        const usage = await fetchUsageSummary();
        if (cancelled) return;

        if (!usage.isPremium && usage.requestsRemaining === 0) {
          setDailyQuotaRemainingSeconds(Math.max(1, usage.resetInSeconds));
          return;
        }

        setDailyQuotaRemainingSeconds(null);
      } catch {
        if (!cancelled) {
          setDailyQuotaRemainingSeconds(null);
        }
      }
    };

    void loadUsage();

    return () => {
      cancelled = true;
    };
  }, [isLoading, user]);

  useEffect(() => {
    if (dailyQuotaRemainingSeconds === null) return;
    if (dailyQuotaRemainingSeconds <= 0) {
      setDailyQuotaRemainingSeconds(null);
      return;
    }

    const timeout = window.setTimeout(() => {
      setDailyQuotaRemainingSeconds((current) =>
        current === null ? null : Math.max(0, current - 1)
      );
    }, 1000);

    return () => {
      window.clearTimeout(timeout);
    };
  }, [dailyQuotaRemainingSeconds]);

  const createConversation = () => {
    setActiveId(null);
    if (isMobile) {
      setSidebarOpen(false);
    }
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
      if (e instanceof ChatStreamError && e.code === "daily_quota") {
        setDailyQuotaRemainingSeconds(Math.max(1, e.retryAfterSeconds ?? 1));
        return;
      }

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
    <div className="flex h-dvh min-h-0 w-full overflow-hidden bg-background">
      <ChatSidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={(id) => {
          setActiveId(id);
          if (isMobile) {
            setSidebarOpen(false);
          }
        }}
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
        onOpenSidebar={() => setSidebarOpen(true)}
        dailyQuotaRemainingSeconds={dailyQuotaRemainingSeconds}
      />
    </div>
  );
};

export default Index;
