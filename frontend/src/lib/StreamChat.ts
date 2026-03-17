import { CHAT_STREAM_URL } from "@/lib/api";
import { isRecord } from "@/lib/errors";

type Msg = { role: "user" | "assistant"; content: string };

export class ChatStreamError extends Error {
  status: number;
  code: "daily_quota" | "usage_limit" | "request_failed";
  retryAfterSeconds: number | null;

  constructor({
    message,
    status,
    code,
    retryAfterSeconds = null,
  }: {
    message: string;
    status: number;
    code: "daily_quota" | "usage_limit" | "request_failed";
    retryAfterSeconds?: number | null;
  }) {
    super(message);
    this.name = "ChatStreamError";
    this.status = status;
    this.code = code;
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

export type StreamStatus =
  | "thinking"
  | "retrieving_documents"
  | "retrieval_timeout"
  | "retrieval_failed";

const parseErrorMessage = async (response: Response): Promise<string> => {
  const text = await response.text();
  if (!text) {
    return `Request failed with status ${response.status}.`;
  }

  try {
    const payload = JSON.parse(text);
    if (typeof payload?.detail === "string") return payload.detail;
  } catch {
    return text;
  }

  return `Request failed with status ${response.status}.`;
};

const parseRetryAfterSeconds = (value: string | null): number | null => {
  if (!value) return null;

  const seconds = Number.parseInt(value, 10);
  if (Number.isNaN(seconds) || seconds < 0) {
    return null;
  }

  return seconds;
};

type StreamPayload = {
  type?: string;
  status?: string;
  message?: string;
  choices?: Array<{
    delta?: {
      content?: string;
    };
  }>;
};

const parseStreamPayload = (value: unknown): StreamPayload | null => {
  if (!isRecord(value)) {
    return null;
  }

  return value as StreamPayload;
};

const handleParsedPayload = (
  parsed: unknown,
  onDelta: (deltaText: string) => void,
  onStatus?: (status: StreamStatus | string) => void
) => {
  const payload = parseStreamPayload(parsed);
  if (!payload) {
    return;
  }

  if (payload.type === "status") {
    const status = payload.status;
    if (typeof status === "string") {
      onStatus?.(status);
    }
    return;
  }

  if (payload.type === "error") {
    const message = typeof payload.message === "string" ? payload.message : "Stream failed.";
    throw new Error(message);
  }

  const content = payload.choices?.[0]?.delta?.content;
  if (content) {
    onDelta(content);
  }
};

export async function streamChat({
  messages,
  threadId,
  onDelta,
  onDone,
  onStatus,
}: {
  messages: Msg[];
  threadId?: string;
  onDelta: (deltaText: string) => void;
  onDone: () => void;
  onStatus?: (status: StreamStatus | string) => void;
}) {
  const resp = await fetch(CHAT_STREAM_URL, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ messages, thread_id: threadId }),
  });

  if (resp.status === 429) {
    const message = await parseErrorMessage(resp);
    const retryAfterSeconds = parseRetryAfterSeconds(resp.headers.get("Retry-After"));
    const isDailyQuota =
      message.toLowerCase().includes("daily message limit reached") ||
      retryAfterSeconds !== null;

    throw new ChatStreamError({
      message,
      status: resp.status,
      code: isDailyQuota ? "daily_quota" : "request_failed",
      retryAfterSeconds,
    });
  }

  if (resp.status === 402) {
    throw new ChatStreamError({
      message: await parseErrorMessage(resp),
      status: resp.status,
      code: "usage_limit",
    });
  }

  if (!resp.ok || !resp.body) {
    throw new ChatStreamError({
      message: await parseErrorMessage(resp),
      status: resp.status,
      code: "request_failed",
    });
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let textBuffer = "";
  let streamDone = false;

  while (!streamDone) {
    const { done, value } = await reader.read();
    if (done) break;
    textBuffer += decoder.decode(value, { stream: true });

    let newlineIndex: number;
    while ((newlineIndex = textBuffer.indexOf("\n")) !== -1) {
      let line = textBuffer.slice(0, newlineIndex);
      textBuffer = textBuffer.slice(newlineIndex + 1);

      if (line.endsWith("\r")) line = line.slice(0, -1);
      if (line.startsWith(":") || line.trim() === "") continue;
      if (!line.startsWith("data: ")) continue;

      const jsonStr = line.slice(6).trim();
      if (jsonStr === "[DONE]") {
        streamDone = true;
        break;
      }

      try {
        const parsed = JSON.parse(jsonStr);
        handleParsedPayload(parsed, onDelta, onStatus);
      } catch {
        textBuffer = line + "\n" + textBuffer;
        break;
      }
    }
  }

  // Final flush
  if (textBuffer.trim()) {
    for (let raw of textBuffer.split("\n")) {
      if (!raw) continue;
      if (raw.endsWith("\r")) raw = raw.slice(0, -1);
      if (raw.startsWith(":") || raw.trim() === "") continue;
      if (!raw.startsWith("data: ")) continue;
      const jsonStr = raw.slice(6).trim();
      if (jsonStr === "[DONE]") continue;
      try {
        const parsed = JSON.parse(jsonStr);
        handleParsedPayload(parsed, onDelta, onStatus);
      } catch {
        // Ignore malformed final chunk
      }
    }
  }

  onDone();
}
