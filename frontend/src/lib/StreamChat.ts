import { CHAT_STREAM_URL } from "@/lib/api";
import { getAccessToken } from "@/lib/authStorage";

type Msg = { role: "user" | "assistant"; content: string };

export type StreamStatus =
  | "thinking"
  | "retrieving_documents"
  | "retrieval_timeout"
  | "retrieval_failed";

const handleParsedPayload = (
  parsed: any,
  onDelta: (deltaText: string) => void,
  onStatus?: (status: StreamStatus | string) => void
) => {
  if (parsed?.type === "status") {
    const status = parsed?.status;
    if (typeof status === "string") {
      onStatus?.(status);
    }
    return;
  }

  if (parsed?.type === "error") {
    const message = typeof parsed?.message === "string" ? parsed.message : "Stream failed.";
    throw new Error(message);
  }

  const content = parsed?.choices?.[0]?.delta?.content as string | undefined;
  if (content) {
    onDelta(content);
  }
};

export async function streamChat({
  messages,
  onDelta,
  onDone,
  onStatus,
}: {
  messages: Msg[];
  onDelta: (deltaText: string) => void;
  onDone: () => void;
  onStatus?: (status: StreamStatus | string) => void;
}) {
  const token = getAccessToken();
  const resp = await fetch(CHAT_STREAM_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ messages }),
  });

  if (resp.status === 429) throw new Error("Rate limited — please try again later.");
  if (resp.status === 402) throw new Error("Usage limit reached — please add credits.");
  if (!resp.ok || !resp.body) throw new Error("Failed to get a response.");

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
