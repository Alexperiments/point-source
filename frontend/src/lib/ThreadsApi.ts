import { THREADS_BASE_URL } from "@/lib/api";

export type ThreadMessagePayload = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  sequence_num: number;
  created_at: string;
};

export type ThreadPayload = {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  messages: ThreadMessagePayload[];
};

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

const authorizedRequest = async (input: RequestInfo, init: RequestInit = {}) => {
  const response = await fetch(input, {
    ...init,
    credentials: init.credentials ?? "include",
    headers: {
      ...(init.headers || {}),
    },
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  return response;
};

export const listThreads = async (): Promise<ThreadPayload[]> => {
  const response = await authorizedRequest(THREADS_BASE_URL, { method: "GET" });
  return response.json();
};

export const createThread = async (title: string): Promise<ThreadPayload> => {
  const response = await authorizedRequest(THREADS_BASE_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title }),
  });
  return response.json();
};

export const deleteThread = async (threadId: string): Promise<void> => {
  await authorizedRequest(`${THREADS_BASE_URL}/${threadId}`, {
    method: "DELETE",
  });
};
