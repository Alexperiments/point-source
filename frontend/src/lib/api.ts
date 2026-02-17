const DEFAULT_API_BASE_URL = "";

const rawBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() || DEFAULT_API_BASE_URL;

export const API_BASE_URL = rawBaseUrl.replace(/\/+$/, "");

const authPath = import.meta.env.VITE_AUTH_BASE_PATH?.trim() || "/v1/auth";
const chatPath =
  import.meta.env.VITE_CHAT_STREAM_PATH?.trim() || "/v1/llm/chat/stream";

const ensureLeadingSlash = (value: string) =>
  value.startsWith("/") ? value : `/${value}`;

export const AUTH_BASE_URL = `${API_BASE_URL}${ensureLeadingSlash(authPath)}`;
export const CHAT_STREAM_URL = `${API_BASE_URL}${ensureLeadingSlash(chatPath)}`;
