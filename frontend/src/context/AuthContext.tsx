import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { AUTH_BASE_URL } from "@/lib/api";
import { clearAccessToken, getAccessToken, setAccessToken } from "@/lib/authStorage";

export type AuthUser = {
  id: string;
  name: string;
  email: string;
};

type LoginInput = {
  email: string;
  password: string;
};

type RegisterInput = {
  name: string;
  email: string;
  password: string;
};

type ProfileUpdateInput = {
  name: string;
  email?: string;
  currentPassword?: string;
  newPassword?: string;
  confirmPassword?: string;
};

type AuthContextValue = {
  user: AuthUser | null;
  isLoading: boolean;
  login: (input: LoginInput) => Promise<void>;
  register: (input: RegisterInput) => Promise<void>;
  updateProfile: (input: ProfileUpdateInput) => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const normalizeEmail = (email: string) => email.trim().toLowerCase();

const parseErrorMessage = (payload: unknown): string | null => {
  if (!payload) return null;
  if (typeof payload === "string") return payload;
  if (typeof payload !== "object") return null;

  const asRecord = payload as Record<string, unknown>;
  const detail = asRecord.detail;

  if (typeof detail === "string") return detail;

  if (Array.isArray(detail) && detail.length > 0) {
    const firstError = detail[0];
    if (typeof firstError === "string") return firstError;
    if (typeof firstError === "object" && firstError !== null) {
      const message = (firstError as Record<string, unknown>).msg;
      if (typeof message === "string") return message;
    }
  }

  const message = asRecord.message;
  if (typeof message === "string") return message;

  return null;
};

const requestJson = async <T,>(url: string, init: RequestInit): Promise<T> => {
  const response = await fetch(url, init);
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
    throw new Error(
      parseErrorMessage(payload) || `Request failed with status ${response.status}.`
    );
  }

  return payload as T;
};

const parseUser = (payload: unknown): AuthUser => {
  if (!payload || typeof payload !== "object") {
    throw new Error("Invalid user data from server.");
  }

  const raw = payload as Record<string, unknown>;

  const id = raw.id;
  const email = raw.email;
  const name =
    raw.name ||
    raw.full_name ||
    raw.username ||
    raw.display_name;

  if (typeof id !== "string" || typeof email !== "string" || typeof name !== "string") {
    throw new Error("Unexpected user payload from server.");
  }

  return { id, email, name };
};

const fetchCurrentUser = async (token: string) => {
  const user = await requestJson<unknown>(`${AUTH_BASE_URL}/users/me`, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  return parseUser(user);
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    const loadSession = async () => {
      const token = getAccessToken();

      if (!token) {
        if (mounted) setIsLoading(false);
        return;
      }

      try {
        const currentUser = await fetchCurrentUser(token);
        if (mounted) setUser(currentUser);
      } catch {
        clearAccessToken();
        if (mounted) setUser(null);
      } finally {
        if (mounted) setIsLoading(false);
      }
    };

    void loadSession();

    return () => {
      mounted = false;
    };
  }, []);

  const login = async ({ email, password }: LoginInput) => {
    const normalizedEmail = normalizeEmail(email);
    const trimmedPassword = password.trim();

    if (!normalizedEmail || !trimmedPassword) {
      throw new Error("Email and password are required.");
    }

    const tokenData = await requestJson<Record<string, unknown>>(`${AUTH_BASE_URL}/token`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email: normalizedEmail,
        password: trimmedPassword,
      }),
    });

    const token =
      (typeof tokenData.access_token === "string" && tokenData.access_token) ||
      (typeof tokenData.token === "string" && tokenData.token) ||
      null;

    if (!token) {
      throw new Error("Token missing from login response.");
    }

    setAccessToken(token);
    try {
      const currentUser = await fetchCurrentUser(token);
      setUser(currentUser);
    } catch (error) {
      clearAccessToken();
      throw error;
    }
  };

  const register = async ({ name, email, password }: RegisterInput) => {
    const trimmedName = name.trim();
    const normalizedEmail = normalizeEmail(email);
    const trimmedPassword = password.trim();

    if (!trimmedName || !normalizedEmail || !trimmedPassword) {
      throw new Error("All fields are required.");
    }

    await requestJson(`${AUTH_BASE_URL}/register`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: trimmedName,
        email: normalizedEmail,
        password: trimmedPassword,
      }),
    });

    await login({
      email: normalizedEmail,
      password: trimmedPassword,
    });
  };

  const updateProfile = async ({
    name,
    email,
    currentPassword,
    newPassword,
    confirmPassword,
  }: ProfileUpdateInput) => {
    const token = getAccessToken();

    if (!token) {
      throw new Error("You must be logged in to update your profile.");
    }

    const payload: Record<string, string> = {
      name: name.trim(),
    };

    if (typeof email === "string") {
      payload.email = normalizeEmail(email);
    }

    if (currentPassword?.trim()) {
      payload.current_password = currentPassword.trim();
    }

    if (newPassword?.trim()) {
      payload.new_password = newPassword.trim();
    }

    if (confirmPassword?.trim()) {
      payload.confirm_password = confirmPassword.trim();
    }

    const response = await requestJson<Record<string, unknown>>(`${AUTH_BASE_URL}/users/me`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(payload),
    });

    const nextToken =
      (typeof response.access_token === "string" && response.access_token) || token;

    if (nextToken !== token) {
      setAccessToken(nextToken);
    }

    try {
      const currentUser = await fetchCurrentUser(nextToken);
      setUser(currentUser);
    } catch (error) {
      clearAccessToken();
      setUser(null);
      throw error;
    }
  };

  const logout = async () => {
    const token = getAccessToken();
    if (token) {
      try {
        await requestJson(`${AUTH_BASE_URL}/logout`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
      } catch {
        // Ignore logout request failures and clear local auth state anyway.
      }
    }
    clearAccessToken();
    setUser(null);
  };

  const value = useMemo(
    () => ({
      user,
      isLoading,
      login,
      register,
      updateProfile,
      logout,
    }),
    [user, isLoading]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider.");
  }
  return context;
};
