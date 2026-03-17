import { useEffect, useState, type ReactNode } from "react";
import { AUTH_BASE_URL } from "@/lib/api";
import { AuthContext, type AuthUser } from "@/context/auth-context";

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
  const response = await fetch(url, {
    ...init,
    credentials: init.credentials ?? "include",
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

const fetchCurrentUser = async () => {
  const user = await requestJson<unknown>(`${AUTH_BASE_URL}/users/me`, {
    method: "GET",
  });
  return parseUser(user);
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    const loadSession = async () => {
      try {
        const currentUser = await fetchCurrentUser();
        if (mounted) setUser(currentUser);
      } catch {
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

    await requestJson<Record<string, unknown>>(`${AUTH_BASE_URL}/token`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email: normalizedEmail,
        password: trimmedPassword,
      }),
    });

    try {
      const currentUser = await fetchCurrentUser();
      setUser(currentUser);
    } catch (error) {
      setUser(null);
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

    await requestJson<Record<string, unknown>>(`${AUTH_BASE_URL}/users/me`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    try {
      const currentUser = await fetchCurrentUser();
      setUser(currentUser);
    } catch (error) {
      setUser(null);
      throw error;
    }
  };

  const logout = async () => {
    try {
      await requestJson(`${AUTH_BASE_URL}/logout`, {
        method: "POST",
      });
    } catch {
      // Ignore logout request failures and clear local auth state anyway.
    }
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        login,
        register,
        updateProfile,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
