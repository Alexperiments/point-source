import { useEffect, useState, type ReactNode } from "react";
import { AUTH_BASE_URL } from "@/lib/api";
import {
  AuthContext,
  type AuthUser,
  type LoginInput,
  type ProfileUpdateInput,
  type RegisterInput,
} from "@/context/auth-context";

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
  const emailVerified = raw.email_verified;
  const name =
    raw.name ||
    raw.full_name ||
    raw.username ||
    raw.display_name;

  if (
    typeof id !== "string" ||
    typeof email !== "string" ||
    typeof name !== "string" ||
    typeof emailVerified !== "boolean"
  ) {
    throw new Error("Unexpected user payload from server.");
  }

  return { id, email, name, emailVerified };
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

    await requestJson<Record<string, unknown>>(`${AUTH_BASE_URL}/register`, {
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
  };

  const resendVerification = async (email: string) => {
    const normalizedEmail = normalizeEmail(email);

    if (!normalizedEmail) {
      throw new Error("Email is required.");
    }

    await requestJson<Record<string, unknown>>(`${AUTH_BASE_URL}/email/verify/request`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email: normalizedEmail,
      }),
    });
  };

  const requestPasswordReset = async (email: string) => {
    const normalizedEmail = normalizeEmail(email);

    if (!normalizedEmail) {
      throw new Error("Email is required.");
    }

    await requestJson<Record<string, unknown>>(`${AUTH_BASE_URL}/password-reset/request`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email: normalizedEmail,
      }),
    });
  };

  const verifyEmail = async (token: string) => {
    const trimmedToken = token.trim();

    if (!trimmedToken) {
      throw new Error("Verification token is required.");
    }

    const response = await requestJson<{ message?: string }>(`${AUTH_BASE_URL}/email/verify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        token: trimmedToken,
      }),
    });

    return response.message || "Email verified.";
  };

  const resetPassword = async (
    token: string,
    newPassword: string,
    confirmPassword: string,
  ) => {
    const trimmedToken = token.trim();
    const trimmedPassword = newPassword.trim();
    const trimmedConfirmPassword = confirmPassword.trim();

    if (!trimmedToken) {
      throw new Error("Reset token is required.");
    }

    if (!trimmedPassword || !trimmedConfirmPassword) {
      throw new Error("Both password fields are required.");
    }

    const response = await requestJson<{ message?: string }>(
      `${AUTH_BASE_URL}/password-reset/confirm`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          token: trimmedToken,
          new_password: trimmedPassword,
          confirm_password: trimmedConfirmPassword,
        }),
      }
    );

    setUser(null);
    return response.message || "Password updated.";
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
        resendVerification,
        requestPasswordReset,
        verifyEmail,
        resetPassword,
        updateProfile,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
