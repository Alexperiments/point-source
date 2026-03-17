import { useEffect, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Cpu, Gauge, LogOut, Monitor, Moon, Save, Sun, User } from "lucide-react";
import { AUTH_BASE_URL } from "@/lib/api";
import { useAuth } from "@/context/useAuth";
import { getErrorMessage } from "@/lib/errors";
import { toast } from "sonner";

type ThemePreference = "light" | "dark" | "system";

type UsageSummary = {
  isPremium: boolean;
  dailyMessageLimit: number | null;
  requestsUsed: number;
  requestsRemaining: number | null;
  resetAt: string;
  resetInSeconds: number;
};

const PROFILE_PREFERENCES_KEY = "point-source.profile-preferences";
const DEFAULT_MODEL_ID = "openai/gpt-5-mini";

const models = [
  { id: DEFAULT_MODEL_ID, label: "GPT-5 Mini", desc: "Strong all-rounded" },
];

const themes: { value: ThemePreference; label: string; icon: typeof Sun }[] = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
];

const isThemePreference = (value: unknown): value is ThemePreference =>
  value === "light" || value === "dark" || value === "system";

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

const parseUsageSummary = (payload: unknown): UsageSummary => {
  if (!payload || typeof payload !== "object") {
    throw new Error("Invalid usage payload from server.");
  }

  const raw = payload as Record<string, unknown>;
  const isPremium = raw.is_premium;
  const dailyMessageLimit = raw.daily_message_limit;
  const requestsUsed = raw.requests_used;
  const requestsRemaining = raw.requests_remaining;
  const resetAt = raw.reset_at;
  const resetInSeconds = raw.reset_in_seconds;

  if (
    typeof isPremium !== "boolean" ||
    (dailyMessageLimit !== null && typeof dailyMessageLimit !== "number") ||
    typeof requestsUsed !== "number" ||
    (requestsRemaining !== null && typeof requestsRemaining !== "number") ||
    typeof resetAt !== "string" ||
    typeof resetInSeconds !== "number"
  ) {
    throw new Error("Unexpected usage payload from server.");
  }

  return {
    isPremium,
    dailyMessageLimit,
    requestsUsed,
    requestsRemaining,
    resetAt,
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
    throw new Error(
      parseErrorMessage(payload) || `Request failed with status ${response.status}.`
    );
  }

  return parseUsageSummary(payload);
};

const formatResetTimer = (totalSeconds: number) => {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  return [hours, minutes, seconds].map((value) => value.toString().padStart(2, "0")).join(":");
};

const Profile = () => {
  const navigate = useNavigate();
  const { user, logout, updateProfile } = useAuth();

  const [name, setName] = useState("");
  const [model, setModel] = useState(DEFAULT_MODEL_ID);
  const [theme, setTheme] = useState<ThemePreference>("system");
  const [preferencesLoaded, setPreferencesLoaded] = useState(false);
  const [shouldApplyTheme, setShouldApplyTheme] = useState(false);
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [usageError, setUsageError] = useState<string | null>(null);
  const [isLoadingUsage, setIsLoadingUsage] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;

    try {
      const raw = localStorage.getItem(PROFILE_PREFERENCES_KEY);
      if (!raw) return;

      const parsed = JSON.parse(raw) as {
        model?: string;
        theme?: ThemePreference;
      };

      if (parsed.model === DEFAULT_MODEL_ID) {
        setModel(parsed.model);
      }

      if (isThemePreference(parsed.theme)) {
        setTheme(parsed.theme);
        setShouldApplyTheme(true);
      }
    } catch {
      // Ignore invalid local preference payload.
    } finally {
      setPreferencesLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (!user) return;
    setName(user.name);
  }, [user]);

  useEffect(() => {
    if (typeof window === "undefined" || !preferencesLoaded) return;

    const nextPreferences: {
      model: string;
      theme?: ThemePreference;
    } = { model };

    if (shouldApplyTheme) {
      nextPreferences.theme = theme;
    }

    localStorage.setItem(PROFILE_PREFERENCES_KEY, JSON.stringify(nextPreferences));
  }, [model, theme, preferencesLoaded, shouldApplyTheme]);

  useEffect(() => {
    if (typeof window === "undefined" || !shouldApplyTheme) return;

    const root = document.documentElement;
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

    const applyTheme = () => {
      const resolvedTheme =
        theme === "system" ? (mediaQuery.matches ? "dark" : "light") : theme;
      root.classList.toggle("dark", resolvedTheme === "dark");
    };

    applyTheme();

    if (theme !== "system") {
      return;
    }

    mediaQuery.addEventListener("change", applyTheme);
    return () => mediaQuery.removeEventListener("change", applyTheme);
  }, [theme, shouldApplyTheme]);

  useEffect(() => {
    if (!user) {
      setUsage(null);
      setUsageError(null);
      setIsLoadingUsage(false);
      return;
    }

    let cancelled = false;
    setIsLoadingUsage(true);
    setUsageError(null);

    const loadUsage = async () => {
      try {
        const nextUsage = await fetchUsageSummary();
        if (!cancelled) {
          setUsage(nextUsage);
        }
      } catch (error: unknown) {
        if (!cancelled) {
          setUsage(null);
          setUsageError(getErrorMessage(error) || "Failed to load usage.");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingUsage(false);
        }
      }
    };

    void loadUsage();

    return () => {
      cancelled = true;
    };
  }, [user]);

  useEffect(() => {
    if (!usage || typeof window === "undefined") return;

    const interval = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);

    return () => window.clearInterval(interval);
  }, [usage]);

  const resetAtTimestamp = usage ? Date.parse(usage.resetAt) : Number.NaN;
  const secondsUntilReset =
    usage === null
      ? null
      : Number.isFinite(resetAtTimestamp)
        ? Math.max(0, Math.ceil((resetAtTimestamp - now) / 1000))
        : usage.resetInSeconds;

  useEffect(() => {
    if (!user || !usage || secondsUntilReset !== 0) return;

    let cancelled = false;

    const refreshUsage = async () => {
      try {
        const nextUsage = await fetchUsageSummary();
        if (!cancelled) {
          setUsage(nextUsage);
          setUsageError(null);
        }
      } catch (error: unknown) {
        if (!cancelled) {
          setUsageError(getErrorMessage(error) || "Failed to load usage.");
        }
      }
    };

    void refreshUsage();

    return () => {
      cancelled = true;
    };
  }, [secondsUntilReset, usage, user]);

  const handleLogout = async () => {
    await logout();
    toast.success("You are now logged out.");
    navigate("/");
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!user) {
      return;
    }

    setIsSubmitting(true);

    try {
      await updateProfile({
        name,
      });
      toast.success("Profile updated successfully.");
    } catch (error: unknown) {
      toast.error(getErrorMessage(error) || "Failed to update profile.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const usageCards = [
    {
      label: "Rate limit",
      value: isLoadingUsage
        ? "..."
        : usage?.isPremium
          ? "∞"
          : usage?.dailyMessageLimit?.toString() || "--",
    },
    {
      label: "Requests today",
      value: isLoadingUsage ? "..." : usage ? usage.requestsUsed.toString() : "--",
    },
    {
      label: "Reset in",
      value:
        isLoadingUsage || secondsUntilReset === null
          ? "..."
          : formatResetTimer(secondsUntilReset),
    },
  ];

  return (
    <div className="min-h-screen bg-background starfield">
      <div className="mx-auto max-w-xl px-6 py-12">
        <button
          onClick={() => navigate("/")}
          className="mb-8 flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft size={16} /> Back to chat
        </button>

        {!user ? (
          <section className="space-y-2">
            <h1 className="text-2xl font-semibold text-foreground">Profile &amp; Settings</h1>
            <p className="text-sm text-muted-foreground">
              You are not authenticated. Please login or register from the sidebar.
            </p>
          </section>
        ) : (
          <form onSubmit={handleSubmit}>
            <h1 className="mb-8 text-2xl font-semibold text-foreground">Profile &amp; Settings</h1>

            <section className="mb-8">
              <h2 className="mb-3 flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
                <User size={14} /> Identity
              </h2>
              <div className="space-y-3">
                <div>
                  <label htmlFor="profile-name" className="mb-1 block text-sm text-foreground/70">
                    Display name
                  </label>
                  <input
                    id="profile-name"
                    type="text"
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground transition-colors focus:outline-none focus:ring-2 focus:ring-ring md:max-w-xs"
                    placeholder="Jane Doe"
                    required
                  />
                </div>

                <div>
                  <p className="mb-1 block text-sm text-foreground/70">Email</p>
                  <div className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground/80 md:max-w-xs">
                    {user.email}
                  </div>
                </div>
              </div>
            </section>

            <section className="mb-8">
              <h2 className="mb-3 flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
                <Gauge size={14} /> Usage
              </h2>
              <div className="grid gap-3 sm:grid-cols-3">
                {usageCards.map((card) => (
                  <div
                    key={card.label}
                    className="rounded-lg border border-input bg-background px-3 py-3"
                  >
                    <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                      {card.label}
                    </p>
                    <p className="mt-2 text-lg font-medium text-foreground">{card.value}</p>
                  </div>
                ))}
              </div>
              {usageError ? (
                <p className="mt-3 text-sm text-destructive">{usageError}</p>
              ) : usage ? (
                <p className="mt-3 text-xs text-muted-foreground">
                  {usage.isPremium
                    ? "Premium accounts have unlimited daily requests. Usage still resets every day at 00:00 UTC."
                    : `${usage.requestsRemaining ?? 0} requests remaining before the reset at 00:00 UTC.`}
                </p>
              ) : null}
            </section>

            <section className="mb-8">
              <h2 className="mb-3 flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
                <Cpu size={14} /> AI model{" "}
                <span className="text-[11px] normal-case tracking-normal text-muted-foreground">
                  (Default only, other models are coming soon)
                </span>
              </h2>
              <div className="space-y-1.5">
                {models.map((entry) => (
                  <button
                    key={entry.id}
                    type="button"
                    onClick={() => setModel(entry.id)}
                    className={`flex w-full items-center justify-between rounded-lg border px-3 py-2.5 text-sm transition-colors ${
                      model === entry.id
                        ? "border-primary/40 bg-accent/60 text-foreground"
                        : "border-transparent text-foreground/70 hover:bg-accent/30"
                    }`}
                  >
                    <span className="font-medium">{entry.label}</span>
                    <span className="text-xs text-muted-foreground">{entry.desc}</span>
                  </button>
                ))}
              </div>
            </section>

            <section className="mb-8">
              <h2 className="mb-3 flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
                <Sun size={14} /> Appearance
              </h2>
              <div className="flex flex-wrap gap-2">
                {themes.map((entry) => {
                  const Icon = entry.icon;
                  return (
                    <button
                      key={entry.value}
                      type="button"
                      onClick={() => {
                        setTheme(entry.value);
                        setShouldApplyTheme(true);
                      }}
                      className={`flex items-center gap-1.5 rounded-lg border px-4 py-2 text-sm transition-colors ${
                        theme === entry.value
                          ? "border-primary/40 bg-accent/60 text-foreground"
                          : "border-transparent text-foreground/70 hover:bg-accent/30"
                      }`}
                    >
                      <Icon size={15} />
                      {entry.label}
                    </button>
                  );
                })}
              </div>
            </section>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="submit"
                disabled={isSubmitting}
                className="inline-flex items-center gap-2 rounded-lg bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Save size={14} />
                {isSubmitting ? "Saving..." : "Save changes"}
              </button>

              <button
                type="button"
                onClick={handleLogout}
                className="inline-flex items-center gap-2 rounded-lg bg-destructive px-3 py-2 text-sm font-medium text-destructive-foreground transition-opacity hover:opacity-90"
              >
                <LogOut size={14} />
                Logout
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

export default Profile;
