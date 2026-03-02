import { useEffect, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Cpu, LogOut, Monitor, Moon, Save, Sun, User } from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import { toast } from "sonner";

type ThemePreference = "light" | "dark" | "system";

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

const Profile = () => {
  const navigate = useNavigate();
  const { user, logout, updateProfile } = useAuth();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [model, setModel] = useState(DEFAULT_MODEL_ID);
  const [theme, setTheme] = useState<ThemePreference>("system");
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
      }
    } catch {
      // Ignore invalid local preference payload.
    }
  }, []);

  useEffect(() => {
    if (!user) return;
    setName(user.name);
    setEmail(user.email);
  }, [user]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    localStorage.setItem(
      PROFILE_PREFERENCES_KEY,
      JSON.stringify({
        model,
        theme,
      })
    );
  }, [model, theme]);

  useEffect(() => {
    if (typeof window === "undefined") return;

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
  }, [theme]);

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
        email,
        currentPassword,
        newPassword,
        confirmPassword,
      });

      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      toast.success("Profile updated successfully.");
    } catch (error: any) {
      toast.error(error?.message || "Failed to update profile.");
    } finally {
      setIsSubmitting(false);
    }
  };

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
                  <label htmlFor="profile-email" className="mb-1 block text-sm text-foreground/70">
                    Email
                  </label>
                  <input
                    id="profile-email"
                    type="email"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground transition-colors focus:outline-none focus:ring-2 focus:ring-ring md:max-w-xs"
                    placeholder="you@example.com"
                    required
                  />
                </div>
              </div>
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
                      onClick={() => setTheme(entry.value)}
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

            <section className="mb-8">
              <h2 className="mb-3 text-xs uppercase tracking-wide text-muted-foreground">Change password</h2>
              <div className="space-y-3">
                <div>
                  <label
                    htmlFor="profile-current-password"
                    className="mb-1 block text-sm text-foreground/70"
                  >
                    Current password
                  </label>
                  <input
                    id="profile-current-password"
                    type="password"
                    value={currentPassword}
                    onChange={(event) => setCurrentPassword(event.target.value)}
                    className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground transition-colors focus:outline-none focus:ring-2 focus:ring-ring md:max-w-xs"
                    placeholder="Current password"
                  />
                </div>

                <div>
                  <label htmlFor="profile-new-password" className="mb-1 block text-sm text-foreground/70">
                    New password
                  </label>
                  <input
                    id="profile-new-password"
                    type="password"
                    value={newPassword}
                    onChange={(event) => setNewPassword(event.target.value)}
                    className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground transition-colors focus:outline-none focus:ring-2 focus:ring-ring md:max-w-xs"
                    placeholder="New password"
                  />
                </div>

                <div>
                  <label
                    htmlFor="profile-confirm-password"
                    className="mb-1 block text-sm text-foreground/70"
                  >
                    Confirm new password
                  </label>
                  <input
                    id="profile-confirm-password"
                    type="password"
                    value={confirmPassword}
                    onChange={(event) => setConfirmPassword(event.target.value)}
                    className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground transition-colors focus:outline-none focus:ring-2 focus:ring-ring md:max-w-xs"
                    placeholder="Confirm new password"
                  />
                </div>
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
