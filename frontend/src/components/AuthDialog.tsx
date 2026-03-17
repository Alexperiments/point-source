import { useEffect, useState, type FormEvent } from "react";
import { X } from "lucide-react";
import { useAuth } from "@/context/useAuth";
import { getErrorMessage } from "@/lib/errors";
import { toast } from "sonner";

type Mode = "login" | "register";

interface Props {
  open: boolean;
  mode: Mode;
  onModeChange: (mode: Mode) => void;
  onClose: () => void;
}

const AuthDialog = ({ open, mode, onModeChange, onClose }: Props) => {
  const { login, register } = useAuth();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!open) return;
    setName("");
    setEmail("");
    setPassword("");
    setIsSubmitting(false);
  }, [open, mode]);

  if (!open) return null;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      if (mode === "register") {
        await register({ name, email, password });
        toast.success("Account created. You are now logged in.");
      } else {
        await login({ email, password });
        toast.success("Logged in successfully.");
      }

      onClose();
    } catch (error: unknown) {
      toast.error(getErrorMessage(error) || "Authentication failed.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/30 p-4">
      <div className="w-full max-w-sm rounded-2xl border border-border bg-card p-5 shadow-xl">
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold text-card-foreground">
              {mode === "login" ? "Login" : "Create account"}
            </h2>
            <p className="text-xs text-muted-foreground">
              {mode === "login"
                ? "Access your existing account."
                : "Create your profile to get started."}
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent"
            aria-label="Close authentication dialog"
          >
            <X size={16} />
          </button>
        </div>

        <form onSubmit={submit} className="space-y-3">
          {mode === "register" && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground" htmlFor="auth-name">
                Name
              </label>
              <input
                id="auth-name"
                type="text"
                autoComplete="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                placeholder="Ada Lovelace"
                required
              />
            </div>
          )}

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground" htmlFor="auth-email">
              Email
            </label>
            <input
              id="auth-email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              placeholder="ada.lovelace@example.com"
              required
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground" htmlFor="auth-password">
              Password
            </label>
            <input
              id="auth-password"
              type="password"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              placeholder="••••••••"
              required
            />
          </div>

          <button
            type="submit"
            disabled={isSubmitting}
            className="mt-1 w-full rounded-lg bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {mode === "login" ? "Login" : "Register"}
          </button>
        </form>

        <div className="mt-4 text-center text-xs text-muted-foreground">
          {mode === "login" ? "No account yet?" : "Already have an account?"}{" "}
          <button
            type="button"
            onClick={() => onModeChange(mode === "login" ? "register" : "login")}
            className="font-medium text-primary hover:opacity-80"
          >
            {mode === "login" ? "Register" : "Login"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default AuthDialog;
