import { useEffect, useState, type FormEvent } from "react";
import { X } from "lucide-react";
import { useAuth } from "@/context/useAuth";
import { getErrorMessage } from "@/lib/errors";
import { toast } from "sonner";

type Mode = "login" | "register" | "forgot-password";

interface Props {
  open: boolean;
  mode: Mode;
  onModeChange: (mode: Mode) => void;
  onClose: () => void;
}

const AuthDialog = ({ open, mode, onModeChange, onClose }: Props) => {
  const { login, register, resendVerification, requestPasswordReset } = useAuth();
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
        toast.success("Account created. Verify your email before logging in.");
      } else if (mode === "forgot-password") {
        await requestPasswordReset(email);
        toast.success("If that email exists, a reset link has been sent.");
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

  const submitVerificationResend = async () => {
    setIsSubmitting(true);
    try {
      await resendVerification(email);
      toast.success("If that email can be verified, a new link has been sent.");
    } catch (error: unknown) {
      toast.error(getErrorMessage(error) || "Could not resend verification email.");
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
              {mode === "login"
                ? "Login"
                : mode === "register"
                  ? "Create account"
                  : "Reset password"}
            </h2>
            <p className="text-xs text-muted-foreground">
              {mode === "login"
                ? "Access your existing account."
                : mode === "register"
                  ? "Create your profile and verify your email."
                  : "Request a secure password reset link by email."}
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

          {mode !== "forgot-password" && (
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
          )}

          <button
            type="submit"
            disabled={isSubmitting}
            className="mt-1 w-full rounded-lg bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {mode === "login"
              ? "Login"
              : mode === "register"
                ? "Register"
                : "Send reset link"}
          </button>
        </form>

        {mode === "login" && (
          <div className="mt-3 flex items-center justify-between gap-3 text-xs">
            <button
              type="button"
              onClick={() => onModeChange("forgot-password")}
              className="font-medium text-primary hover:opacity-80"
            >
              Forgot password?
            </button>
            <button
              type="button"
              disabled={isSubmitting}
              onClick={() => void submitVerificationResend()}
              className="font-medium text-primary hover:opacity-80 disabled:opacity-60"
            >
              Resend verification
            </button>
          </div>
        )}

        <div className="mt-4 text-center text-xs text-muted-foreground">
          {mode === "login"
            ? "No account yet?"
            : mode === "register"
              ? "Already have an account?"
              : "Back to login?"}{" "}
          <button
            type="button"
            onClick={() =>
              onModeChange(
                mode === "login" ? "register" : "login"
              )
            }
            className="font-medium text-primary hover:opacity-80"
          >
            {mode === "login"
              ? "Register"
              : mode === "register"
                ? "Login"
                : "Login"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default AuthDialog;
