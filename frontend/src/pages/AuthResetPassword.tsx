import { useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useAuth } from "@/context/useAuth";
import { getErrorMessage } from "@/lib/errors";

const AuthResetPassword = () => {
  const { requestPasswordReset, resetPassword } = useAuth();
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token")?.trim() || "";

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submitResetRequest = async (e: FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setMessage(null);

    try {
      await requestPasswordReset(email);
      setMessage("If that email exists, a password reset link has been sent.");
    } catch (submitError: unknown) {
      setError(getErrorMessage(submitError) || "Could not send the reset link.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const submitPasswordReset = async (e: FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setMessage(null);

    try {
      const responseMessage = await resetPassword(token, password, confirmPassword);
      setMessage(responseMessage);
      setPassword("");
      setConfirmPassword("");
    } catch (submitError: unknown) {
      setError(getErrorMessage(submitError) || "Could not reset the password.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="min-h-screen bg-background px-4 py-16 text-foreground">
      <div className="mx-auto max-w-lg rounded-3xl border border-border bg-card p-8 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          Point-source
        </p>
        <h1 className="mt-3 text-3xl font-semibold">
          {token ? "Choose a new password" : "Reset your password"}
        </h1>
        <p className="mt-4 text-sm leading-6 text-muted-foreground">
          {token
            ? "Enter a new password for your account. The email link can only be used once."
            : "Enter your email address and we will send a password reset link if the account exists."}
        </p>

        <form
          onSubmit={token ? submitPasswordReset : submitResetRequest}
          className="mt-8 space-y-4"
        >
          {!token && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground" htmlFor="reset-email">
                Email
              </label>
              <input
                id="reset-email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-xl border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                placeholder="ada.lovelace@example.com"
                required
              />
            </div>
          )}

          {token && (
            <>
              <div className="space-y-1.5">
                <label
                  className="text-xs font-medium text-muted-foreground"
                  htmlFor="reset-password"
                >
                  New password
                </label>
                <input
                  id="reset-password"
                  type="password"
                  autoComplete="new-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full rounded-xl border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  placeholder="••••••••"
                  required
                />
              </div>

              <div className="space-y-1.5">
                <label
                  className="text-xs font-medium text-muted-foreground"
                  htmlFor="reset-confirm-password"
                >
                  Confirm password
                </label>
                <input
                  id="reset-confirm-password"
                  type="password"
                  autoComplete="new-password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full rounded-xl border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  placeholder="••••••••"
                  required
                />
              </div>
            </>
          )}

          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full rounded-full bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {token ? "Update password" : "Send reset link"}
          </button>
        </form>

        {message && <p className="mt-4 text-sm text-foreground">{message}</p>}
        {error && <p className="mt-4 text-sm text-destructive">{error}</p>}

        <div className="mt-8">
          <Link
            to="/"
            className="text-sm font-medium text-primary transition-opacity hover:opacity-80"
          >
            Return home
          </Link>
        </div>
      </div>
    </main>
  );
};

export default AuthResetPassword;
