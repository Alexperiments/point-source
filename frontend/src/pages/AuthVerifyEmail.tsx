import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useAuth } from "@/context/useAuth";
import { getErrorMessage } from "@/lib/errors";

type Status = "idle" | "loading" | "success" | "error";

const AuthVerifyEmail = () => {
  const { verifyEmail } = useAuth();
  const [searchParams] = useSearchParams();
  const [token, setToken] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState(
    "Confirm your email address to activate your Point-source account."
  );

  useEffect(() => {
    const nextToken = searchParams.get("token")?.trim() || "";

    if (!nextToken) {
      setStatus("error");
      setMessage("The verification link is missing a token.");
      return;
    }
    setToken(nextToken);
    setStatus("idle");
    setMessage("Confirm your email address to activate your Point-source account.");
  }, [searchParams]);

  const submitVerification = async () => {
    if (!token) {
      setStatus("error");
      setMessage("The verification link is missing a token.");
      return;
    }

    setStatus("loading");
    setMessage("Verifying your email...");

    try {
      const responseMessage = await verifyEmail(token);
      setStatus("success");
      setMessage(responseMessage);
    } catch (error: unknown) {
      setStatus("error");
      setMessage(getErrorMessage(error) || "Could not verify this email link.");
    }
  };

  return (
    <main className="min-h-screen bg-background px-4 py-16 text-foreground">
      <div className="mx-auto max-w-lg rounded-3xl border border-border bg-card p-8 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          Point-source
        </p>
        <h1 className="mt-3 text-3xl font-semibold">
          {status === "loading"
            ? "Verifying email"
            : status === "success"
              ? "Email verified"
              : "Verification failed"}
        </h1>
        <p className="mt-4 text-sm leading-6 text-muted-foreground">{message}</p>
        <div className="mt-8 flex flex-wrap gap-3">
          {status !== "success" && token ? (
            <button
              type="button"
              onClick={() => void submitVerification()}
              disabled={status === "loading"}
              className="rounded-full bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {status === "loading" ? "Verifying..." : "Verify email"}
            </button>
          ) : (
            <Link
              to="/"
              className="rounded-full bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90"
            >
              Return home
            </Link>
          )}
          {status !== "loading" && (
            <Link
              to="/auth/reset-password"
              className="rounded-full border border-border px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"
            >
              Need a password reset?
            </Link>
          )}
        </div>
      </div>
    </main>
  );
};

export default AuthVerifyEmail;
