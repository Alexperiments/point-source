import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useAuth } from "@/context/useAuth";
import { getErrorMessage } from "@/lib/errors";

type Status = "loading" | "success" | "error";

const AuthVerifyEmail = () => {
  const { verifyEmail } = useAuth();
  const [searchParams] = useSearchParams();
  const [status, setStatus] = useState<Status>("loading");
  const [message, setMessage] = useState("Verifying your email...");

  useEffect(() => {
    const token = searchParams.get("token")?.trim() || "";

    if (!token) {
      setStatus("error");
      setMessage("The verification link is missing a token.");
      return;
    }

    let cancelled = false;

    const run = async () => {
      try {
        const responseMessage = await verifyEmail(token);
        if (cancelled) return;
        setStatus("success");
        setMessage(responseMessage);
      } catch (error: unknown) {
        if (cancelled) return;
        setStatus("error");
        setMessage(getErrorMessage(error) || "Could not verify this email link.");
      }
    };

    void run();

    return () => {
      cancelled = true;
    };
  }, [searchParams, verifyEmail]);

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
          <Link
            to="/"
            className="rounded-full bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90"
          >
            Return home
          </Link>
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
