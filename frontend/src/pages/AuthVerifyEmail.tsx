import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "@/context/useAuth";
import { getErrorMessage } from "@/lib/errors";

type Status = "idle" | "loading" | "success" | "error";

const AuthVerifyEmail = () => {
  const { verifyEmail } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [token, setToken] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState(
    "Preparing your verification link..."
  );
  const submittedTokenRef = useRef<string | null>(null);

  const submitVerification = async (nextToken: string) => {
    setStatus("loading");
    setMessage("Verifying your email...");

    try {
      const responseMessage = await verifyEmail(nextToken);
      setStatus("success");
      setMessage(responseMessage || "Email verified. Redirecting...");
      navigate("/", { replace: true });
    } catch (error: unknown) {
      setStatus("error");
      setMessage(getErrorMessage(error) || "Could not verify this email link.");
    }
  };

  useEffect(() => {
    const nextToken = searchParams.get("token")?.trim() || "";

    if (!nextToken) {
      setToken("");
      setStatus("error");
      setMessage("The verification link is missing a token.");
      return;
    }

    setToken(nextToken);
    if (submittedTokenRef.current === nextToken) {
      return;
    }

    submittedTokenRef.current = nextToken;
    void submitVerification(nextToken);
  }, [navigate, searchParams, verifyEmail]);

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
              : status === "error"
                ? "Verification failed"
                : "Verify your email"}
        </h1>
        <p className="mt-4 text-sm leading-6 text-muted-foreground">{message}</p>
        <div className="mt-8 flex flex-wrap gap-3">
          {status !== "success" && token ? (
            <button
              type="button"
              onClick={() => void submitVerification(token)}
              disabled={status === "loading"}
              className="rounded-full bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {status === "loading" ? "Verifying..." : "Try again"}
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
