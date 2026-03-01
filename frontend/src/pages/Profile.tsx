import { useEffect, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, LogOut, Save, UserCircle2 } from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import { toast } from "sonner";

const Profile = () => {
  const navigate = useNavigate();
  const { user, logout, updateProfile } = useAuth();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!user) return;
    setName(user.name);
    setEmail(user.email);
  }, [user]);

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
    <div className="min-h-screen bg-background px-4 py-8">
      <div className="mx-auto w-full max-w-2xl space-y-5">
        <button
          onClick={() => navigate("/")}
          className="inline-flex items-center gap-2 rounded-md px-2 py-1 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          <ArrowLeft size={16} />
          Back to chat
        </button>

        <section className="rounded-2xl border border-border bg-card p-6 shadow-sm">
          {!user ? (
            <div className="space-y-2 text-center">
              <h1 className="text-xl font-semibold text-card-foreground">Profile</h1>
              <p className="text-sm text-muted-foreground">
                You are not authenticated. Please login or register from the sidebar.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="flex items-center gap-3">
                <UserCircle2 size={40} className="text-primary" />
                <div>
                  <h1 className="text-xl font-semibold text-card-foreground">Your Profile</h1>
                  <p className="text-sm text-muted-foreground">
                    Update your account details and password.
                  </p>
                </div>
              </div>

              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-3 rounded-xl border border-input bg-background p-4">
                  <h2 className="text-sm font-medium text-foreground">Account Details</h2>

                  <div className="space-y-1.5">
                    <label htmlFor="profile-name" className="text-xs font-medium text-muted-foreground">
                      Name
                    </label>
                    <input
                      id="profile-name"
                      type="text"
                      value={name}
                      onChange={(event) => setName(event.target.value)}
                      className="w-full rounded-lg border border-input bg-card px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                      placeholder="Jane Doe"
                      required
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label htmlFor="profile-email" className="text-xs font-medium text-muted-foreground">
                      Email
                    </label>
                    <input
                      id="profile-email"
                      type="email"
                      value={email}
                      onChange={(event) => setEmail(event.target.value)}
                      className="w-full rounded-lg border border-input bg-card px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                      placeholder="you@example.com"
                      required
                    />
                  </div>
                </div>

                <div className="space-y-3 rounded-xl border border-input bg-background p-4">
                  <h2 className="text-sm font-medium text-foreground">Change Password (optional)</h2>

                  <div className="space-y-1.5">
                    <label htmlFor="profile-current-password" className="text-xs font-medium text-muted-foreground">
                      Current password
                    </label>
                    <input
                      id="profile-current-password"
                      type="password"
                      value={currentPassword}
                      onChange={(event) => setCurrentPassword(event.target.value)}
                      className="w-full rounded-lg border border-input bg-card px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                      placeholder="Current password"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label htmlFor="profile-new-password" className="text-xs font-medium text-muted-foreground">
                      New password
                    </label>
                    <input
                      id="profile-new-password"
                      type="password"
                      value={newPassword}
                      onChange={(event) => setNewPassword(event.target.value)}
                      className="w-full rounded-lg border border-input bg-card px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                      placeholder="New password"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label htmlFor="profile-confirm-password" className="text-xs font-medium text-muted-foreground">
                      Confirm new password
                    </label>
                    <input
                      id="profile-confirm-password"
                      type="password"
                      value={confirmPassword}
                      onChange={(event) => setConfirmPassword(event.target.value)}
                      className="w-full rounded-lg border border-input bg-card px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                      placeholder="Confirm new password"
                    />
                  </div>
                </div>

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
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default Profile;
