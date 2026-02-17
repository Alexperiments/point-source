import { useNavigate } from "react-router-dom";
import { ArrowLeft, LogOut, UserCircle2 } from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import { toast } from "sonner";

const Profile = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    toast.success("You are now logged out.");
    navigate("/");
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
            <div className="space-y-5">
              <div className="flex items-center gap-3">
                <UserCircle2 size={40} className="text-primary" />
                <div>
                  <h1 className="text-xl font-semibold text-card-foreground">Your Profile</h1>
                  <p className="text-sm text-muted-foreground">
                    Frontend-only account details
                  </p>
                </div>
              </div>

              <div className="space-y-3 rounded-xl border border-input bg-background p-4">
                <div>
                  <p className="text-xs text-muted-foreground">Name</p>
                  <p className="text-sm font-medium text-foreground">{user.name}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Email</p>
                  <p className="text-sm font-medium text-foreground">{user.email}</p>
                </div>
              </div>

              <button
                onClick={handleLogout}
                className="inline-flex items-center gap-2 rounded-lg bg-destructive px-3 py-2 text-sm font-medium text-destructive-foreground transition-opacity hover:opacity-90"
              >
                <LogOut size={14} />
                Logout
              </button>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default Profile;
