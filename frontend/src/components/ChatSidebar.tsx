import { useEffect, useRef, useState } from "react";
import {
  ChevronDown,
  CircleHelp,
  LogIn,
  LogOut,
  MessageSquare,
  MoreHorizontal,
  Plus,
  Telescope,
  Trash2,
  User,
  UserPlus,
  X,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import type { Conversation } from "@/pages/Index";
import { useAuth } from "@/context/AuthContext";
import AuthDialog from "@/components/AuthDialog";
import { toast } from "sonner";

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  loginPromptVersion: number;
  open: boolean;
  onClose: () => void;
}

const ChatSidebar = ({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  loginPromptVersion,
  open,
  onClose,
}: Props) => {
  const navigate = useNavigate();
  const { user, logout, isLoading } = useAuth();
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authDialogOpen, setAuthDialogOpen] = useState(false);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const [conversationMenuId, setConversationMenuId] = useState<string | null>(null);
  const profileMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!profileMenuOpen) return;

    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (!profileMenuRef.current?.contains(target)) {
        setProfileMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [profileMenuOpen]);

  useEffect(() => {
    if (!conversationMenuId) return;

    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (!target.closest("[data-conversation-menu]")) {
        setConversationMenuId(null);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [conversationMenuId]);

  useEffect(() => {
    if (loginPromptVersion === 0) return;
    setAuthMode("login");
    setAuthDialogOpen(true);
  }, [loginPromptVersion]);

  const openAuthDialog = (mode: "login" | "register") => {
    setAuthMode(mode);
    setAuthDialogOpen(true);
  };

  const userInitials = user?.name
    .split(" ")
    .filter(Boolean)
    .map((segment) => segment[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  const handleLogout = async () => {
    await logout();
    setProfileMenuOpen(false);
    toast.success("You are now logged out.");
  };

  const openProfile = () => {
    setProfileMenuOpen(false);
    navigate("/profile");
    if (window.innerWidth < 768) {
      onClose();
    }
  };

  const openAbout = () => {
    setProfileMenuOpen(false);
    navigate("/about");
    if (window.innerWidth < 768) {
      onClose();
    }
  };

  return (
    <>
      {/* Overlay for mobile */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/35 md:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`
          fixed inset-y-0 left-0 z-50 flex h-full flex-col
          bg-sidebar-background border-r border-sidebar-border
          shadow-xl transition-transform duration-200 ease-in-out md:relative md:shadow-none
          ${open ? "w-64 translate-x-0" : "w-64 -translate-x-full md:translate-x-0"}
        `}
      >
        {!open ? null : (
          <div
            className="flex h-full w-64 max-w-[85vw] flex-col"
            style={{
              paddingTop: "env(safe-area-inset-top)",
              paddingBottom: "env(safe-area-inset-bottom)",
            }}
          >
            <div className="border-b border-sidebar-border px-3 py-3">
              <div className="flex items-center gap-2">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-accent/60 ring-1 ring-primary/10">
                  <Telescope size={15} className="text-primary/70" />
                </div>
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-sidebar-foreground">
                    Point-source
                  </p>
                  <p className="truncate text-[11px] text-muted-foreground">
                    Astrophysics RAG
                  </p>
                </div>
              </div>
            </div>

            {/* Header */}
            <div className="flex items-center justify-between p-3">
              <button
                onClick={() => {
                  setConversationMenuId(null);
                  onNew();
                }}
                className="flex flex-1 items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-sidebar-foreground hover:bg-sidebar-accent transition-colors"
              >
                <Plus size={16} />
                New chat
              </button>
              <button
                onClick={onClose}
                className="rounded-md p-1.5 text-sidebar-foreground hover:bg-sidebar-accent transition-colors md:hidden"
              >
                <X size={16} />
              </button>
            </div>

            {/* Conversation list */}
            <div className="flex-1 overflow-y-auto px-2 pb-4">
              {conversations.length === 0 && (
                <p className="px-3 py-8 text-center text-xs text-muted-foreground">
                  No conversations yet
                </p>
              )}
              {conversations.map((conv) => (
                <div key={conv.id} className="relative">
                  <div
                    className={`
                      group flex items-center gap-2 rounded-lg px-3 py-2 mb-0.5 cursor-pointer text-sm transition-colors
                      ${conv.id === activeId
                        ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                        : "text-sidebar-foreground hover:bg-sidebar-accent/60"
                      }
                    `}
                    onClick={() => {
                      setConversationMenuId(null);
                      onSelect(conv.id);
                    }}
                  >
                    <MessageSquare size={14} className="shrink-0 opacity-50" />
                    <span className="flex-1 truncate">{conv.title}</span>
                    <button
                      data-conversation-menu
                      onClick={(e) => {
                        e.stopPropagation();
                        setConversationMenuId((prev) => (prev === conv.id ? null : conv.id));
                      }}
                      className="rounded p-1 opacity-100 transition-all hover:bg-accent md:opacity-0 md:group-hover:opacity-100"
                      aria-label="Open chat options"
                    >
                      <MoreHorizontal size={13} />
                    </button>
                  </div>

                  {conversationMenuId === conv.id && (
                    <div
                      data-conversation-menu
                      className="absolute right-2 top-9 z-20 min-w-32 rounded-lg border border-border bg-popover p-1 shadow-lg"
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setConversationMenuId(null);
                          void onDelete(conv.id);
                        }}
                        className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs text-destructive transition-colors hover:bg-destructive/10"
                      >
                        <Trash2 size={13} />
                        Delete chat
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="border-t border-sidebar-border p-3">
              {isLoading ? (
                <p className="px-1 py-2 text-center text-xs text-muted-foreground">
                  Loading account...
                </p>
              ) : !user ? (
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => openAuthDialog("login")}
                    className="flex items-center justify-center gap-1 rounded-lg border border-input bg-background px-2 py-2 text-xs font-medium text-foreground transition-colors hover:bg-accent"
                  >
                    <LogIn size={13} />
                    Login
                  </button>
                  <button
                    onClick={() => openAuthDialog("register")}
                    className="flex items-center justify-center gap-1 rounded-lg bg-primary px-2 py-2 text-xs font-medium text-primary-foreground transition-opacity hover:opacity-90"
                  >
                    <UserPlus size={13} />
                    Register
                  </button>
                </div>
              ) : (
                <div className="relative" ref={profileMenuRef}>
                  <button
                    onClick={() => setProfileMenuOpen((prev) => !prev)}
                    className="flex w-full items-center gap-2 rounded-lg border border-input bg-background px-2 py-2 text-left transition-colors hover:bg-accent"
                  >
                    <div className="flex h-7 w-7 items-center justify-center rounded-full bg-primary/15 text-xs font-semibold text-primary">
                      {userInitials || "U"}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-medium text-foreground">{user.name}</p>
                      <p className="truncate text-[11px] text-muted-foreground">{user.email}</p>
                    </div>
                    <ChevronDown
                      size={14}
                      className={`text-muted-foreground transition-transform ${profileMenuOpen ? "rotate-180" : ""}`}
                    />
                  </button>

                  {profileMenuOpen && (
                    <div className="absolute bottom-full left-0 right-0 z-20 mb-2 rounded-lg border border-border bg-popover p-1 shadow-lg">
                      <button
                        onClick={openProfile}
                        className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs text-popover-foreground transition-colors hover:bg-accent"
                      >
                        <User size={13} />
                        Profile
                      </button>
                      <button
                        onClick={openAbout}
                        className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs text-popover-foreground transition-colors hover:bg-accent"
                      >
                        <CircleHelp size={13} />
                        About this project
                      </button>
                      <button
                        onClick={handleLogout}
                        className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs text-destructive transition-colors hover:bg-destructive/10"
                      >
                        <LogOut size={13} />
                        Logout
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </aside>

      <AuthDialog
        open={authDialogOpen}
        mode={authMode}
        onModeChange={setAuthMode}
        onClose={() => setAuthDialogOpen(false)}
      />
    </>
  );
};

export default ChatSidebar;
