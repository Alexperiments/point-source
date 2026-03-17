import { createContext } from "react";

export type AuthUser = {
  id: string;
  name: string;
  email: string;
};

type LoginInput = {
  email: string;
  password: string;
};

type RegisterInput = {
  name: string;
  email: string;
  password: string;
};

type ProfileUpdateInput = {
  name: string;
  email?: string;
  currentPassword?: string;
  newPassword?: string;
  confirmPassword?: string;
};

export type AuthContextValue = {
  user: AuthUser | null;
  isLoading: boolean;
  login: (input: LoginInput) => Promise<void>;
  register: (input: RegisterInput) => Promise<void>;
  updateProfile: (input: ProfileUpdateInput) => Promise<void>;
  logout: () => Promise<void>;
};

export const AuthContext = createContext<AuthContextValue | undefined>(undefined);
