const ACCESS_TOKEN_STORAGE_KEY = "point-source-access-token";

export const getAccessToken = () => {
  try {
    return localStorage.getItem(ACCESS_TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
};

export const setAccessToken = (token: string) => {
  localStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, token);
};

export const clearAccessToken = () => {
  localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
};
