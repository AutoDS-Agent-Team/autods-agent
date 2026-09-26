import { API_BASE_URL, apiFetch, readApiError } from "./client.js";
async function submit(path, payload) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/auth/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await readApiError(response));
  return response.json();
}
export const register = (email, password) =>
  submit("register", { email, password });
export const login = (email, password) => submit("login", { email, password });
export const loginWithGoogleCredential = (credential) =>
  submit("google", { credential });
export async function getCurrentUser() {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/auth/me`);
  if (!response.ok) throw new Error(await readApiError(response));
  return response.json();
}
