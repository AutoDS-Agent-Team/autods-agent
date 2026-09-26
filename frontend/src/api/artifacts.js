import { API_BASE_URL, apiFetch, readApiError } from "./client.js";

/** Fetch a protected artifact with the normal JWT-bearing API client. */
export async function downloadProtectedArtifact(downloadUrl, filename) {
  const response = await apiFetch(`${API_BASE_URL}${downloadUrl}`);
  if (!response.ok) throw new Error(await readApiError(response));
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

/** Open a protected HTML artifact in a new tab while preserving authenticated access. */
export async function viewProtectedArtifact(downloadUrl) {
  const response = await apiFetch(`${API_BASE_URL}${downloadUrl}`);
  if (!response.ok) throw new Error(await readApiError(response));
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  window.open(objectUrl, "_blank", "noopener,noreferrer");
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
}
