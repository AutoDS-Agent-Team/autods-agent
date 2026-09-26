import { API_BASE_URL, apiFetch, readApiError } from "./client.js";
export async function startExperimentJob(experimentId) {
  const response = await apiFetch(
    `${API_BASE_URL}/api/v1/experiments/${experimentId}/run`,
    { method: "POST" },
  );
  if (!response.ok) throw new Error(await readApiError(response));
  return response.json();
}
export async function getJob(jobId) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/jobs/${jobId}`);
  if (!response.ok) throw new Error(await readApiError(response));
  return response.json();
}
export async function getExperimentHistory(page = 1) {
  const response = await apiFetch(
    `${API_BASE_URL}/api/v1/experiments?page=${page}`,
  );
  if (!response.ok) throw new Error(await readApiError(response));
  return response.json();
}

export async function getExperimentDetail(experimentId) {
  const response = await apiFetch(
    `${API_BASE_URL}/api/v1/experiments/${experimentId}`,
  );
  if (!response.ok) throw new Error(await readApiError(response));
  return response.json();
}
