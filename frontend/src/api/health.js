import { API_BASE_URL } from './client.js'

export async function getBackendHealth(signal) {
  const response = await fetch(`${API_BASE_URL}/api/v1/health`, { signal })

  if (!response.ok) {
    throw new Error(`Health request failed with status ${response.status}`)
  }

  return response.json()
}
