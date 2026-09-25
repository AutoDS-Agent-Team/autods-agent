const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
// API modules already include `/api/v1`; production Nginx proxies that path.
export const API_BASE_URL = configuredApiBaseUrl === '/api'
  ? ''
  : configuredApiBaseUrl.replace(/\/$/, '')

const TOKEN_KEY = 'autods_access_token'
export const getAccessToken = () => localStorage.getItem(TOKEN_KEY)
export const setAccessToken = (token) => localStorage.setItem(TOKEN_KEY, token)
export const clearAccessToken = () => localStorage.removeItem(TOKEN_KEY)
export async function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers || {})
  const token = getAccessToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(url, { ...options, headers })
  if (response.status === 401) { clearAccessToken(); window.dispatchEvent(new Event('autods-auth-expired')) }
  return response
}

export async function readApiError(response) {
  try {
    const payload = await response.json()
    return payload.detail || `Request failed with status ${response.status}`
  } catch {
    return `Request failed with status ${response.status}`
  }
}
