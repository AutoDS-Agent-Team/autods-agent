import { API_BASE_URL, apiFetch, readApiError } from './client.js'

export async function uploadDataset(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await apiFetch(`${API_BASE_URL}/api/v1/datasets`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error(await readApiError(response))
  }

  return response.json()
}

export async function listDatasets() {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/datasets`)
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function deleteDataset(datasetId, cascade = false) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}?cascade=${cascade}`, { method: 'DELETE' })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function getDatasetProfile(datasetId) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/profile`)

  if (!response.ok) {
    throw new Error(await readApiError(response))
  }

  return response.json()
}

export async function analyzeDatasetProfile(datasetId) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/analysis`, { method: 'POST' })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function clusterDataset(datasetId) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/clusters`, { method: 'POST' })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function getDatasetAutoAnalysis(datasetId) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/auto-analysis`)
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function queryDataset(datasetId, query) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/analytics/query`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(query),
  })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function selectWorksheet(datasetId, worksheetName) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/worksheet`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ worksheet_name: worksheetName }),
  })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function detectAnomalies(datasetId, algorithm = 'isolation_forest') {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/anomalies`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ algorithm }),
  })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function forecastDataset(datasetId, timeColumn, targetColumn, algorithm = 'naive_forecast') {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/datasets/${datasetId}/forecast`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ time_column: timeColumn, target_column: targetColumn, algorithm }),
  })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}
