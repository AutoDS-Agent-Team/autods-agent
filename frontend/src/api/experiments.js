import { API_BASE_URL, apiFetch, readApiError } from './client.js'

export async function createExperiment(datasetId, objective) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/experiments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, objective }),
  })

  if (!response.ok) {
    throw new Error(await readApiError(response))
  }

  return response.json()
}

export async function confirmExperiment(experimentId, targetColumn, taskType) {
  const response = await apiFetch(
    `${API_BASE_URL}/api/v1/experiments/${experimentId}/confirm`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_column: targetColumn, task_type: taskType }),
    },
  )

  if (!response.ok) {
    throw new Error(await readApiError(response))
  }

  return response.json()
}

export async function generatePipelinePlan(experimentId) {
  const response = await apiFetch(
    `${API_BASE_URL}/api/v1/experiments/${experimentId}/plan`,
    { method: 'POST' },
  )

  if (!response.ok) {
    throw new Error(await readApiError(response))
  }

  return response.json()
}

export async function updateAdaptivePipeline(experimentId, balancedClassWeightModels) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/experiments/${experimentId}/plan/adaptive`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ balanced_class_weight_models: balancedClassWeightModels }),
  })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function trainExperiment(experimentId) {
  const response = await apiFetch(
    `${API_BASE_URL}/api/v1/experiments/${experimentId}/train`,
    { method: 'POST' },
  )

  if (!response.ok) {
    throw new Error(await readApiError(response))
  }

  return response.json()
}

export async function evaluateExperiment(experimentId) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/experiments/${experimentId}/evaluate`, { method: 'POST' })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function optimizeExperiment(experimentId) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/experiments/${experimentId}/optimize`, { method: 'POST' })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function getExplainability(experimentId) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/experiments/${experimentId}/explainability`)
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function createPredictions(experimentId, file) {
  const form = new FormData()
  form.append('file', file)
  const response = await apiFetch(`${API_BASE_URL}/api/v1/experiments/${experimentId}/predictions`, { method: 'POST', body: form })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function createReport(experimentId) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/experiments/${experimentId}/report`, { method: 'POST' })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function createPdfReport(experimentId) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/experiments/${experimentId}/report/pdf`, { method: 'POST' })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function getReportHistory() {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/reports`)
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}

export async function createReflection(experimentId) {
  const response = await apiFetch(`${API_BASE_URL}/api/v1/experiments/${experimentId}/reflection`, { method: 'POST' })
  if (!response.ok) throw new Error(await readApiError(response))
  return response.json()
}
