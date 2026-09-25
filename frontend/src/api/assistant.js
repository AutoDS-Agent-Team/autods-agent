import { API_BASE_URL, apiFetch, readApiError } from './client.js'

export async function askAssistant(question, { contextType = 'auto', datasetId, experimentId, conversationId = 'default' } = {}) {
  try {
    const response = await apiFetch(`${API_BASE_URL}/api/v1/assistant/query`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, context_type: contextType, dataset_id: datasetId || undefined, experiment_id: experimentId || undefined, conversation_id: conversationId }),
    })
    if (response.status === 401) throw new Error('Session expired. Please sign in again.')
    if (response.status === 404) throw new Error('The selected dataset or experiment is unavailable.')
    if (!response.ok) throw new Error(await readApiError(response))
    return response.json()
  } catch (error) {
    if (error instanceof TypeError && /fetch/i.test(error.message)) throw new Error('Cannot reach analytics service. Check that the backend is running and the API URL is configured.')
    throw error
  }
}
