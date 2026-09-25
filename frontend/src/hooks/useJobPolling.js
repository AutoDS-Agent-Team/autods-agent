import { useEffect } from 'react'
import { getJob } from '../api/jobs.js'
const TERMINAL = new Set(['COMPLETED', 'FAILED', 'CANCELLED'])
export function useJobPolling(jobId, onUpdate, onError) {
  useEffect(() => {
    if (!jobId) return undefined
    let active = true
    let timer
    const poll = async () => {
      try { const job = await getJob(jobId); if (active) onUpdate(job); return TERMINAL.has(job.status) } catch (error) { if (active) onError(error); return true }
    }
    const schedule = async () => { if (await poll() && timer) { clearTimeout(timer); timer = undefined } else if (active) timer = setTimeout(schedule, 2000) }
    schedule()
    return () => { active = false; if (timer) clearTimeout(timer) }
  // Callbacks intentionally belong to the job lifecycle that started this poll.
  // Depending only on the ID prevents state updates from restarting the polling loop.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])
}
