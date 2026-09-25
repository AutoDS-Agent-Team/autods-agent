import { useEffect, useState } from 'react'
import { Navigate, Outlet } from 'react-router-dom'
import { clearAccessToken, getAccessToken } from '../api/client.js'
import { getCurrentUser } from '../api/auth.js'
import Sidebar from '../components/navigation/Sidebar.jsx'
import Topbar from '../components/navigation/Topbar.jsx'
export default function AppLayout() { const [user, setUser] = useState(undefined); useEffect(() => { if (!getAccessToken()) { setUser(null); return } getCurrentUser().then(setUser).catch(() => { clearAccessToken(); setUser(null) }) }, []); if (user === undefined) return <main className="page-shell"><p>Restoring secure session…</p></main>; if (!user) return <Navigate to="/login" replace />; return <div className="workspace"><Sidebar /><section className="workspace-main"><Topbar user={user} /><main className="route-content"><Outlet /></main></section></div> }
