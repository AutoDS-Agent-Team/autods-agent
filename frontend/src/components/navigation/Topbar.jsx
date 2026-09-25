import { useNavigate } from 'react-router-dom'
import { clearAccessToken } from '../../api/client.js'
export default function Topbar({ user }) { const navigate=useNavigate(); return <header className="topbar"><span>{user.email}</span><button onClick={() => { clearAccessToken(); navigate('/login') }}>Log out</button></header> }
