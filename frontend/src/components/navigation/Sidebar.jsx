import { Link, NavLink } from "react-router-dom";
const links = [
  ["/dashboard", "Dashboard"],
  ["/experiments/new", "New Experiment"],
  ["/experiments", "History"],
  ["/datasets", "Datasets"],
  ["/models", "Models"],
  ["/reports", "Reports"],
  ["/settings", "Settings"],
];
export default function Sidebar() {
  return (
    <aside className="sidebar">
      <Link className="brand" to="/dashboard">
        AutoDS-Agent
      </Link>
      {links.map(([path, label]) => (
        <NavLink key={path} to={path}>
          {label}
        </NavLink>
      ))}
    </aside>
  );
}
