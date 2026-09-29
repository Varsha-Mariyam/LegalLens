import { useEffect, useState } from "react";
import { Link, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { api, hasToken, setToken } from "./api.js";
import AuthPage from "./pages/AuthPage.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import DocumentView from "./pages/DocumentView.jsx";
import SystemStatus from "./pages/SystemStatus.jsx";

export default function App() {
  const [user, setUser] = useState(null);
  const [checking, setChecking] = useState(hasToken());
  const navigate = useNavigate();

  useEffect(() => {
    if (!hasToken()) return;
    api.me().then(setUser).catch(() => setToken(null)).finally(() => setChecking(false));
  }, []);

  useEffect(() => {
    const onLogout = () => { setUser(null); navigate("/login"); };
    window.addEventListener("legallens:logout", onLogout);
    return () => window.removeEventListener("legallens:logout", onLogout);
  }, [navigate]);

  const onAuth = (data) => { setToken(data.access_token); setUser(data.user); navigate("/"); };
  const logout = () => { setToken(null); setUser(null); navigate("/login"); };

  if (checking) return <div className="boot">Loading…</div>;

  return (
    <div className="app">
      {user && (
        <header className="topbar">
          <Link to="/" className="brand" aria-label="LegalLens home">
            <span className="brand-mark" aria-hidden="true"><i /><i /></span>
            LegalLens
          </Link>
          <nav className="topnav">
            <Link to="/">Documents</Link>
            <Link to="/system">System</Link>
            <span className="whoami">{user.name}</span>
            <button className="linklike" onClick={logout}>Log out</button>
          </nav>
        </header>
      )}
      <Routes>
        <Route path="/login" element={user ? <Navigate to="/" /> : <AuthPage mode="login" onAuth={onAuth} />} />
        <Route path="/register" element={user ? <Navigate to="/" /> : <AuthPage mode="register" onAuth={onAuth} />} />
        <Route path="/" element={user ? <Dashboard /> : <Navigate to="/login" />} />
        <Route path="/documents/:id" element={user ? <DocumentView /> : <Navigate to="/login" />} />
        <Route path="/system" element={user ? <SystemStatus /> : <Navigate to="/login" />} />
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
      <footer className="footnote">
        LegalLens is an academic document-analysis assistant. It is not a substitute for advice from a qualified lawyer.
      </footer>
    </div>
  );
}
