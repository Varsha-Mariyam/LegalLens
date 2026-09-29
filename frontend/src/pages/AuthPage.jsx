import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";

export default function AuthPage({ mode, onAuth }) {
  const isLogin = mode === "login";
  const [form, setForm] = useState({ name: "", email: "", password: "" });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const data = isLogin ? await api.login(form.email, form.password) : await api.register(form.name, form.email, form.password);
      onAuth(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="auth">
      <section className="auth-intro">
        <div className="auth-rail" aria-hidden="true">
          {["low", "low", "high", "low", "medium", "high", "low", "medium", "low"].map((r, i) => (
            <span key={i} className={`seg risk-${r}`} style={{ flexGrow: [2, 1, 3, 1, 2, 3, 1, 2, 1][i], animationDelay: `${i * 70}ms` }} />
          ))}
        </div>
        <div>
          <h1>Read the contract before it reads you.</h1>
          <p>
            Upload an employment, rental, NDA or service agreement. LegalLens splits it into clauses, explains each one in
            plain language, marks potential risks with the evidence behind them, compares it with a reference template and
            answers your questions about it.
          </p>
        </div>
      </section>
      <form className="auth-card" onSubmit={submit}>
        <h2>{isLogin ? "Log in" : "Create an account"}</h2>
        {!isLogin && (
          <label>Name<input value={form.name} onChange={set("name")} required autoComplete="name" /></label>
        )}
        <label>Email<input type="email" value={form.email} onChange={set("email")} required autoComplete="email" /></label>
        <label>
          Password
          <input type="password" value={form.password} onChange={set("password")} required minLength={isLogin ? 1 : 8}
                 autoComplete={isLogin ? "current-password" : "new-password"} />
          {!isLogin && <small>At least 8 characters.</small>}
        </label>
        {error && <p className="error" role="alert">{error}</p>}
        <button className="primary" disabled={busy}>{busy ? "Please wait…" : isLogin ? "Log in" : "Create account"}</button>
        <p className="muted">
          {isLogin ? <>New here? <Link to="/register">Create an account</Link></> : <>Already registered? <Link to="/login">Log in</Link></>}
        </p>
      </form>
    </main>
  );
}
