// Thin client for the FastAPI backend. The token is kept in memory and in sessionStorage.
const TOKEN_KEY = "legallens_token";
let token = sessionStorage.getItem(TOKEN_KEY);

export function setToken(t) {
  token = t;
  if (t) sessionStorage.setItem(TOKEN_KEY, t);
  else sessionStorage.removeItem(TOKEN_KEY);
}
export const hasToken = () => Boolean(token);

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function request(path, { method = "GET", body, form, raw } = {}) {
  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(path, { method, headers, body: form ?? (body !== undefined ? JSON.stringify(body) : undefined) });
  if (res.status === 401 && token) {
    setToken(null);
    window.dispatchEvent(new Event("legallens:logout"));
  }
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") msg = data.detail;
      else if (Array.isArray(data.detail)) msg = data.detail.map((d) => d.msg).join("; ");
    } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, msg);
  }
  if (raw) return res;
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  register: (name, email, password) => request("/api/auth/register", { method: "POST", body: { name, email, password } }),
  login: (email, password) => request("/api/auth/login", { method: "POST", body: { email, password } }),
  me: () => request("/api/auth/me"),
  listDocuments: () => request("/api/documents"),
  upload: (file, documentType) => {
    const form = new FormData();
    form.append("file", file);
    form.append("document_type", documentType);
    return request("/api/documents", { method: "POST", form });
  },
  getDocument: (id) => request(`/api/documents/${id}`),
  status: (id) => request(`/api/documents/${id}/status`),
  analysis: (id) => request(`/api/documents/${id}/analysis`),
  text: (id) => request(`/api/documents/${id}/text`),
  reprocess: (id) => request(`/api/documents/${id}/reprocess`, { method: "POST" }),
  deleteDocument: (id) => request(`/api/documents/${id}`, { method: "DELETE" }),
  personas: (id) => request(`/api/documents/${id}/persona`),
  generatePersona: (id, persona) => request(`/api/documents/${id}/persona`, { method: "POST", body: { persona } }),
  qaHistory: (id) => request(`/api/documents/${id}/qa`),
  ask: (id, question) => request(`/api/documents/${id}/qa`, { method: "POST", body: { question } }),
  clearQa: (id) => request(`/api/documents/${id}/qa`, { method: "DELETE" }),
  comparison: (id) => request(`/api/documents/${id}/comparison`),
  report: (id) => request(`/api/documents/${id}/report`, { raw: true }),
  systemStatus: () => request("/api/system/status"),
  knowledge: () => request("/api/system/knowledge"),
  reindex: () => request("/api/system/reindex", { method: "POST" }),
};
