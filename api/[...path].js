module.exports = async function handler(req, res) {
  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    return res.status(500).json({ error: "BACKEND_URL environment variable is not set.", hint: "Add BACKEND_URL in Vercel Dashboard → Project → Settings → Environment Variables" });
  }
  const pathParts = req.query.path || [];
  const targetPath = "/api/" + pathParts.join("/");
  const queryParams = Object.entries(req.query).filter(([k]) => k !== "path").map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&");
  const targetUrl = `${backendUrl}${targetPath}${queryParams ? "?" + queryParams : ""}`;
  try {
    const fetchOptions = { method: req.method, headers: { "Content-Type": "application/json" } };
    if (["POST", "PUT", "PATCH"].includes(req.method) && req.body) {
      fetchOptions.body = JSON.stringify(req.body);
    }
    const response = await fetch(targetUrl, fetchOptions);
    const data = await response.json();
    res.setHeader("Content-Type", "application/json");
    res.status(response.status).json(data);
  } catch (err) {
    res.status(502).json({ error: `Backend unreachable: ${err.message}`, hint: `Check that BACKEND_URL (${backendUrl}) is correct and the Railway server is running.` });
  }
};
