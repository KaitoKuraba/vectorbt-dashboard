module.exports = async function handler(req, res) {
  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    return res.status(500).json({ error: "BACKEND_URL not set" });
  }
  const pathParts = [].concat(req.query['...path'] || []);
  const targetPath = "/api/" + pathParts.join("/");
  const queryParams = Object.entries(req.query)
    .filter(([k]) => k !== '...path')
    .map(([k, v]) => encodeURIComponent(k) + "=" + encodeURIComponent(v))
    .join("&");
  const targetUrl = backendUrl + targetPath + (queryParams ? "?" + queryParams : "");
  try {
    const fetchOptions = { method: req.method, headers: { "Content-Type": "application/json" } };
    if (["POST", "PUT", "PATCH"].includes(req.method) && req.body) {
      fetchOptions.body = JSON.stringify(req.body);
    }
    const response = await fetch(targetUrl, fetchOptions);
    const text = await response.text();
    try {
      const data = JSON.parse(text);
      res.setHeader("Content-Type", "application/json");
      res.status(response.status).json(data);
    } catch (parseErr) {
      res.status(502).json({ error: "Non-JSON from backend", targetUrl, httpStatus: response.status, raw: text.slice(0, 300) });
    }
  } catch (err) {
    res.status(502).json({ error: "Fetch failed: " + err.message, targetUrl });
  }
};
