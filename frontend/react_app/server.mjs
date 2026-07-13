import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve } from "node:path";

const distDir = resolve(process.env.FRONTEND_DIST || "./dist");
const backendUrl = (process.env.BACKEND_URL || "http://backend:8000").replace(/\/$/, "");
const port = Number(process.env.PORT || 3000);

const apiPrefixes = ["/health", "/readyz", "/metadata", "/places", "/recommendations", "/routes", "/system"];
const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
};

function isApiRequest(pathname) {
  return apiPrefixes.some((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));
}

function sendFile(response, filePath) {
  response.writeHead(200, {
    "Content-Type": contentTypes[extname(filePath)] || "application/octet-stream",
    "Cache-Control": "no-store",
  });
  createReadStream(filePath).pipe(response);
}

async function proxyApi(request, response) {
  const upstream = await fetch(`${backendUrl}${request.url}`, {
    method: request.method,
    headers: request.headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : request,
    duplex: "half",
  });

  response.writeHead(upstream.status, Object.fromEntries(upstream.headers));
  if (upstream.body) {
    const reader = upstream.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      response.write(value);
    }
  }
  response.end();
}

createServer(async (request, response) => {
  try {
    const url = new URL(request.url || "/", `http://${request.headers.host || "localhost"}`);
    if (isApiRequest(url.pathname)) {
      await proxyApi(request, response);
      return;
    }

    const requestedPath = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, "");
    const filePath = join(distDir, requestedPath === "/" ? "index.html" : requestedPath);
    const resolvedPath = resolve(filePath);
    if (resolvedPath.startsWith(distDir) && existsSync(resolvedPath) && statSync(resolvedPath).isFile()) {
      sendFile(response, resolvedPath);
      return;
    }

    sendFile(response, join(distDir, "index.html"));
  } catch (error) {
    response.writeHead(502, { "Content-Type": "text/plain; charset=utf-8" });
    response.end(error instanceof Error ? error.message : "Frontend proxy error");
  }
}).listen(port, "0.0.0.0", () => {
  console.log(`Frontend server listening on http://0.0.0.0:${port}`);
  console.log(`Proxying API requests to ${backendUrl}`);
});
