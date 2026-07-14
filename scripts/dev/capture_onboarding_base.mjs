import fs from "node:fs";

const outPath =
  process.argv[2] ?? "frontend/react_app/src/assets/onboarding/03-before-route.png";
const targetUrl = process.argv[3] ?? "http://localhost:3000/";
const debuggerUrl = process.argv[4] ?? "http://127.0.0.1:9223";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function main() {
  const page = await fetch(`${debuggerUrl}/json/new?${encodeURIComponent(targetUrl)}`, {
    method: "PUT",
  }).then((response) => response.json());
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map();

  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    }
  };

  await new Promise((resolve) => {
    ws.onopen = resolve;
  });

  const send = (method, params = {}) =>
    new Promise((resolve) => {
      const messageId = ++id;
      pending.set(messageId, resolve);
      ws.send(JSON.stringify({ id: messageId, method, params }));
    });

  await send("Page.enable");
  await send("Runtime.enable");
  await sleep(1500);
  await send("Runtime.evaluate", {
    expression: `
      localStorage.setItem("wbm_onboarding_seen", "true");
      localStorage.setItem("wbm_planner_help_seen", "true");
      location.reload();
    `,
  });
  await sleep(6500);

  const screenshot = await send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: false,
    fromSurface: true,
  });
  fs.writeFileSync(outPath, Buffer.from(screenshot.result.data, "base64"));
  ws.close();
  console.log(outPath);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
