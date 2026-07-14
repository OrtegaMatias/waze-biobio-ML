import fs from "node:fs";

const outPath = process.argv[2] ?? "frontend/react_app/src/assets/onboarding-step-check.png";
const step = Number(process.argv[3] ?? 1);
const targetUrl = process.argv[4] ?? "http://localhost:3000/";
const debuggerUrl = process.argv[5] ?? "http://127.0.0.1:9224";

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
  await sleep(1800);
  await send("Runtime.evaluate", {
    expression: `
      localStorage.setItem("wbm_onboarding_seen", "true");
    `,
  });
  await send("Page.reload", { ignoreCache: true });
  await sleep(1800);
  await send("Runtime.evaluate", {
    expression: `
      [...document.querySelectorAll(".topbar-product button")]
        .find((button) => button.textContent.trim() === "Ver paso a paso")
        ?.click();
    `,
  });
  await sleep(600);
  for (let index = 0; index < step; index += 1) {
    await send("Runtime.evaluate", {
      expression: `
        [...document.querySelectorAll(".onboarding-dialog button")]
          .find((button) => button.textContent.trim() === "Ver paso a paso" || button.textContent.trim() === "Siguiente")
          ?.click();
      `,
    });
    await sleep(450);
  }

  const screenshot = await send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: false,
    fromSurface: true,
  });
  fs.writeFileSync(outPath, Buffer.from(screenshot.result.data, "base64"));
  await send("Page.close");
  ws.close();
  console.log(outPath);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
