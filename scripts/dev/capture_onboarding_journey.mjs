import fs from "node:fs";
import path from "node:path";

const targetUrl = process.argv[2] ?? "http://localhost:3000/";
const debuggerUrl = process.argv[3] ?? "http://127.0.0.1:9224";
const outputDirectory = path.resolve(
  process.argv[4] ?? "frontend/react_app/src/assets/onboarding",
);

const ORIGIN = { lat: -36.8267, lon: -73.0498 };
const DESTINATION = { lat: -36.8114, lon: -73.0490 };
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function main() {
  fs.mkdirSync(outputDirectory, { recursive: true });
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
    new Promise((resolve, reject) => {
      const messageId = ++id;
      const timeout = setTimeout(() => {
        pending.delete(messageId);
        reject(new Error(`Chrome DevTools no respondio a ${method}.`));
      }, 120_000);
      pending.set(messageId, (message) => {
        clearTimeout(timeout);
        if (message.error) {
          reject(new Error(`${method}: ${message.error.message}`));
          return;
        }
        resolve(message);
      });
      ws.send(JSON.stringify({ id: messageId, method, params }));
    });

  const evaluate = async (expression) => {
    const response = await send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (response.result.exceptionDetails) {
      throw new Error(
        response.result.exceptionDetails.exception?.description ??
          response.result.exceptionDetails.text ??
          "Fallo al evaluar JavaScript en la pagina.",
      );
    }
    return response.result.result.value;
  };

  const waitFor = async (expression, description, timeoutMs = 120_000) => {
    const startedAt = Date.now();
    while (Date.now() - startedAt < timeoutMs) {
      if (await evaluate(expression)) {
        return;
      }
      await sleep(350);
    }
    throw new Error(`Tiempo agotado esperando: ${description}`);
  };

  const capture = async (filename) => {
    await sleep(900);
    const screenshot = await send("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: false,
      fromSurface: true,
    });
    const outputPath = path.join(outputDirectory, filename);
    fs.writeFileSync(outputPath, Buffer.from(screenshot.result.data, "base64"));
    console.log(outputPath);
  };

  const clickButton = async (label, selector = "button") => {
    const clicked = await evaluate(`(() => {
      const target = [...document.querySelectorAll(${JSON.stringify(selector)})]
        .find((node) => node.textContent.replace(/\\s+/g, " ").trim().includes(${JSON.stringify(label)}));
      if (!target || target.disabled) return false;
      target.click();
      return true;
    })()`);
    if (!clicked) {
      throw new Error(`No se encontro un boton habilitado con el texto: ${label}`);
    }
  };

  await send("Page.enable");
  await send("Runtime.enable");
  const navigation = await send("Page.navigate", { url: targetUrl });
  if (navigation.result.errorText) {
    throw new Error(`No se pudo abrir la aplicacion: ${navigation.result.errorText}`);
  }
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1902,
    height: 884,
    deviceScaleFactor: 1,
    mobile: false,
  });

  await waitFor(
    `location.href.startsWith(${JSON.stringify(new URL(targetUrl).origin)}) && document.readyState === "complete"`,
    "navegacion y carga completa del documento",
  );
  await evaluate(`(() => {
    localStorage.setItem("wbm_onboarding_seen", "true");
    localStorage.setItem("wbm_planner_help_seen", "true");
    return true;
  })()`);
  await send("Page.reload", { ignoreCache: true });

  await waitFor('document.readyState === "complete"', "recarga completa del documento");
  await waitFor('Boolean(document.querySelector(".maplibregl-canvas"))', "instancia visual del mapa");
  const mapExposed = await evaluate(`(() => {
    const node = document.querySelector(".planner-map");
    const fiberKey = node && Object.keys(node).find((key) => key.startsWith("__reactFiber$"));
    let fiber = fiberKey ? node[fiberKey] : null;
    while (fiber && typeof fiber.memoizedProps?.onPickPoint !== "function") fiber = fiber.return;
    let hook = fiber?.memoizedState ?? null;
    while (hook) {
      const candidate = hook.memoizedState?.current;
      if (candidate && typeof candidate.fitBounds === "function" && typeof candidate.areTilesLoaded === "function") {
        window.__onboardingMap = candidate;
        return true;
      }
      hook = hook.next;
    }
    return false;
  })()`);
  if (!mapExposed) {
    throw new Error("No se pudo validar el estado interno del mapa.");
  }
  await waitFor(
    `window.__onboardingMap.areTilesLoaded() &&
      document.body.innerText.includes("Jueves, 13 de marzo de 2025") &&
      [...document.querySelectorAll("select")].some((select) => select.value === "8" && !select.disabled) &&
      !document.body.innerText.includes("Cargando condiciones") &&
      !document.body.innerText.includes("Cargando horas") &&
      !document.querySelector(".environment-side-panel")?.innerText.includes("Sin dato") &&
      !document.querySelector(".skeleton")`,
    "mapa, calendario, hora y condiciones ambientales completas",
  );
  await capture("01-date-time.png");

  const pointsCommitted = await evaluate(`(() => {
    const node = document.querySelector(".planner-map");
    const fiberKey = node && Object.keys(node).find((key) => key.startsWith("__reactFiber$"));
    let fiber = fiberKey ? node[fiberKey] : null;
    while (fiber && typeof fiber.memoizedProps?.onPickPoint !== "function") fiber = fiber.return;
    if (!fiber) return false;
    fiber.memoizedProps.onPickPoint("origin", ${JSON.stringify(ORIGIN)});
    fiber.memoizedProps.onPickPoint("destination", ${JSON.stringify(DESTINATION)});
    return true;
  })()`);
  if (!pointsCommitted) {
    throw new Error("No se pudo acceder al selector de puntos del mapa.");
  }
  await evaluate(`(() => {
    window.__onboardingMap.fitBounds(
      [[${ORIGIN.lon}, ${ORIGIN.lat}], [${DESTINATION.lon}, ${DESTINATION.lat}]],
      { padding: { top: 145, right: 390, bottom: 130, left: 100 }, maxZoom: 13.5, duration: 0 }
    );
    window.__onboardingMap.triggerRepaint();
    return true;
  })()`);

  await waitFor(
    `document.body.innerText.includes("-36.82670, -73.04980") &&
      document.body.innerText.includes("-36.81140, -73.04900") &&
      document.querySelectorAll(".maplibregl-marker.planner-route-pin").length === 2 &&
      window.__onboardingMap.areTilesLoaded() &&
      [...document.querySelectorAll("button")].some((button) =>
        button.textContent.includes("Planificar viaje") && !button.disabled
      )`,
    "origen, destino y accion de planificacion listos",
  );
  await capture("02-origin-destination.png");
  await capture("03-plan-trip.png");

  await clickButton("Planificar viaje");
  await waitFor(
    `document.querySelectorAll(".route-card-list .preference-card").length === 3 &&
      [...document.querySelectorAll(".route-card-list .preference-card")].every((card) =>
        card.innerText.includes("min") && card.innerText.includes("km")
      ) &&
      document.querySelectorAll(".route-svg-line").length === 3 &&
      window.__onboardingMap.areTilesLoaded() &&
      ![...document.querySelectorAll("button")].some((button) =>
        button.textContent.includes("Planificando")
      )`,
    "las tres rutas dibujadas con sus metricas reales",
  );
  await capture("04-route-priorities.png");

  await clickButton("Menor exposición ambiental", ".route-card-list .preference-card");
  await sleep(1_000);
  console.log(await evaluate(`JSON.stringify({
    guidance: document.querySelector(".route-guidance-panel")?.innerText ?? null,
    insightCount: document.querySelectorAll(".route-guidance-panel .route-insight-card").length,
    metrics: document.querySelector(".route-side-metrics")?.innerText ?? null
  })`));
  await waitFor(
    `document.querySelector(".route-guidance-panel") &&
      document.querySelectorAll(".route-guidance-panel .route-insight-card").length >= 4 &&
      document.querySelector(".route-side-metrics")?.innerText.includes("Menor exposición ambiental") &&
      !document.querySelector(".route-side-metrics")?.innerText.includes("Sin dato") &&
      window.__onboardingMap.areTilesLoaded()`,
    "recomendacion ambiental y mensajes explicativos completos",
  );
  await capture("05-recommendation.png");

  await evaluate(`(() => {
    const panel = document.querySelector(".results-sheet > .product-panel:first-child");
    const layers = document.querySelector(".urban-layers-panel");
    if (!panel || !layers) return false;
    const wanted = ["Parques y areas verdes", "Lagos, lagunas y cursos de agua", "Ciclovias"];
    for (const label of layers.querySelectorAll("label")) {
      const normalized = label.textContent.normalize("NFD").replace(/[\\u0300-\\u036f]/g, "");
      if (wanted.some((name) => normalized.includes(name))) {
        const checkbox = label.querySelector('input[type="checkbox"]');
        if (checkbox && !checkbox.checked) checkbox.click();
      }
    }
    panel.scrollTop = Math.max(0, layers.offsetTop - 8);
    window.__onboardingMap.triggerRepaint();
    return true;
  })()`);
  await waitFor(
    `document.querySelector(".urban-layers-panel")?.innerText.includes("3 activas") &&
      [...document.querySelectorAll(".urban-layers-panel input[type=checkbox]")].filter((input) => input.checked).length === 3 &&
      document.querySelector(".urban-layers-panel").getBoundingClientRect().width > 0 &&
      document.querySelector(".urban-layers-panel").getBoundingClientRect().height > 0 &&
      document.querySelector(".urban-layers-panel").getBoundingClientRect().top >= 0 &&
      document.querySelector(".urban-layers-panel").getBoundingClientRect().top < innerHeight &&
      window.__onboardingMap.areTilesLoaded()`,
    "capas sustentables dibujadas y panel de activacion visible",
  );
  await capture("07-sustainable-layers.png");

  await evaluate(`(() => {
    const layers = document.querySelector(".urban-layers-panel");
    if (!layers) return false;
    for (const checkbox of layers.querySelectorAll('input[type="checkbox"]:checked')) checkbox.click();
    window.__onboardingMap.triggerRepaint();
    return true;
  })()`);
  await waitFor(
    `[...document.querySelectorAll(".urban-layers-panel input[type=checkbox]")].every((input) => !input.checked) &&
      window.__onboardingMap.areTilesLoaded()`,
    "capas opcionales desactivadas antes de iniciar el viaje",
  );

  await clickButton("Iniciar viaje");
  await waitFor(
    `document.querySelector(".journey-bar") &&
      document.querySelector(".journey-priority-card")?.innerText.includes("Menor exposición") &&
      [...document.querySelectorAll("button")].some((button) => button.textContent.includes("Finalizar viaje"))`,
    "recorrido ambiental listo para finalizar",
  );
  await capture("06-finish-action.png");

  await clickButton("Finalizar viaje");
  await waitFor(
    `document.querySelector(".journey-finish-dialog") &&
      document.querySelectorAll(".journey-finish-summary strong").length === 4 &&
      [...document.querySelectorAll(".journey-finish-summary strong")].every((node) => node.textContent.trim()) &&
      Boolean(document.querySelector(".journey-finish-reward"))`,
    "cierre real, resumen, sello y recompensa del viaje",
  );
  await capture("06-finish-result.png");

  await clickButton("Revisar recorrido");
  await waitFor(
    `Boolean(document.querySelector(".journey-review-panel")) &&
      document.querySelectorAll(".journey-review-panel .journey-review-metrics strong").length === 4 &&
      window.__onboardingMap.areTilesLoaded()`,
    "revision posterior del recorrido ambiental",
  );

  ws.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
