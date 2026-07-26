# QA de rutas por objetivo

Alcance: reestructuración del cálculo vehicular de **Llegar antes**, **Circulación más fluida** y **Menor exposición ambiental**.

## Comprobaciones automáticas

| Estado | Comprobación | Evidencia esperada |
| --- | --- | --- |
| Aprobada | Pruebas del backend | `python -m pytest backend/fastapi_app/tests -q` finaliza sin fallos. |
| Aprobada | Pruebas del frontend | `npm test -- --run` finaliza sin fallos. |
| Aprobada | Compilación del frontend | `npm run build` finaliza sin errores. |
| Aprobada | Contrato de costos | Cada ruta informa internamente su objetivo y el desglose agregado por tramos lógicos. |
| Aprobada | Límites temporales | Dijkstra restringido no admite que las rutas fluida o ambiental superen el 150 % del tiempo contextual de Llegar antes. |
| Aprobada | Datos faltantes | PM2.5 y entorno urbano faltantes se mantienen neutros, se registran y se informan. |
| Aprobada | Restricción vehicular | Se bloquean caminos OSM no transitables en automóvil y se permiten vías vehiculares válidas. |
| Aprobada | Coherencia ambiental | Las mismas zonas visibles, incluidas las de memoria ponderada, alimentan el costo de la ruta ambiental. |

## Comprobaciones manuales obligatorias

Registrar cada resultado como **Aprobada**, **Fallida** o **Pendiente**.

| Estado | Paso | Criterio de aceptación |
| --- | --- | --- |
| Pendiente | 1. Caso original de Cerro Caracol | La ruta no entra a senderos interiores; puede bordear el cerro por calles vehiculares y no realiza el rodeo artificial observado anteriormente. |
| Pendiente | 2. Llegar antes con congestión | Para una fecha/hora con congestión, minimiza el tiempo contextual; al cambiar fecha u hora se recalcula y puede cambiar la ruta o su duración. |
| Pendiente | 3. Circulación más fluida | Evita congestión roja y naranja cuando existe una alternativa cuyo tiempo no supera el 150 % de Llegar antes. |
| Pendiente | 4. Menor exposición ambiental | Tolera verde, evita principalmente naranja y rojo, responde a PM2.5 y capas urbanas, y no supera el 150 % del tiempo de Llegar antes. |
| Pendiente | 5. Rutas coincidentes | Si dos o tres objetivos dan la misma geometría, la interfaz las conserva sin fabricar alternativas. |
| Pendiente | 6. Sentidos de tránsito | Ninguna ruta recorre una calle en sentido contrario según OpenStreetMap. |
| Pendiente | 7. Datos ambientales faltantes | Una fecha/hora sin PM2.5 informa la ausencia y no muestra un valor inventado. |
| Pendiente | 8. Interfaz y recálculo | Cambiar fecha/hora vuelve a calcular las tres rutas; no aparece ni se envía `travel_style`. |
| Pendiente | 9. Memoria ambiental | Las zonas de una y dos horas anteriores permanecen visibles y afectan el costo con intensidad reducida. |

## Aprobación final

- Estado: **Pendiente**.
- Responsable: usuario responsable de la validación funcional.
- Producción certificada: **No**, hasta completar los nueve pasos manuales y registrar la aprobación final.
