# QA de rutas por objetivo

Alcance: reestructuración del cálculo vehicular de **Llegar antes**, **Circulación más fluida** y **Menor exposición ambiental**.

Ejecución registrada: **12 de agosto de 2026**, sobre `integration/main` en `8f7b18d`.

## Comprobaciones automáticas

| Estado | Comprobación | Evidencia esperada |
| --- | --- | --- |
| Aprobada | Pruebas del backend | `python -m pytest backend/fastapi_app/tests -q`: 158 aprobadas. |
| Aprobada | Pruebas del frontend | `npx vitest run --pool=threads --maxWorkers=1`: 28 aprobadas en 7 archivos. |
| Aprobada | Compilación del frontend | `NODE_OPTIONS=--max-old-space-size=4096 npm run build`: compilación finalizada. |
| Aprobada | Contrato de costos | Cada ruta informa internamente su objetivo y el desglose agregado por tramos lógicos. |
| Aprobada | Límites temporales | La ruta fluida y la ambiental de viajes de más de 5 minutos no superan el 150 %; la ambiental de viajes de hasta 5 minutos no supera el 300 %. |
| Aprobada | Datos faltantes | PM2.5 y entorno urbano faltantes se mantienen neutros, se registran y se informan. |
| Aprobada | Restricción vehicular | Se bloquean caminos OSM no transitables en automóvil y se permiten vías vehiculares válidas. |
| Aprobada | Coherencia ambiental | Las mismas zonas visibles, incluidas las de memoria ponderada, alimentan el costo de la ruta ambiental. |

## Comprobaciones manuales obligatorias

Registrar cada resultado como **Aprobada**, **Fallida** o **Pendiente**.

| Estado | Paso | Criterio de aceptación | Evidencia del 12-08-2026 |
| --- | --- | --- | --- |
| Aprobada | 1. Caso original de Cerro Caracol | La ruta no entra a senderos interiores; puede bordear el cerro por calles vehiculares y no realiza el rodeo artificial observado anteriormente. | Pruebas unitarias aprobadas y validación visual confirmada por el usuario el 12-08-2026. |
| Aprobada | 2. Llegar antes con congestión | Para una fecha/hora con congestión, minimiza el tiempo contextual; al cambiar fecha u hora se recalcula y puede cambiar la ruta o su duración. | Caso API del 01-07-2025, 17:00 y 18:00: las tres geometrías cambiaron; Llegar antes mantuvo el menor tiempo. |
| Pendiente | 3. Circulación más fluida | Evita congestión roja y naranja cuando existe una alternativa cuyo tiempo no supera el 150 % de Llegar antes. | En el caso API observado coincidió justificadamente con Llegar antes; falta un caso visual con alternativa fluida distinta. |
| Pendiente | 4. Menor exposición ambiental | Tolera verde, evita principalmente naranja y rojo, responde a PM2.5 y capas urbanas; no supera el 300 % en viajes de hasta 5 minutos ni el 150 % en viajes más largos. | En el caso API redujo la cobertura congestionada de 2,8 % a 1,5 % y respetó el límite corto; falta inspección visual de zonas naranja y roja. |
| Aprobada | 5. Rutas coincidentes | Si dos o tres objetivos dan la misma geometría, la interfaz las conserva sin fabricar alternativas. | La API conservó Llegar antes y Circulación más fluida con la misma geometría y explicó la coincidencia. |
| Pendiente | 6. Sentidos de tránsito | Ninguna ruta recorre una calle en sentido contrario según OpenStreetMap. | Las pruebas automáticas de `oneway` pasan; falta inspección visual contra OSM. |
| Pendiente | 7. Datos ambientales faltantes | Una fecha/hora sin PM2.5 informa la ausencia y no muestra un valor inventado. | La prueba automática pasa, pero no se encontró una fecha/hora seleccionable sin PM2.5 para validación manual. |
| Pendiente | 8. Interfaz y recálculo | Cambiar fecha/hora vuelve a calcular las tres rutas; no aparece ni se envía `travel_style`. | La API recalculó las tres rutas y el contrato ya no contiene `travel_style`; falta inspección visual porque no había navegador conectado. |
| Aprobada | 9. Memoria ambiental | Las zonas de una y dos horas anteriores permanecen visibles y afectan el costo con intensidad reducida. | A las 19:00 se observaron zonas solo de memoria con desfase de 1 y 2 horas y peso reducido de 0,25. |

## Aprobación final

- Estado: **Pendiente**.
- Responsable: usuario responsable de la validación funcional.
- Producción certificada: **No**, hasta aprobar los pasos 3, 4, 6, 7 y 8 y registrar la aprobación final.
