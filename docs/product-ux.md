# Product UX

## Tono de copy

- Claro, directo y no tecnico.
- Mezcla natural de espanol con labels cortos en ingles si ayudan a escanear rapido:
  `Safe`, `Balanced`, `Fast`.
- Evitar lenguaje de tesis en la home.
- Reservar terminos como `reference`, `UBCF`, `IBCF`, `calidad de datos` y `perfil sintetico` para `/demo`.

## Labels principales

- CTA principal:
  `Planificar viaje`
- Alternativas:
  `Ruta recomendada`, `Ruta mas rapida`, `Menor exposicion historica`
- Acciones:
  `Intercambiar`, `Planificar viaje`, `Como usar`, `Ir al modo demo`
- Estados:
  `Preparando rutas`, `Selecciona origen y destino`, `No se pudo completar la accion`

## Journeys

### Usuario nuevo

1. Abre `/`.
2. Ve origen, destino y CTA principal en la parte superior.
3. Selecciona origen y destino directamente sobre el mapa.
4. Toca `Planificar viaje`.
5. Entiende en menos de 30 segundos:
   ETA, demora, alertas y motivo principal.

### Usuario con prisa

1. Marca el origen en el mapa.
2. Marca el destino.
3. Cambia a `Fast`.
4. Revisa la tarjeta `Ruta mas rapida`.

### Usuario cauteloso

1. Marca puntos en Concepcion.
2. Usa `Safe`.
3. Valida alertas, hotspots y vias clave antes de salir.

### Evaluador academico

1. Entra a `/`.
2. Comprueba experiencia producto.
3. Abre `/demo`.
4. Revisa escenarios curados, comparacion `reference/UBCF/IBCF` y explicacion del modelo.

## Estados operativos

- Warm-up:
  explicar que el backend esta preparando el grafo.
- Fuera de cobertura:
  pedir mover el pin a una calle cubierta.
- Config faltante:
  explicar que el mapa interactivo queda desactivado.
- Backend no listo:
  mantener CTA deshabilitado y mostrar mensaje.
- Rutas duplicadas:
  colapsar alternativas y mostrar multiples badges.
