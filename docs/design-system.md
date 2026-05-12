# Movilidad clara

## Direccion

`Movilidad clara` define una interfaz luminosa, orientada a movilidad urbana y comprension rapida.
La app debe sentirse util antes de salir, no como dashboard tecnico.

## Color

- `--bg-page`: base aireada verde-azulada con gradientes suaves.
- `--bg-panel`: paneles blancos con transparencia ligera y blur.
- `--safe`: `#0f766e` para ruta segura y CTA principal.
- `--balanced`: `#2563eb` para balance y comparacion neutra.
- `--fast`: `#ea580c` para velocidad y urgencia controlada.
- `--low`: verde para riesgo bajo.
- `--medium`: amarillo para riesgo medio.
- `--high`: rojo para riesgo alto o error operativo.
- Alertas:
  `warning` usa fondo calido suave.
  `error` usa fondo rosado claro.
  `info` usa azul muy claro.

## Tipografia

- Titulares: `Segoe UI Variable`, `Aptos`, `Trebuchet MS`, sans-serif.
- Titulares compactos, tracking negativo, line-height corto.
- Texto general: 16px base, copy breve y respirado.
- Eyebrows: uppercase, 0.12em tracking, tono secundario.

## Espaciado

- Escala base:
  `8, 12, 16, 20, 24, 32, 40`.
- Shell principal:
  24px desktop, 14px mobile.
- Separacion entre paneles:
  16px a 24px.

## Radios y sombras

- Panel grande: 28px.
- Card y chips: 18px a 22px.
- Pills: 999px.
- Sombra principal:
  `0 22px 60px rgba(18,49,58,.12)`.
- Sombra secundaria:
  `0 14px 32px rgba(18,49,58,.08)`.

## Iconografia y semantica

- Origen: verde azulado.
- Destino: rojo ladrillo.
- Hotspot: naranja translcido.
- Ruta seleccionada: linea mas gruesa.
- Badges:
  `recommended`, `fastest`, `least_exposure`.

## Layout

- Desktop:
  split layout con mapa dominante y panel lateral.
- Mobile:
  mapa arriba y sheet de resultados debajo.
- Barra de busqueda sticky.
- CTA unico:
  `Planificar viaje`.

## Accesibilidad

- Mantener contraste AA minimo en textos y estados.
- No depender solo de color para diferenciar rutas.
- Todo control interactivo debe tener hit area amplia.
- Estados de error y warming deben incluir accion clara o texto recuperable.

## Motion

- Micro movimiento discreto:
  hover con `translateY(-1px)`.
- Fit de mapa suave.
- Evitar animaciones continuas o decorativas.
