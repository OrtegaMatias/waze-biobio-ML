"""
Custom exceptions for the Waze Biobío ML API.
"""


class WazeBiobioException(Exception):
    """Base exception for all Waze Biobío errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class NoRouteFoundException(WazeBiobioException):
    """Raised when no route can be found between two points."""

    def __init__(self, origin: tuple, destination: tuple):
        message = (
            f"No se encontró una ruta entre origen ({origin[0]:.4f}, {origin[1]:.4f}) "
            f"y destino ({destination[0]:.4f}, {destination[1]:.4f}). "
            f"Verifica que ambos puntos estén dentro de la región del Biobío."
        )
        super().__init__(message, status_code=404)


class InvalidCoordinatesException(WazeBiobioException):
    """Raised when coordinates are invalid or out of bounds."""

    def __init__(self, lat: float, lon: float, reason: str = ""):
        message = f"Coordenadas inválidas: ({lat}, {lon})"
        if reason:
            message += f". {reason}"
        super().__init__(message, status_code=400)


class UserNotFoundException(WazeBiobioException):
    """Raised when a user ID is not found in the system."""

    def __init__(self, user_id: str):
        message = f"Usuario '{user_id}' no encontrado en el sistema."
        super().__init__(message, status_code=404)


class InvalidStrategyException(WazeBiobioException):
    """Raised when an invalid recommendation strategy is provided."""

    def __init__(self, strategy: str):
        message = (
            f"Estrategia '{strategy}' no válida. "
            f"Estrategias disponibles: 'ubcf', 'ibcf'"
        )
        super().__init__(message, status_code=400)


class DataNotLoadedException(WazeBiobioException):
    """Raised when required data has not been loaded."""

    def __init__(self, data_type: str):
        message = (
            f"Datos '{data_type}' no cargados. "
            f"Ejecuta /system/bootstrap primero."
        )
        super().__init__(message, status_code=503)


class ConfigurationException(WazeBiobioException):
    """Raised when there's a configuration error."""

    def __init__(self, message: str):
        super().__init__(f"Error de configuración: {message}", status_code=500)
