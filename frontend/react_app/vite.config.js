import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
export default defineConfig({
    plugins: [react()],
    server: {
        host: "0.0.0.0",
        port: 3000,
        proxy: {
            "/health": "http://localhost:8000",
            "/readyz": "http://localhost:8000",
            "/metadata": "http://localhost:8000",
            "/places": "http://localhost:8000",
            "/recommendations": "http://localhost:8000",
            "/routes": "http://localhost:8000",
            "/system": "http://localhost:8000",
        },
    },
    test: {
        environment: "jsdom",
        globals: true,
        setupFiles: "./src/setupTests.ts",
    },
});
