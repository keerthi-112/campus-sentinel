import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In Docker the backend is reachable as http://backend:8000 (set BACKEND_URL).
const target = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": { target, changeOrigin: true },
      "/health": { target, changeOrigin: true },
    },
  },
});
