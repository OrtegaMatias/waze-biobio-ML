import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { DemoPage } from "./pages/DemoPage";
import { PlanPage } from "./pages/PlanPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<PlanPage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
