/**
 * App.jsx — Main Application Component
 *
 * Sets up React Router with four routes:
 * - /           → Upload Page (compare new files)
 * - /ai-detect  → AI Detection Page (check if code is AI-generated)
 * - /results/:id → Results Page (view comparison results)
 * - /dashboard  → Dashboard (view past submissions)
 */

import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import UploadPage from "./pages/UploadPage";
import AIDetectPage from "./pages/AIDetectPage";
import ResultsPage from "./pages/ResultsPage";
import DashboardPage from "./pages/DashboardPage";

export default function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/ai-detect" element={<AIDetectPage />} />
        <Route path="/results/:id" element={<ResultsPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
      </Routes>
    </Router>
  );
}

