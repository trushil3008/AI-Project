/**
 * Navbar.jsx — Navigation Bar Component
 *
 * Sticky top navigation with links to Upload and Dashboard pages.
 * Highlights the current active route.
 */

import { Link, useLocation } from "react-router-dom";
import { FiUpload, FiGrid, FiShield, FiCpu } from "react-icons/fi";

export default function Navbar() {
  const location = useLocation();

  // Check if a path is the current route
  const isActive = (path) => {
    if (path === "/") return location.pathname === "/";
    return location.pathname.startsWith(path);
  };

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        {/* Logo */}
        <Link to="/" className="navbar-logo">
          <span className="logo-icon">
            <FiShield />
          </span>
          <span className="logo-text">PlagDetect AI</span>
        </Link>

        {/* Navigation Links */}
        <div className="navbar-links">
          <Link to="/" className={isActive("/") ? "active" : ""}>
            <FiUpload size={16} />
            Upload
          </Link>
          <Link to="/ai-detect" className={isActive("/ai-detect") ? "active" : ""}>
            <FiCpu size={16} />
            AI Detect
          </Link>
          <Link to="/dashboard" className={isActive("/dashboard") ? "active" : ""}>
            <FiGrid size={16} />
            Dashboard
          </Link>
        </div>
      </div>
    </nav>
  );
}
