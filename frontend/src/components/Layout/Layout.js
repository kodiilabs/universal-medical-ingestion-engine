// ============================================================================
// Layout Component - Main application shell
// ============================================================================

import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Upload,
  FileText,
  Activity,
  GitBranch,
  AlertCircle,
  Settings,
  LayoutDashboard
} from 'lucide-react';
import './Layout.css';

const navItems = [
  { path: '/', icon: Upload, label: 'Upload' },
  { path: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/documents', icon: FileText, label: 'Documents' },
  { path: '/flow', icon: GitBranch, label: 'Process Flow' },
  { path: '/review', icon: AlertCircle, label: 'Review Queue' },
  { path: '/templates', icon: Settings, label: 'Templates' },
];

const Layout = ({ children }) => {
  const location = useLocation();

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <Activity size={24} />
            <span>MedIngest</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          {navItems.map(({ path, icon: Icon, label }) => (
            <NavLink
              key={path}
              to={path}
              className={({ isActive }) =>
                `nav-item ${isActive ? 'active' : ''}`
              }
            >
              <Icon size={20} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="version">v1.0.0</div>
        </div>
      </aside>

      <main className="main-content">
        {children}
      </main>
    </div>
  );
};

export default Layout;
