// ============================================================================
// App.js - Main Application with Routing
// ============================================================================

import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Upload from './pages/Upload';
import Dashboard from './pages/Dashboard';
import Documents from './pages/Documents';
import DocumentDetail from './pages/DocumentDetail';
import ProcessFlow from './pages/ProcessFlow';
import ReviewQueue from './pages/ReviewQueue';
import Templates from './pages/Templates';

import './styles/design-system.css';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Upload />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/documents/:jobId" element={<DocumentDetail />} />
          <Route path="/flow" element={<ProcessFlow />} />
          <Route path="/flow/:jobId" element={<ProcessFlow />} />
          <Route path="/review" element={<ReviewQueue />} />
          <Route path="/templates" element={<Templates />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
