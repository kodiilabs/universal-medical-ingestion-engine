import React, { useState, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import './DocumentViewer.css';

// Set up PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;

function DocumentViewer({ file, extractedData }) {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(1.0);
  const [isImage, setIsImage] = useState(false);

  useEffect(() => {
    if (file) {
      const ext = file.name?.toLowerCase().split('.').pop();
      setIsImage(['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(ext));
      setPageNumber(1);
    }
  }, [file]);

  function onDocumentLoadSuccess({ numPages }) {
    setNumPages(numPages);
  }

  if (!file) {
    return (
      <div className="document-viewer empty">
        <div className="empty-state">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
          </svg>
          <p>Select a document to view</p>
        </div>
      </div>
    );
  }

  return (
    <div className="document-viewer">
      <div className="viewer-toolbar">
        <div className="toolbar-left">
          <span className="file-name">{file.name}</span>
        </div>
        <div className="toolbar-center">
          {!isImage && numPages && (
            <>
              <button
                onClick={() => setPageNumber(Math.max(1, pageNumber - 1))}
                disabled={pageNumber <= 1}
              >
                &lt;
              </button>
              <span>
                Page {pageNumber} of {numPages}
              </span>
              <button
                onClick={() => setPageNumber(Math.min(numPages, pageNumber + 1))}
                disabled={pageNumber >= numPages}
              >
                &gt;
              </button>
            </>
          )}
        </div>
        <div className="toolbar-right">
          <button onClick={() => setScale(Math.max(0.5, scale - 0.1))}>-</button>
          <span>{Math.round(scale * 100)}%</span>
          <button onClick={() => setScale(Math.min(2, scale + 0.1))}>+</button>
        </div>
      </div>

      <div className="viewer-content">
        {isImage ? (
          <img
            src={file.path}
            alt={file.name}
            style={{ transform: `scale(${scale})`, transformOrigin: 'top center' }}
          />
        ) : (
          <Document
            file={file.path}
            onLoadSuccess={onDocumentLoadSuccess}
            loading={<div className="loading">Loading PDF...</div>}
            error={<div className="error">Failed to load PDF</div>}
          >
            <Page
              pageNumber={pageNumber}
              scale={scale}
              renderTextLayer={true}
              renderAnnotationLayer={true}
            />
          </Document>
        )}
      </div>
    </div>
  );
}

export default DocumentViewer;
