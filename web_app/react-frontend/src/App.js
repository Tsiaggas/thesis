import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import TextAnalysis from './components/TextAnalysis';
import BatchAnalysis from './components/BatchAnalysis';

function App() {
  const [modelStatus, setModelStatus] = useState({
    loaded: false,
    checking: true,
    error: null
  });

  useEffect(() => {
    // Έλεγχος κατάστασης του μοντέλου στο backend
    const checkModelStatus = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/status');
        const data = await response.json();
        
        setModelStatus({
          loaded: data.model_loaded,
          checking: false,
          error: null
        });
      } catch (error) {
        setModelStatus({
          loaded: false,
          checking: false,
          error: 'Αδυναμία σύνδεσης με τον server. Παρακαλώ βεβαιωθείτε ότι ο server είναι ενεργός.'
        });
      }
    };

    checkModelStatus();
  }, []);

  return (
    <Router>
      <div className="d-flex flex-column min-vh-100">
        <header className="app-header p-3 mb-4">
          <div className="container">
            <div className="d-flex flex-wrap align-items-center justify-content-center justify-content-lg-start">
              <span className="d-flex align-items-center mb-2 mb-lg-0 text-white text-decoration-none me-4">
                <i className="bi bi-emoji-smile-fill fs-3 me-2"></i>
                <h4 className="mb-0">Ανάλυση Συναισθημάτων</h4>
              </span>

              <ul className="nav col-12 col-lg-auto me-lg-auto mb-2 justify-content-center mb-md-0">
                <li>
                  <Link to="/" className="nav-link px-2 text-white">Ανάλυση Κειμένου</Link>
                </li>
                <li>
                  <Link to="/batch" className="nav-link px-2 text-white">Ανάλυση CSV</Link>
                </li>
              </ul>

              <div className="text-end">
                {modelStatus.checking ? (
                  <span className="badge bg-info">Έλεγχος μοντέλου...</span>
                ) : modelStatus.loaded ? (
                  <span className="badge bg-success">Μοντέλο: Ενεργό</span>
                ) : (
                  <span className="badge bg-danger">Μοντέλο: Μη διαθέσιμο</span>
                )}
              </div>
            </div>
          </div>
        </header>
        
        <main className="flex-grow-1">
          <div className="container py-4">
            {modelStatus.error && (
              <div className="alert alert-danger" role="alert">
                <i className="bi bi-exclamation-triangle-fill me-2"></i>
                {modelStatus.error}
              </div>
            )}
            
            <Routes>
              <Route path="/" element={<TextAnalysis modelLoaded={modelStatus.loaded} />} />
              <Route path="/batch" element={<BatchAnalysis modelLoaded={modelStatus.loaded} />} />
            </Routes>
          </div>
        </main>
        
        <footer className="footer mt-auto py-3 bg-dark text-white">
          <div className="container text-center">
            <span>© 2025 Εφαρμογή Ανάλυσης Συναισθημάτων</span>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App; 