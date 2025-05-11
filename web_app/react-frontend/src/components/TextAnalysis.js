import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

const TextAnalysis = ({ modelLoaded }) => {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [useAdvanced, setUseAdvanced] = useState(true);

  useEffect(() => {
    if (result) {
      if (window.bootstrap) {
        const tabElements = document.querySelectorAll('#resultTabs button');
        tabElements.forEach(tab => {
          tab.addEventListener('click', event => {
            event.preventDefault();
            const bsTab = new window.bootstrap.Tab(tab);
            bsTab.show();
          });
        });
      }
    }
  }, [result]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!text.trim()) {
      setError('Παρακαλώ εισάγετε κείμενο για ανάλυση.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          text,
          use_advanced: useAdvanced
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Σφάλμα κατά την ανάλυση');
      }

      const data = await response.json();
      
      // Μετατροπή των chart JSON strings σε αντικείμενα
      if (data.bert && data.bert.chart) {
        data.bert.chartData = JSON.parse(data.bert.chart);
      }
      
      // Μετατροπή του γραφήματος συναισθημάτων αν υπάρχει
      if (data.advanced && data.advanced.emotions_chart) {
        data.advanced.emotions_chartData = JSON.parse(data.advanced.emotions_chart);
      }
      
      setResult(data);
    } catch (err) {
      setError(err.message || 'Προέκυψε σφάλμα κατά την ανάλυση του κειμένου.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getSentimentClass = (sentiment) => {
    switch (sentiment) {
      case 'Θετικό': return 'sentiment-positive';
      case 'Αρνητικό': return 'sentiment-negative';
      case 'Ουδέτερο': return 'sentiment-neutral';
      default: return '';
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'Θετικό': return 'bi bi-emoji-smile-fill';
      case 'Αρνητικό': return 'bi bi-emoji-frown-fill';
      case 'Ουδέτερο': return 'bi bi-emoji-neutral-fill';
      default: return 'bi bi-question-circle-fill';
    }
  };

  const getSentimentBadgeClass = (sentiment) => {
    switch (sentiment) {
      case 'Θετικό': return 'sentiment-badge-positive';
      case 'Αρνητικό': return 'sentiment-badge-negative';
      case 'Ουδέτερο': return 'sentiment-badge-neutral';
      default: return 'bg-secondary';
    }
  };

  const getEmotionIcon = (emotion) => {
    switch (emotion) {
      case 'Χαρά': return 'bi bi-emoji-laughing-fill';
      case 'Εμπιστοσύνη': return 'bi bi-shield-check';
      case 'Φόβος': return 'bi bi-emoji-dizzy-fill';
      case 'Έκπληξη': return 'bi bi-emoji-surprise-fill';
      case 'Λύπη': return 'bi bi-emoji-tear-fill';
      case 'Αποστροφή': return 'bi bi-emoji-expressionless-fill';
      case 'Θυμός': return 'bi bi-emoji-angry-fill';
      case 'Προσδοκία': return 'bi bi-hourglass-split';
      default: return 'bi bi-question-circle-fill';
    }
  };

  const getEmotionColor = (emotion) => {
    switch (emotion) {
      case 'Χαρά': return '#FFEB3B';
      case 'Εμπιστοσύνη': return '#4CAF50';
      case 'Φόβος': return '#9C27B0';
      case 'Έκπληξη': return '#FF9800';
      case 'Λύπη': return '#2196F3';
      case 'Αποστροφή': return '#795548';
      case 'Θυμός': return '#F44336';
      case 'Προσδοκία': return '#03A9F4';
      default: return '#9E9E9E';
    }
  };

  return (
    <div className="row">
      <div className="col-lg-10 mx-auto">
        <div className="card shadow-sm mb-4">
          <div className="card-header bg-primary text-white">
            <h5 className="card-title mb-0">
              <i className="bi bi-chat-text-fill me-2"></i>
              Ανάλυση Συναισθήματος Κειμένου
            </h5>
          </div>
          <div className="card-body">
            {!modelLoaded && (
              <div className="alert alert-warning" role="alert">
                <i className="bi bi-exclamation-triangle-fill me-2"></i>
                Το μοντέλο ανάλυσης συναισθημάτων δεν είναι διαθέσιμο αυτή τη στιγμή.
              </div>
            )}

            <form onSubmit={handleSubmit}>
              <div className="mb-3">
                <label htmlFor="textInput" className="form-label">Εισάγετε κείμενο για ανάλυση:</label>
                <textarea 
                  id="textInput"
                  className="form-control" 
                  rows="5"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Εισάγετε το κείμενο που θέλετε να αναλύσετε..."
                  disabled={!modelLoaded || isAnalyzing}
                ></textarea>
              </div>
              <div className="mb-3 form-check">
                <input
                  type="checkbox"
                  className="form-check-input"
                  id="useAdvancedCheck"
                  checked={useAdvanced}
                  onChange={(e) => setUseAdvanced(e.target.checked)}
                  disabled={!modelLoaded || isAnalyzing}
                />
                <label className="form-check-label" htmlFor="useAdvancedCheck">
                  <i className="bi bi-gear-fill me-1"></i>
                  Χρήση προηγμένης ανάλυσης (VADER, NRCLex, LSA)
                </label>
              </div>
              <div className="d-grid">
                <button 
                  type="submit" 
                  className="btn btn-primary"
                  disabled={!modelLoaded || isAnalyzing || !text.trim()}
                >
                  {isAnalyzing ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                      Ανάλυση...
                    </>
                  ) : (
                    <>
                      <i className="bi bi-search me-2"></i>
                      Ανάλυση Συναισθήματος
                    </>
                  )}
                </button>
              </div>
            </form>

            {error && (
              <div className="alert alert-danger mt-4" role="alert">
                <i className="bi bi-exclamation-circle-fill me-2"></i>
                {error}
              </div>
            )}

            {result && (
              <div className="mt-4">
                <ul className="nav nav-tabs" id="resultTabs" role="tablist">
                  <li className="nav-item" role="presentation">
                    <button 
                      className="nav-link active" 
                      id="bert-tab" 
                      data-bs-toggle="tab" 
                      data-bs-target="#bert" 
                      type="button" 
                      role="tab" 
                      aria-controls="bert" 
                      aria-selected="true"
                    >
                      <i className="bi bi-robot me-1"></i>
                      BERT Ανάλυση
                    </button>
                  </li>
                  {result.advanced && (
                    <li className="nav-item" role="presentation">
                      <button 
                        className="nav-link" 
                        id="advanced-tab" 
                        data-bs-toggle="tab" 
                        data-bs-target="#advanced" 
                        type="button" 
                        role="tab" 
                        aria-controls="advanced" 
                        aria-selected="false"
                      >
                        <i className="bi bi-graph-up-arrow me-1"></i>
                        Προηγμένη Ανάλυση
                      </button>
                    </li>
                  )}
                </ul>
                
                <div className="tab-content mt-3" id="resultTabContent">
                  {/* BERT Ανάλυση Tab */}
                  <div 
                    className="tab-pane fade show active" 
                    id="bert" 
                    role="tabpanel" 
                    aria-labelledby="bert-tab"
                  >
                    <div className="card mb-4">
                      <div className="card-body">
                        {result.bert && (
                          <div className="d-flex align-items-center mb-3">
                            <i className={`${getSentimentIcon(result.bert.sentiment)} fs-1 me-3 ${getSentimentClass(result.bert.sentiment)}`}></i>
                            <div>
                              <h5 className="mb-0">Συναίσθημα: 
                                <span className={`ms-2 ${getSentimentClass(result.bert.sentiment)}`}>
                                  {result.bert.sentiment}
                                </span>
                              </h5>
                              <p className="mb-0">
                                Πιθανότητα: <strong>{result.bert.probability}%</strong>
                              </p>
                            </div>
                          </div>
                        )}
                        
                        <div className="card bg-light">
                          <div className="card-body">
                            <h6 className="card-subtitle mb-2 text-muted">Αναλυθέν Κείμενο:</h6>
                            <p className="card-text">"{result.text}"</p>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    {result.bert && result.bert.chartData && (
                      <div className="chart-container mt-4">
                        <Plot
                          data={result.bert.chartData.data}
                          layout={result.bert.chartData.layout}
                          config={{ responsive: true }}
                          style={{ width: '100%', height: '100%' }}
                        />
                      </div>
                    )}
                  </div>
                  
                  {/* Προηγμένη Ανάλυση Tab */}
                  <div 
                    className="tab-pane fade" 
                    id="advanced" 
                    role="tabpanel" 
                    aria-labelledby="advanced-tab"
                  >
                    {result.advanced && (
                      <>
                        {/* VADER Analysis */}
                        <div className="card mb-4">
                          <div className="card-header bg-info text-white">
                            <h5 className="mb-0">
                              <i className="bi bi-bar-chart-fill me-2"></i>
                              Ανάλυση VADER
                            </h5>
                          </div>
                          <div className="card-body">
                            {result.advanced.vader && (
                              <>
                                <div className="row">
                                  <div className="col-md-6">
                                    <div className="alert alert-light">
                                      <h6>Συναίσθημα: <span className={getSentimentClass(result.advanced.vader.sentiment)}>{result.advanced.vader.sentiment}</span></h6>
                                      <p className="mb-1">Compound Score: <strong>{result.advanced.vader.compound_scaled}%</strong></p>
                                      <div className="progress mb-3" style={{ height: '20px' }}>
                                        <div 
                                          className="progress-bar bg-danger" 
                                          role="progressbar" 
                                          style={{ width: `${result.advanced.vader.neg_percent}%` }} 
                                          aria-valuenow={result.advanced.vader.neg_percent} 
                                          aria-valuemin="0" 
                                          aria-valuemax="100"
                                        >
                                          {result.advanced.vader.neg_percent}%
                                        </div>
                                        <div 
                                          className="progress-bar bg-warning" 
                                          role="progressbar" 
                                          style={{ width: `${result.advanced.vader.neu_percent}%` }} 
                                          aria-valuenow={result.advanced.vader.neu_percent} 
                                          aria-valuemin="0" 
                                          aria-valuemax="100"
                                        >
                                          {result.advanced.vader.neu_percent}%
                                        </div>
                                        <div 
                                          className="progress-bar bg-success" 
                                          role="progressbar" 
                                          style={{ width: `${result.advanced.vader.pos_percent}%` }} 
                                          aria-valuenow={result.advanced.vader.pos_percent} 
                                          aria-valuemin="0" 
                                          aria-valuemax="100"
                                        >
                                          {result.advanced.vader.pos_percent}%
                                        </div>
                                      </div>
                                      <div className="d-flex justify-content-between">
                                        <small>Αρνητικό: {result.advanced.vader.neg_percent}%</small>
                                        <small>Ουδέτερο: {result.advanced.vader.neu_percent}%</small>
                                        <small>Θετικό: {result.advanced.vader.pos_percent}%</small>
                                      </div>
                                    </div>
                                  </div>
                                  <div className="col-md-6">
                                    <div className="card bg-light">
                                      <div className="card-body">
                                        <h6>Τι είναι το VADER;</h6>
                                        <p>Το VADER (Valence Aware Dictionary and sEntiment Reasoner) είναι ένας λεξικογραφικός και βασισμένος σε κανόνες αναλυτής συναισθημάτων ειδικά προσαρμοσμένος για συναισθήματα που εκφράζονται στα μέσα κοινωνικής δικτύωσης.</p>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              </>
                            )}
                          </div>
                        </div>

                        {/* NRCLex Emotions Analysis */}
                        <div className="card mb-4">
                          <div className="card-header bg-success text-white">
                            <h5 className="mb-0">
                              <i className="bi bi-emoji-smile me-2"></i>
                              Ανάλυση Συναισθημάτων NRCLex
                            </h5>
                          </div>
                          <div className="card-body">
                            {result.advanced.emotions && (
                              <>
                                <div className="row align-items-center mb-4">
                                  <div className="col-md-6">
                                    <div className="card mb-3">
                                      <div className="card-body text-center">
                                        <h4 className="mb-1">Επικρατέστερο Συναίσθημα</h4>
                                        {result.advanced.emotions.dominant_emotion !== "Δεν εντοπίστηκε" ? (
                                          <>
                                            <div className="display-1 mb-2">
                                              <i 
                                                className={getEmotionIcon(result.advanced.emotions.dominant_emotion)} 
                                                style={{ color: getEmotionColor(result.advanced.emotions.dominant_emotion) }}
                                              ></i>
                                            </div>
                                            <h3>{result.advanced.emotions.dominant_emotion}</h3>
                                            <div className="badge bg-primary fs-6">{result.advanced.emotions.dominant_score}%</div>
                                          </>
                                        ) : (
                                          <div className="alert alert-warning">
                                            Δεν ήταν δυνατό να εντοπιστεί κάποιο συγκεκριμένο συναίσθημα.
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                  <div className="col-md-6">
                                    <div className="card bg-light">
                                      <div className="card-body">
                                        <h6>Τι είναι το NRCLex;</h6>
                                        <p>Η βιβλιοθήκη NRCLex χρησιμοποιεί το λεξικό NRC Emotion Lexicon για να αναλύσει κείμενο και να εντοπίσει συγκεκριμένα συναισθήματα όπως χαρά, φόβος, θυμός, κλπ.</p>
                                      </div>
                                    </div>
                                  </div>
                                </div>

                                {result.advanced.emotions_chartData ? (
                                  <div className="chart-container mt-4">
                                    <Plot
                                      data={result.advanced.emotions_chartData.data}
                                      layout={result.advanced.emotions_chartData.layout}
                                      config={{ responsive: true }}
                                      style={{ width: '100%', height: '100%' }}
                                    />
                                  </div>
                                ) : (
                                  <div className="row">
                                    {Object.entries(result.advanced.emotions || {})
                                      .filter(([key]) => !['dominant_emotion', 'dominant_score'].includes(key))
                                      .sort((a, b) => b[1] - a[1])
                                      .map(([emotion, score], index) => (
                                        <div className="col-md-3 col-6 mb-3" key={index}>
                                          <div className="card h-100">
                                            <div className="card-body text-center p-2">
                                              <h3>
                                                <i 
                                                  className={getEmotionIcon(emotion)} 
                                                  style={{ color: getEmotionColor(emotion) }}
                                                ></i>
                                              </h3>
                                              <h6>{emotion}</h6>
                                              <div className="mt-2">
                                                <div className="progress" style={{ height: '8px' }}>
                                                  <div 
                                                    className="progress-bar" 
                                                    role="progressbar" 
                                                    style={{ 
                                                      width: `${score}%`,
                                                      backgroundColor: getEmotionColor(emotion)
                                                    }} 
                                                    aria-valuenow={score} 
                                                    aria-valuemin="0" 
                                                    aria-valuemax="100"
                                                  />
                                                </div>
                                                <small className="mt-1 d-block">{score}%</small>
                                              </div>
                                            </div>
                                          </div>
                                        </div>
                                      ))}
                                  </div>
                                )}
                              </>
                            )}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TextAnalysis; 