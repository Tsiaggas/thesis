import React, { useState } from 'react';
import Plot from 'react-plotly.js';

const BatchAnalysis = ({ modelLoaded }) => {
  const [file, setFile] = useState(null);
  const [textColumn, setTextColumn] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [useAdvanced, setUseAdvanced] = useState(true);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.name.endsWith('.csv')) {
      setFile(selectedFile);
      setError(null);
    } else if (selectedFile) {
      setFile(null);
      setError('Παρακαλώ επιλέξτε ένα αρχείο CSV.');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Παρακαλώ επιλέξτε ένα αρχείο CSV για ανάλυση.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('use_advanced', useAdvanced.toString());
      
      if (textColumn.trim()) {
        formData.append('text_column', textColumn.trim());
      }

      const response = await fetch('http://localhost:5000/api/analyze_batch', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Σφάλμα κατά την ανάλυση');
      }

      const data = await response.json();
      console.log("Data received from server:", JSON.stringify(data, null, 2)); // DEBUG: Log raw data
      
      // Μετατροπή των chart JSON strings σε αντικείμενα
      if (data.bert && data.bert.chart) {
        data.bert.chartData = JSON.parse(data.bert.chart);
      }
      
      // Μετατροπή των γραφημάτων προηγμένης ανάλυσης αν υπάρχουν
      if (data.advanced) {
        if (data.advanced.vader_chart) {
          data.advanced.vader_chartData = JSON.parse(data.advanced.vader_chart);
        }
        if (data.advanced.emotions_chart) {
          data.advanced.emotions_chartData = JSON.parse(data.advanced.emotions_chart);
        }
      }
      
      setResult(data);
      console.log('Result state updated:', data); // DEBUG: Log updated state
    } catch (err) {
      setError(err.message || 'Προέκυψε σφάλμα κατά την ανάλυση του αρχείου.');
      console.error("Analysis error:", err); // DEBUG: Log error object
    } finally {
      setIsAnalyzing(false);
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
              <i className="bi bi-file-earmark-text-fill me-2"></i>
              Ανάλυση Συναισθήματος CSV
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
                <label htmlFor="csvFile" className="form-label">Επιλέξτε αρχείο CSV:</label>
                <input 
                  id="csvFile"
                  type="file" 
                  className="form-control"
                  accept=".csv"
                  onChange={handleFileChange}
                  disabled={!modelLoaded || isAnalyzing}
                />
                <div className="form-text">
                  Το αρχείο πρέπει να είναι σε μορφή CSV και να περιέχει στήλη με κείμενο για ανάλυση.
                </div>
              </div>
              
              <div className="mb-3">
                <label htmlFor="textColumn" className="form-label">Όνομα στήλης κειμένου (προαιρετικό):</label>
                <input 
                  id="textColumn"
                  type="text" 
                  className="form-control"
                  value={textColumn}
                  onChange={(e) => setTextColumn(e.target.value)}
                  placeholder="π.χ. text, comment, review"
                  disabled={!modelLoaded || isAnalyzing}
                />
                <div className="form-text">
                  Εάν δεν οριστεί, θα χρησιμοποιηθεί η πρώτη στήλη του αρχείου.
                </div>
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
                  disabled={!modelLoaded || isAnalyzing || !file}
                >
                  {isAnalyzing ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                      Ανάλυση...
                    </>
                  ) : (
                    <>
                      <i className="bi bi-search me-2"></i>
                      Ανάλυση Αρχείου
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
                <ul className="nav nav-tabs mb-3" id="resultTabs" role="tablist">
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
                  {result?.advanced_analysis_available && result?.advanced && (
                    <>
                      <li className="nav-item" role="presentation">
                        <button 
                          className="nav-link" 
                          id="vader-tab" 
                          data-bs-toggle="tab" 
                          data-bs-target="#vader" 
                          type="button" 
                          role="tab" 
                          aria-controls="vader" 
                          aria-selected="false"
                        >
                          <i className="bi bi-bar-chart-fill me-1"></i>
                          VADER
                        </button>
                      </li>
                      <li className="nav-item" role="presentation">
                        <button 
                          className="nav-link" 
                          id="emotions-tab" 
                          data-bs-toggle="tab" 
                          data-bs-target="#emotions" 
                          type="button" 
                          role="tab" 
                          aria-controls="emotions" 
                          aria-selected="false"
                        >
                          <i className="bi bi-emoji-smile me-1"></i>
                          Συναισθήματα
                        </button>
                      </li>
                      <li className="nav-item" role="presentation">
                        <button 
                          className="nav-link" 
                          id="lsa-tab" 
                          data-bs-toggle="tab" 
                          data-bs-target="#lsa" 
                          type="button" 
                          role="tab" 
                          aria-controls="lsa" 
                          aria-selected="false"
                        >
                          <i className="bi bi-grid-3x3-gap-fill me-1"></i>
                          LSA
                        </button>
                      </li>
                    </>
                  )}
                </ul>
                
                <div className="tab-content" id="resultTabsContent">
                  {/* BERT Tab */}
                  <div 
                    className="tab-pane fade show active" 
                    id="bert" 
                    role="tabpanel" 
                    aria-labelledby="bert-tab"
                  >
                    <div className="row mb-4">
                      <div className="col-md-4">
                        <div className="card">
                          <div className="card-body text-center">
                            <h6 className="card-subtitle mb-2 text-muted">Συνολικές Εγγραφές</h6>
                            <p className="card-text display-6">{result?.count}</p>
                          </div>
                        </div>
                      </div>
                      <div className="col-md-8">
                        <div className="card">
                          <div className="card-body">
                            <h6 className="card-subtitle mb-2 text-muted">Κατανομή Συναισθημάτων</h6>
                            <div className="d-flex justify-content-around">
                              <div className="text-center">
                                <span className="badge sentiment-badge-positive fs-6 mb-1">Θετικά</span>
                                <p className="card-text h4">{result?.sentiment_counts?.Θετικό || 0}</p>
                              </div>
                              <div className="text-center">
                                <span className="badge sentiment-badge-negative fs-6 mb-1">Αρνητικά</span>
                                <p className="card-text h4">{result?.sentiment_counts?.Αρνητικό || 0}</p>
                              </div>
                              <div className="text-center">
                                <span className="badge sentiment-badge-neutral fs-6 mb-1">Ουδέτερα</span>
                                <p className="card-text h4">{result?.sentiment_counts?.Ουδέτερο || 0}</p>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    {result?.chartData && (
                      <div className="chart-container mt-4">
                        <Plot
                          data={result?.chartData.data}
                          layout={result?.chartData.layout}
                          config={{ responsive: true }}
                          style={{ width: '100%', height: '400px' }}
                        />
                      </div>
                    )}
                    
                    <div className="mt-4">
                      <h6 className="border-bottom pb-2">Λεπτομέρειες Ανάλυσης</h6>
                      
                      <div className="table-responsive">
                        <table className="table table-hover">
                          <thead>
                            <tr>
                              <th>#</th>
                              <th>Κείμενο</th>
                              <th>Συναίσθημα</th>
                              <th>Πιθανότητα</th>
                            </tr>
                          </thead>
                          <tbody>
                            {result?.results?.map((item, index) => (
                              <tr key={index}>
                                <td>{item.row_number}</td>
                                <td>{item.text}</td>
                                <td>
                                  <span className={`badge ${getSentimentBadgeClass(item.sentiment)}`}>
                                    {item.sentiment}
                                  </span>
                                </td>
                                <td>{item.probability}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                  
                  {/* VADER Tab */}
                  {result?.advanced_analysis_available && result?.advanced && (
                    <div 
                      className="tab-pane fade" 
                      id="vader" 
                      role="tabpanel" 
                      aria-labelledby="vader-tab"
                    >
                      <div className="card mb-4">
                        <div className="card-body">
                          <h5 className="card-title mb-3">Κατανομή Συναισθημάτων με VADER</h5>
                          
                          <div className="row mb-4">
                            <div className="col-md-6">
                              <div className="card">
                                <div className="card-body">
                                  <h6 className="card-subtitle mb-2 text-muted">Κατανομή</h6>
                                  <div className="d-flex justify-content-around">
                                    <div className="text-center">
                                      <span className="badge sentiment-badge-positive fs-6 mb-1">Θετικά</span>
                                      <p className="card-text h4">{result.advanced.vader_sentiments.Θετικό || 0}</p>
                                    </div>
                                    <div className="text-center">
                                      <span className="badge sentiment-badge-negative fs-6 mb-1">Αρνητικά</span>
                                      <p className="card-text h4">{result.advanced.vader_sentiments.Αρνητικό || 0}</p>
                                    </div>
                                    <div className="text-center">
                                      <span className="badge sentiment-badge-neutral fs-6 mb-1">Ουδέτερα</span>
                                      <p className="card-text h4">{result.advanced.vader_sentiments.Ουδέτερο || 0}</p>
                                    </div>
                                  </div>
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
                          
                          {result.advanced.vader_chartData && (
                            <div className="chart-container mt-4">
                              <Plot
                                data={result.advanced.vader_chartData.data}
                                layout={result.advanced.vader_chartData.layout}
                                config={{ responsive: true }}
                                style={{ width: '100%', height: '400px' }}
                              />
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* Συναισθήματα Tab */}
                  {result?.advanced_analysis_available && result?.advanced && (
                    <div 
                      className="tab-pane fade" 
                      id="emotions" 
                      role="tabpanel" 
                      aria-labelledby="emotions-tab"
                    >
                      <div className="card mb-4">
                        <div className="card-body">
                          <h5 className="card-title mb-3">Ανάλυση Συγκεκριμένων Συναισθημάτων (NRCLex)</h5>
                          
                          <div className="row mb-4">
                            <div className="col-md-6">
                              {result.advanced.emotions_chartData && (
                                <div className="chart-container">
                                  <Plot
                                    data={result.advanced.emotions_chartData.data}
                                    layout={result.advanced.emotions_chartData.layout}
                                    config={{ responsive: true }}
                                    style={{ width: '100%', height: '400px' }}
                                  />
                                </div>
                              )}
                            </div>
                            <div className="col-md-6">
                              <div className="card">
                                <div className="card-body">
                                  <h6 className="card-subtitle mb-3">Συχνότερα Συναισθήματα</h6>
                                  
                                  {result.advanced.emotion_counts && Object.entries(result.advanced.emotion_counts)
                                    .sort((a, b) => b[1] - a[1])
                                    .slice(0, 5)
                                    .map(([emotion, count], index) => (
                                      <div key={index} className="mb-2">
                                        <div className="d-flex justify-content-between align-items-center">
                                          <span style={{ color: getEmotionColor(emotion) }}>{emotion}</span>
                                          <span className="badge bg-secondary">{count}</span>
                                        </div>
                                        <div className="progress">
                                          <div 
                                            className="progress-bar" 
                                            role="progressbar" 
                                            style={{ 
                                              width: `${(count / Math.max(...Object.values(result.advanced.emotion_counts))) * 100}%`,
                                              backgroundColor: getEmotionColor(emotion) 
                                            }}
                                            aria-valuemin="0" 
                                            aria-valuemax="100"
                                          ></div>
                                        </div>
                                      </div>
                                    ))
                                  }
                                </div>
                              </div>
                              
                              <div className="card bg-light mt-3">
                                <div className="card-body">
                                  <h6>Τι είναι το NRCLex;</h6>
                                  <p>Το NRCLex είναι ένα λεξικό που βασίζεται στο National Research Council Canada Emotion Lexicon. Εντοπίζει συγκεκριμένα συναισθήματα όπως χαρά, εμπιστοσύνη, φόβο, έκπληξη κτλ. σε κείμενα.</p>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* LSA Tab */}
                  {result?.advanced_analysis_available && result?.advanced && (
                    <div 
                      className="tab-pane fade" 
                      id="lsa" 
                      role="tabpanel" 
                      aria-labelledby="lsa-tab"
                    >
                      <div className="card mb-4">
                        <div className="card-body">
                          <h5 className="card-title mb-3">Λανθάνουσα Σημασιολογική Ανάλυση (LSA)</h5>
                          
                          {result.advanced.lsa_topics && result.advanced.lsa_topics.length > 0 ? (
                            <div className="row">
                              {result.advanced.lsa_topics.map((topic, index) => (
                                <div key={index} className="col-md-6 mb-4">
                                  <div className="card">
                                    <div className="card-header bg-light">
                                      <h6 className="mb-0">Θέμα {topic.topic_id}</h6>
                                    </div>
                                    <div className="card-body">
                                      <h6 className="card-subtitle mb-2 text-muted">Κορυφαίες Λέξεις</h6>
                                      
                                      <div className="table-responsive">
                                        <table className="table table-sm">
                                          <thead>
                                            <tr>
                                              <th>Λέξη</th>
                                              <th>Βάρος</th>
                                            </tr>
                                          </thead>
                                          <tbody>
                                            {topic.top_words.map((word, wordIndex) => (
                                              <tr key={wordIndex}>
                                                <td>{word}</td>
                                                <td>
                                                  <div className="progress">
                                                    <div 
                                                      className="progress-bar bg-info" 
                                                      role="progressbar" 
                                                      style={{ width: `${Math.abs(topic.weights[wordIndex] * 100)}%` }} 
                                                      aria-valuemin="0" 
                                                      aria-valuemax="100"
                                                    ></div>
                                                  </div>
                                                </td>
                                              </tr>
                                            ))}
                                          </tbody>
                                        </table>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="alert alert-info">
                              <i className="bi bi-info-circle-fill me-2"></i>
                              Δεν υπάρχουν αρκετά δεδομένα για την εκτέλεση LSA. Χρειάζονται περισσότερα κείμενα για την εξαγωγή θεμάτων.
                            </div>
                          )}
                          
                          <div className="card bg-light mt-3">
                            <div className="card-body">
                              <h6>Τι είναι η Λανθάνουσα Σημασιολογική Ανάλυση (LSA);</h6>
                              <p>Η LSA είναι μια τεχνική επεξεργασίας φυσικής γλώσσας που αναλύει τις σχέσεις μεταξύ ενός συνόλου εγγράφων και των όρων που περιέχουν. Αποκαλύπτει κρυμμένα (λανθάνοντα) θέματα στα κείμενα και βρίσκει ομάδες λέξεων που συνδέονται σημασιολογικά.</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BatchAnalysis; 