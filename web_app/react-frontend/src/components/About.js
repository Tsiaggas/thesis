import React from 'react';

const About = () => {
  return (
    <div className="row">
      <div className="col-lg-10 mx-auto">
        <div className="card shadow-sm mb-4">
          <div className="card-header bg-primary text-white">
            <h5 className="card-title mb-0">
              <i className="bi bi-info-circle-fill me-2"></i>
              Σχετικά με την Εφαρμογή
            </h5>
          </div>
          <div className="card-body">
            <h4>Ανάλυση Συναισθημάτων σε Κείμενα Πελατών</h4>
            
            <div className="mb-4">
              <h5>Περίληψη</h5>
              <p>
                Το συναισθηματικό περιεχόμενο που μπορεί να έχουν οι κριτικές καθώς και τα μηνύματα των πελατών, 
                έχει πολύ σημαντικό αντίκτυπο στις αποφάσεις, τις αντιλήψεις αλλά και της συνολικής εμπειρίας ενός πελάτη. 
                Η παρούσα εργασία διερευνά τρόπους όπου θα εντοπίζετε το συναίσθημα στις αλληλοεπιδράσεις των πελατών, 
                και πιο συγκεκριμένα στις κριτικές και τα μηνύματα αυτών.
              </p>
            </div>
            
            <div className="mb-4">
              <h5>Χαρακτηριστικά Εφαρμογής</h5>
              <ul>
                <li>Ανάλυση συναισθήματος σε μεμονωμένα κείμενα</li>
                <li>Μαζική ανάλυση κειμένων από αρχεία CSV</li>
                <li>Χρήση fine-tuned BERT μοντέλου 3 κλάσεων (Θετικό/Αρνητικό/Ουδέτερο)</li>
                <li>Οπτικοποίηση των αποτελεσμάτων με διαδραστικά γραφήματα</li>
              </ul>
            </div>
            
            <div className="mb-4">
              <h5>Τεχνολογίες</h5>
              <div className="row">
                <div className="col-md-6">
                  <div className="card mb-3">
                    <div className="card-header">Backend</div>
                    <div className="card-body">
                      <ul className="list-unstyled">
                        <li><i className="bi bi-check-circle-fill text-success me-2"></i>Python</li>
                        <li><i className="bi bi-check-circle-fill text-success me-2"></i>Flask</li>
                        <li><i className="bi bi-check-circle-fill text-success me-2"></i>PyTorch</li>
                        <li><i className="bi bi-check-circle-fill text-success me-2"></i>Transformers (HuggingFace)</li>
                        <li><i className="bi bi-check-circle-fill text-success me-2"></i>Pandas</li>
                      </ul>
                    </div>
                  </div>
                </div>
                <div className="col-md-6">
                  <div className="card mb-3">
                    <div className="card-header">Frontend</div>
                    <div className="card-body">
                      <ul className="list-unstyled">
                        <li><i className="bi bi-check-circle-fill text-success me-2"></i>React</li>
                        <li><i className="bi bi-check-circle-fill text-success me-2"></i>Bootstrap 5</li>
                        <li><i className="bi bi-check-circle-fill text-success me-2"></i>Plotly.js</li>
                        <li><i className="bi bi-check-circle-fill text-success me-2"></i>Chart.js</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="mb-4">
              <h5>Μελλοντικές Επεκτάσεις</h5>
              <div className="card">
                <div className="card-body">
                  <p>Σχεδιάζονται οι ακόλουθες επεκτάσεις στην εφαρμογή:</p>
                  <ul>
                    <li>Προσθήκη περισσότερων συναισθηματικών κατηγοριών (χαρά, εμπιστοσύνη, άγχος, κλπ.)</li>
                    <li>Ενσωμάτωση λεξικών συναισθημάτων για πιο λεπτομερή ανάλυση</li>
                    <li>Χρήση λανθάνουσας σημασιολογικής ανάλυσης (LSA)</li>
                    <li>Συνδυασμός του BERT με το VADER για βελτιωμένη ακρίβεια</li>
                    <li>Λεπτομερέστερες αναφορές και εξαγωγή δεδομένων</li>
                  </ul>
                </div>
              </div>
            </div>
            
            <div>
              <h5>Στόχος της Εφαρμογής</h5>
              <p>
                Στόχος της εφαρμογής είναι να κατηγοριοποιήσει και να ποσοτικοποιήσει συναισθήματα 
                χρησιμοποιώντας τεχνικές που ανήκουν στον τομέα της επεξεργασίας φυσικής γλώσσας (NLP). 
                Τα αποτελέσματα μπορούν να χρησιμοποιηθούν από επιχειρήσεις για την βελτίωση των πελατειακών σχέσεων 
                και την αποτελεσματικότερη λήψη αποφάσεων.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About; 