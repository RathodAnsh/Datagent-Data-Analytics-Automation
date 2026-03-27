import { useState, useEffect } from 'react';
import UploadPage from './pages/UploadPage';
import CleaningPage from './pages/CleaningPage';
import FeaturesPage from './pages/FeaturesPage';
import PlatformPage from './pages/PlatformPage';
import { Page, DatasetInfo, CleaningSummary } from './types';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('upload');
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [cleanedData, setCleanedData] = useState<DatasetInfo | null>(null);
  const [cleaningSummary, setCleaningSummary] = useState<CleaningSummary | null>(null);
  const [isAuthed, setIsAuthed] = useState<boolean | null>(null);

  useEffect(() => {
    // Read user_email from URL params first (passed from Next.js dashboard)
    const params = new URLSearchParams(window.location.search);
    const emailFromUrl = params.get('user_email');
    const idFromUrl    = params.get('user_id');

    if (emailFromUrl) {
      localStorage.setItem('user_email', emailFromUrl);
      if (idFromUrl) localStorage.setItem('user_id', idFromUrl);
      window.history.replaceState({}, '', window.location.pathname);
      setIsAuthed(true);
      return;
    }

    const savedEmail = localStorage.getItem('user_email');
    if (savedEmail) {
      setIsAuthed(true);
    } else {
      window.location.href = 'http://localhost:3000/login';
    }
  }, []);

  if (!isAuthed) return null;

  const handleUploadComplete = (data: DatasetInfo) => {
    setDatasetInfo(data);
    setCurrentPage('cleaning');
  };

  const handleSkipToFeatures = (data: DatasetInfo) => {
    setCleanedData(data);
    setDatasetInfo(data);
    setCleaningSummary(null);
    setCurrentPage('features');
  };

  const handleCleaningComplete = (cleaned: DatasetInfo, summary: CleaningSummary) => {
    setCleanedData(cleaned);
    setCleaningSummary(summary);
    setCurrentPage('features');
  };

  const handleProceedToPlatform = () => {
    setCurrentPage('platform');
  };

  const handleBackToFeatures = () => {
    setCurrentPage('features');
  };

  const handleRestart = () => {
    setDatasetInfo(null);
    setCleanedData(null);
    setCleaningSummary(null);
    setCurrentPage('upload');
  };

  return (
    <>
      {currentPage === 'upload' && (
        <UploadPage
          onUploadComplete={handleUploadComplete}
          onSkipToFeatures={handleSkipToFeatures}
        />
      )}
      {currentPage === 'cleaning' && datasetInfo && (
        <CleaningPage
          datasetInfo={datasetInfo}
          onProceedToFeatures={handleCleaningComplete}
        />
      )}
      {currentPage === 'features' && cleanedData && (
        <FeaturesPage
          cleanedData={cleanedData}
          onProceedToPlatform={handleProceedToPlatform}
        />
      )}
      {currentPage === 'platform' && (
        <PlatformPage onBack={handleBackToFeatures} />
      )}
    </>
  );
}

export default App;