import React, { useState } from 'react';
import { ThemeProvider, CssBaseline, Container, Box } from '@mui/material';
import theme from './theme';
import Hero from './components/Hero';
import GenerationForm from './components/GenerationForm';
import ResultDashboard from './components/ResultDashboard';
import { generateMusic } from './api';

function App() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleGenerate = async (params) => {
    setIsGenerating(true);
    setResult(null);
    setError(null);

    try {
      const gResult = await generateMusic(params);
      setResult(gResult);
    } catch (err) {
      setError(err);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', pb: 10 }}>
        <Hero />

        <Container maxWidth="md">
          <GenerationForm
            onGenerate={handleGenerate}
            isGenerating={isGenerating}
          />

          <ResultDashboard
            result={result}
            error={error}
          />
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
