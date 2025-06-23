import { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Slider,
  Grid,
  Box,
  CircularProgress,
  Alert
} from '@mui/material';
import { useModelLoader } from '../hooks/useModelLoader';
import { usePredict } from '../hooks/usePredict';
import type { FeatureData } from '../types';

// Styles now defined as objects for sx prop
const styles = {
  root: {
    padding: 4,
    marginTop: 4,
  },
  paper: {
    padding: 4,
  },
  sliderContainer: {
    marginBottom: 3,
  },
  predictionBox: {
    marginTop: 4,
    padding: 3,
    backgroundColor: 'primary.main',
    color: 'primary.contrastText',
    borderRadius: 1,
    textAlign: 'center',
  },
  loadingContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: 300,
  },
};

export default function DemoPredictor() {
  const { meta, ready, loading, error: loadError, worker } = useModelLoader();
  
  // Initialize features with default values
  const [features, setFeatures] = useState<FeatureData | null>(null);

  // Initialize features when metadata is loaded
  useEffect(() => {
    if (meta?.feature_ranges) {
      const initialFeatures: FeatureData = {};
      Object.entries(meta.feature_ranges).forEach(([name, range]) => {
        initialFeatures[name] = range.default;
      });
      setFeatures(initialFeatures);
    }
  }, [meta]);

  const { prediction, predicting, error: predictError } = usePredict({
    worker,
    meta,
    features
  });

  const handleSliderChange = (featureName: string) => (_: Event, value: number | number[]) => {
    setFeatures(prev => ({
      ...prev!,
      [featureName]: value as number
    }));
  };

  if (loading) {
    return (
      <Container maxWidth="md" sx={styles.root}>
        <Paper sx={styles.paper}>
          <Box sx={styles.loadingContainer}>
            <CircularProgress size={60} />
          </Box>
        </Paper>
      </Container>
    );
  }

  if (loadError) {
    return (
      <Container maxWidth="md" sx={styles.root}>
        <Paper sx={styles.paper}>
          <Alert severity="error">
            Failed to load model: {loadError}
          </Alert>
        </Paper>
      </Container>
    );
  }

  if (!ready || !features || !meta) {
    return null;
  }

  return (
    <Container maxWidth="md" sx={styles.root}>
      <Paper sx={styles.paper}>
        <Typography variant="h4" gutterBottom>
          California House Price Prediction
        </Typography>
        
        <Typography variant="body1" color="textSecondary" paragraph>
          Adjust the sliders below to see real-time predictions using an XGBoost model
          running entirely in your browser.
        </Typography>

        <Grid container spacing={3}>
          {Object.entries(meta.feature_ranges).map(([name, range]) => (
            <Grid item xs={12} sm={6} key={name}>
              <Box sx={styles.sliderContainer}>
                <Typography gutterBottom>
                  {name}: <strong>{features[name].toFixed(range.step < 1 ? 1 : 0)}</strong>
                </Typography>
                <Slider
                  value={features[name]}
                  min={range.min}
                  max={range.max}
                  step={range.step}
                  onChange={handleSliderChange(name)}
                  valueLabelDisplay="auto"
                  marks={[
                    { value: range.min, label: String(range.min) },
                    { value: range.max, label: String(range.max) }
                  ]}
                />
              </Box>
            </Grid>
          ))}
        </Grid>

        {predictError && (
          <Alert severity="error" style={{ marginTop: 16 }}>
            Prediction error: {predictError}
          </Alert>
        )}

        <Box sx={styles.predictionBox}>
          <Typography variant="h6">
            Predicted House Price
          </Typography>
          <Typography variant="h3">
            {predicting ? (
              <CircularProgress size={40} color="inherit" />
            ) : prediction !== null ? (
              `$${(prediction * 100000).toLocaleString('en-US', { 
                minimumFractionDigits: 0,
                maximumFractionDigits: 0 
              })}`
            ) : (
              '—'
            )}
          </Typography>
          <Typography variant="caption">
            (Price in hundreds of thousands of dollars)
          </Typography>
        </Box>

        <Box mt={3}>
          <Typography variant="caption" color="textSecondary">
            Model: {meta.model_info.type} | 
            R² Score: {meta.model_info.r2_score} | 
            Training Samples: {meta.model_info.training_samples.toLocaleString()}
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
}