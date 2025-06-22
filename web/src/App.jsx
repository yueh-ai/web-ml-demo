import React from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { AppBar, Toolbar, Typography, Container, Paper } from '@mui/material';
import DemoPredictor from './components/DemoPredictor';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
});

function App() {
  console.log('App component rendering');
  
  // Add error boundary
  const [hasError, setHasError] = React.useState(false);
  
  React.useEffect(() => {
    window.addEventListener('error', (e) => {
      console.error('Global error:', e);
      setHasError(true);
    });
  }, []);
  
  if (hasError) {
    return (
      <Container>
        <Paper style={{ padding: 20, marginTop: 20 }}>
          <Typography color="error">
            An error occurred. Check the console for details.
          </Typography>
        </Paper>
      </Container>
    );
  }
  
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">
            ML Browser Demo - Real-time Inference with ONNX Runtime
          </Typography>
        </Toolbar>
      </AppBar>
      <DemoPredictor />
    </ThemeProvider>
  );
}

export default App;