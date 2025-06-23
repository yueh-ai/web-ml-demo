import React from 'react'
import ReactDOM from 'react-dom'
import App from './App'
import './index.css'

console.log('Main.tsx loading');

const rootElement = document.getElementById('root');

if (rootElement) {
  ReactDOM.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
    rootElement
  );
  console.log('React app mounted');
} else {
  console.error('Root element not found');
}