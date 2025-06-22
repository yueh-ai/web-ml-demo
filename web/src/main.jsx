import React from 'react'
import ReactDOM from 'react-dom'
import App from './App'
import './index.css'

console.log('Main.jsx loading');

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
)

console.log('React app mounted');