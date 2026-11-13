import React, { createContext, useContext, useState } from 'react';

const ServerContext = createContext();

const STORAGE_KEY = 'edgefl.nodeUrl';
const DEFAULT_NODE_URL = 'localhost:8080';

export const useServer = () => {
  const context = useContext(ServerContext);
  if (!context) {
    throw new Error('useServer must be used within a ServerProvider');
  }
  return context;
};

export const ServerProvider = ({ children }) => {
  const [nodeUrl, setNodeUrlState] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) || DEFAULT_NODE_URL;
    } catch (e) {
      return DEFAULT_NODE_URL;
    }
  });
  const [indexValue, setIndexValue] = useState('test-index');

  const setNodeUrl = (value) => {
    setNodeUrlState(value);
    try {
      localStorage.setItem(STORAGE_KEY, value);
    } catch (e) {
      // Storage unavailable: the value still works for this session.
    }
  };

  // Accepts "host:port" or a full URL; adds http:// if missing, strips trailing slashes.
  const getServerUrl = () => {
    const trimmed = (nodeUrl.trim() || DEFAULT_NODE_URL).replace(/\/+$/, '');
    return /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
  };

  const value = {
    nodeUrl,
    setNodeUrl,
    serverUrl: getServerUrl(),
    indexValue,
    setIndexValue,
  };

  return (
    <ServerContext.Provider value={value}>
      {children}
    </ServerContext.Provider>
  );
};
