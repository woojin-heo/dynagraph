import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

const AUTH_STORAGE_KEY = 'dynagraph_auth_token'

const AuthContext = createContext({
  token: null,
  user: null,
  isAuthenticated: false,
  loading: true,
  login: () => {},
  logout: () => {},
})

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem(AUTH_STORAGE_KEY) || null)
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  const logout = useCallback(() => {
    setToken(null)
    setUser(null)
    localStorage.removeItem(AUTH_STORAGE_KEY)
    navigate('/login', { replace: true })
  }, [navigate])

  const login = useCallback((newToken, newUser) => {
    setToken(newToken)
    setUser(newUser)
    localStorage.setItem(AUTH_STORAGE_KEY, newToken)
  }, [])

  // On mount, verify stored token via /api/auth/me
  useEffect(() => {
    if (!token) {
      setLoading(false)
      return
    }
    fetch('/api/auth/me', {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (res.ok) return res.json()
        throw new Error('invalid')
      })
      .then((data) => {
        setUser({ tenant_id: data.tenant_id, display_name: data.display_name })
      })
      .catch(() => {
        setToken(null)
        localStorage.removeItem(AUTH_STORAGE_KEY)
      })
      .finally(() => setLoading(false))
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <AuthContext.Provider value={{
      token,
      user,
      isAuthenticated: !!user,
      loading,
      login,
      logout,
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}

export default AuthContext
