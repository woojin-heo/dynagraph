import { createContext, useContext, useState, useEffect, useCallback } from 'react'

const STORAGE_KEY = 'dynagraph_tenant_id'

const TenantContext = createContext({
  tenantId: null,
  tenantName: null,
  tenants: [],
  setTenant: () => {},
  refreshTenants: () => {},
  createTenant: () => {},
})

export function TenantProvider({ children }) {
  const [tenantId, setTenantId] = useState(() => localStorage.getItem(STORAGE_KEY) || null)
  const [tenantName, setTenantName] = useState(null)
  const [tenants, setTenants] = useState([])

  const refreshTenants = useCallback(async () => {
    try {
      const res = await fetch('/api/tenants')
      if (res.ok) {
        const data = await res.json()
        setTenants(data)
        if (tenantId) {
          const current = data.find((t) => t.id === tenantId)
          if (current) setTenantName(current.name)
          else {
            setTenantId(null)
            setTenantName(null)
            localStorage.removeItem(STORAGE_KEY)
          }
        }
      }
    } catch (_) {}
  }, [tenantId])

  useEffect(() => {
    refreshTenants()
  }, [refreshTenants])

  const setTenant = useCallback((id, name) => {
    setTenantId(id)
    setTenantName(name)
    if (id) localStorage.setItem(STORAGE_KEY, id)
    else localStorage.removeItem(STORAGE_KEY)
  }, [])

  const createTenantFn = useCallback(async (name) => {
    const res = await fetch('/api/tenants', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.error || 'Failed to create tenant')
    }
    const tenant = await res.json()
    await refreshTenants()
    setTenant(tenant.id, tenant.name)
    return tenant
  }, [refreshTenants, setTenant])

  return (
    <TenantContext.Provider value={{
      tenantId,
      tenantName,
      tenants,
      setTenant,
      refreshTenants,
      createTenant: createTenantFn,
    }}>
      {children}
    </TenantContext.Provider>
  )
}

export function useTenant() {
  return useContext(TenantContext)
}

export default TenantContext
