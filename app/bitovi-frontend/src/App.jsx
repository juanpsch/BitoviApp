import ReactMarkdown from 'react-markdown'
import { useState } from 'react'

const markdownComponents = {
  a: ({ node, ...props }) => (
    <a {...props} target="_blank" rel="noopener noreferrer"
      className="text-blue-600 font-bold hover:text-blue-800 underline decoration-blue-300 underline-offset-4" />
  ),
  li: ({ node, ...props }) => <li {...props} className="mb-2 ml-4 list-disc" />,
  h2: ({ node, ...props }) => (
    <h2 {...props} className="text-xl font-bold text-blue-900 mt-6 mb-4 border-b pb-2" />
  ),
}

function StepDetail({ detail }) {
  const [open, setOpen] = useState(false)
  if (!detail) return null
  const entries = Object.entries(detail).filter(([, v]) => v !== null && v !== undefined && v !== '')
  if (entries.length === 0) return null
  return (
    <div className="mt-1">
      <button onClick={() => setOpen((o) => !o)}
        className="text-[11px] text-slate-400 hover:text-slate-600">
        {open ? '▾ ocultar detalle' : '▸ ver detalle'}
      </button>
      {open && (
        <dl className="mt-1 text-[11px] text-slate-500 bg-slate-50 rounded-lg p-2 space-y-0.5">
          {entries.map(([k, v]) => (
            <div key={k} className="flex gap-2">
              <dt className="font-semibold">{k}:</dt>
              <dd className="break-all">{String(v)}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  )
}

function App() {
  const [question, setQuestion] = useState('')
  const [steps, setSteps] = useState([])
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [loading, setLoading] = useState(false)

  const handleEvent = (evt) => {
    switch (evt.type) {
      case 'step':
        setSteps((prev) => [...prev, evt])
        break
      case 'token':
        setAnswer((prev) => prev + evt.text)
        break
      case 'sources':
        setSources(evt.sources || [])
        break
      case 'error':
        alert('Error del agente: ' + evt.message)
        break
      default:
        break
    }
  }

  const askAi = async (e) => {
    e.preventDefault()
    if (!question) return
    setLoading(true)
    setSteps([])
    setAnswer('')
    setSources([])

    try {
      const res = await fetch('http://localhost:8000/ask/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()
        for (const part of parts) {
          const line = part.trim()
          if (!line.startsWith('data:')) continue
          handleEvent(JSON.parse(line.slice(5).trim()))
        }
      }
    } catch (err) {
      alert('Error: ¿está el backend corriendo en http://localhost:8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-6">
          <h1 className="text-4xl font-extrabold text-blue-900 mb-2">Bitovi AI Expert</h1>
          <p className="text-slate-500 font-medium">Expert insights from Bitovi's Blog</p>
        </div>

        <form onSubmit={askAi} className="flex gap-3 mb-6">
          <input
            type="text"
            className="flex-1 p-4 rounded-xl border-2 border-slate-100 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all text-lg"
            placeholder="Ej: What are the benefits of using Playwright?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-600 text-white px-8 rounded-xl font-bold text-lg hover:bg-blue-700 active:scale-[0.98] transition-all disabled:bg-slate-300"
          >
            {loading ? 'Pensando...' : 'Consultar'}
          </button>
        </form>

        <div className="grid grid-cols-1 md:grid-cols-[320px_1fr] gap-6">
          {/* SIDE PANEL: agent flow */}
          <aside className="bg-white rounded-2xl shadow-lg p-5 border border-slate-100 h-fit">
            <h2 className="text-xs font-black text-slate-400 uppercase tracking-wider mb-4">
              Flujo del agente
            </h2>
            {steps.length === 0 && !loading && (
              <p className="text-sm text-slate-400">Los pasos aparecerán acá.</p>
            )}
            <ol className="space-y-3">
              {steps.map((s, i) => (
                <li key={i} className="text-sm">
                  <div className="flex items-start gap-2">
                    <span className="text-green-500 mt-0.5">✓</span>
                    <div className="flex-1">
                      <span className="text-slate-700 font-medium">{s.label}</span>
                      <StepDetail detail={s.detail} />
                    </div>
                  </div>
                </li>
              ))}
              {loading && (
                <li className="text-sm flex items-center gap-2 text-blue-500 animate-pulse">
                  <span>⟳</span> <span>en curso…</span>
                </li>
              )}
            </ol>
          </aside>

          {/* MAIN: streaming answer */}
          <main className="bg-white rounded-2xl shadow-xl p-8 border border-slate-100 min-h-[200px]">
            {!answer && !loading && (
              <p className="text-slate-400">La respuesta aparecerá acá, escribiéndose en vivo.</p>
            )}
            {answer && (
              <div className="prose prose-blue max-w-none text-slate-700">
                <ReactMarkdown components={markdownComponents}>{answer}</ReactMarkdown>
              </div>
            )}

            {sources.length > 0 && (
              <div className="mt-8">
                <h3 className="text-xs font-bold text-slate-400 uppercase mb-3">Fuentes citadas:</h3>
                <div className="flex flex-col gap-3">
                  {sources.map((s, i) => (
                    <a key={i} href={s.url} target="_blank" rel="noreferrer"
                      className="group block bg-white p-4 border border-slate-200 rounded-xl shadow-sm hover:border-blue-400 hover:shadow-md transition-all">
                      <span className="text-blue-600 font-bold text-sm group-hover:underline">
                        {s.title || 'Documento de Bitovi'}
                      </span>
                      <div className="flex justify-between items-center mt-2">
                        <span className="text-slate-500 text-[11px] font-medium">👤 {s.author || 'Equipo Bitovi'}</span>
                        <span className="text-slate-300 text-[10px] truncate max-w-[150px]">{s.url}</span>
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            )}
          </main>
        </div>

        <footer className="mt-8 text-center text-slate-400 text-sm">
          Built with LangGraph + Ollama + React
        </footer>
      </div>
    </div>
  )
}

export default App
