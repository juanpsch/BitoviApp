import ReactMarkdown from 'react-markdown';
import { useState } from 'react'


function App() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [loading, setLoading] = useState(false)

  const askAi = async (e) => {
    e.preventDefault()
    if (!question) return
    
    setLoading(true)
    setAnswer('')
    setSources([])

    try {
      // Llamada a tu backend de Python
      const res = await fetch(`http://localhost:8000/ask?question=${encodeURIComponent(question)}`);
      const data = await res.json();      
      
      if (data.response) {
        setAnswer(data.response.trim());
        setSources(data.sources || []);
}     
    } catch (err) {
      alert("Error: ¿Está el backend (backend.py) corriendo en la terminal?");
    } finally {
      setLoading(false);
    }
  }

  return (
  <div className="min-h-screen bg-slate-50 flex items-center justify-center p-6">
    <div className="max-w-2xl w-full">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-extrabold text-blue-900 mb-2">Bitovi AI Expert</h1>
        <p className="text-slate-500 font-medium">Expert insights from Bitovi's Blog</p>
      </div>

      <div className="bg-white rounded-2xl shadow-xl p-8 border border-slate-100">
        <form onSubmit={askAi} className="space-y-4">
          <input 
            type="text" 
            className="w-full p-4 rounded-xl border-2 border-slate-100 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all text-lg"
            placeholder="Ej: What are the benefits of using Playwright?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button 
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-4 rounded-xl font-bold text-lg hover:bg-blue-700 active:transform active:scale-[0.98] transition-all disabled:bg-slate-300"
          >
            {loading ? 'Pensando...' : 'Consultar Agente'}
          </button>
        </form>

        {answer && (
          <div className="mt-8 animate-in fade-in duration-500">
            <div className="p-6 bg-blue-50 rounded-xl border border-blue-100 text-left">
              <h2 className="text-sm font-black text-blue-900 uppercase tracking-wider mb-3">Respuesta:</h2>
              <div className="prose prose-blue max-w-none text-slate-700">
                <ReactMarkdown
    components={{
      // Personalizamos cómo se ven los links (a) dentro del Markdown
      a: ({ node, ...props }) => (
        <a 
          {...props} 
          target="_blank" 
          rel="noopener noreferrer" 
          className="text-blue-600 font-bold hover:text-blue-800 underline decoration-blue-300 underline-offset-4"
        />
      ),
      // También podemos ponerle estilo a las listas para que no se vean pegadas
      li: ({ node, ...props }) => (
        <li {...props} className="mb-2 ml-4 list-disc" />
      ),
      // Y a los títulos
      h2: ({ node, ...props }) => (
        <h2 {...props} className="text-xl font-bold text-blue-900 mt-6 mb-4 border-b pb-2" />
      ),
    }}
  >
    {answer}
  </ReactMarkdown>
            </div>
            </div>
            
            {/* --- SECCIÓN DE FUENTES ACTUALIZADA --- */}
            {sources.length > 0 && (
              <div className="mt-6 text-left">
                <h3 className="text-xs font-bold text-slate-400 uppercase mb-3">Fuentes Citadas:</h3>
                <div className="flex flex-col gap-3">
                  {sources.map((s, i) => (
                    <a 
                      key={i} 
                      href={s.url} 
                      target="_blank" 
                      rel="noreferrer" 
                      className="group block bg-white p-4 border border-slate-200 rounded-xl shadow-sm hover:border-blue-400 hover:shadow-md transition-all"
                    >
                      <div className="flex flex-col">
                        <span className="text-blue-600 font-bold text-sm group-hover:underline">
                          {s.title || "Documento de Bitovi"}
                        </span>
                        <div className="flex justify-between items-center mt-2">
                          <span className="text-slate-500 text-[11px] font-medium">
                            👤 {s.author || 'Equipo Bitovi'}
                          </span>
                          <span className="text-slate-300 text-[10px] truncate max-w-[150px]">
                            {s.url}
                          </span>
                        </div>
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            )}
            {/* -------------------------------------- */}
          </div>
        )}
      </div>
      
      <footer className="mt-8 text-center text-slate-400 text-sm">
        Built with LangChain + Ollama + React
      </footer>
    </div>
  </div>
);
}

export default App