import React, { useState, useEffect, useRef } from 'react';
import { Activity, Thermometer, Wind, Droplet, User, AlertTriangle, CheckCircle, Clock, ChevronRight, FileText, Share2, Search, Brain } from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';

export default function App() {
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [patientDetails, setPatientDetails] = useState(null);

  // Vitals form
  const [vitals, setVitals] = useState({
    hr: 75,
    rr: 16,
    temp: 37.0,
    lactate: 1.0
  });

  // Pipeline states
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    fetch('/api/patients')
      .then(res => res.json())
      .then(data => {
        setPatients(data);
        if (data.length > 0) {
          selectPatient(data[0].subject_id);
        }
      })
      .catch(err => console.error("Error fetching patients:", err));
  }, []);

  const extractLatestVitals = (patient) => {
    const fallbackVitals = {
      hr: 75,
      rr: 16,
      temp: 37.0,
      lactate: 1.0,
    };

    const visits = patient?.visits || [];
    if (visits.length === 0) {
      return fallbackVitals;
    }

    const latestVisit = visits[visits.length - 1];
    const events = latestVisit?.events || [];

    const findValue = (code, defaultValue) => {
      const matched = events.find((event) => event.itemid === code && event.valuenum !== null && event.valuenum !== undefined);
      return matched ? Number(matched.valuenum) : defaultValue;
    };

    return {
      hr: findValue('8867-4', fallbackVitals.hr),
      rr: findValue('9279-1', fallbackVitals.rr),
      temp: findValue('8310-5', fallbackVitals.temp),
      lactate: findValue('32693-4', fallbackVitals.lactate),
    };
  };

  const selectPatient = (id) => {
    setSelectedPatientId(id);
    setResults(null);
    setActiveStep(0);
    fetch(`/api/patients/${id}`)
      .then(res => res.json())
      .then(data => {
        setPatientDetails(data);
        setVitals(extractLatestVitals(data));
      })
      .catch(err => console.error("Error fetching patient details:", err));
  };

  const handleVitalChange = (e) => {
    setVitals({
      ...vitals,
      [e.target.name]: parseFloat(e.target.value)
    });
  };

  const runAgents = async () => {
    if (!patientDetails) return;

    setIsProcessing(true);
    setResults(null);
    setActiveStep(1); // Perceptor

    try {
      // Simulate pipeline steps for UI feedback
      await new Promise(r => setTimeout(r, 800));
      setActiveStep(2); // Planner
      await new Promise(r => setTimeout(r, 800));
      setActiveStep(3); // Executor
      await new Promise(r => setTimeout(r, 800));
      setActiveStep(4); // Verifier

      // Get latest visit id or mock one
      const visit_id = patientDetails.visits && patientDetails.visits.length > 0
        ? patientDetails.visits[patientDetails.visits.length - 1].hadm_id
        : `E${selectedPatientId}_001`;

      const response = await fetch('/api/monitor', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          subject_id: selectedPatientId,
          visit_id: visit_id,
          hr: vitals.hr,
          rr: vitals.rr,
          temp: vitals.temp,
          lactate: vitals.lactate
        })
      });

      const data = await response.json();
      setResults(data);
      setActiveStep(5); // Done
    } catch (err) {
      console.error("Error running agents:", err);
      alert("Failed to run agent pipeline");
    } finally {
      setIsProcessing(false);
    }
  };

  const getHeatColor = (heat) => {
    // Return colors from cool blue to hot red based on heat value (0 to 1)
    if (heat < 0.3) return '#3b82f6'; // blue-500
    if (heat < 0.6) return '#eab308'; // yellow-500
    if (heat < 0.8) return '#f97316'; // orange-500
    return '#ef4444'; // red-500
  };

  return (
    <div className="flex h-screen bg-slate-50 font-sans text-slate-800 w-full" style={{ textAlign: 'left', minWidth: '100%', margin: 0, padding: 0, maxWidth: 'none' }}>

      {/* Sidebar: Patient List */}
      <div className="w-80 bg-white border-r border-slate-200 flex flex-col h-full overflow-hidden shadow-sm z-10">
        <div className="p-4 border-b border-slate-200 bg-slate-50 flex items-center gap-2">
          <Activity className="text-blue-600" />
          <h1 className="text-xl font-bold text-slate-800 m-0">Sepsis Monitor</h1>
        </div>
        <div className="p-4 bg-slate-100 border-b border-slate-200 text-sm font-semibold text-slate-500 uppercase tracking-wider">
          ICU Patients
        </div>
        <div className="overflow-y-auto flex-1 p-2 space-y-1">
          {patients.map(p => (
            <div
              key={p.subject_id}
              onClick={() => selectPatient(p.subject_id)}
              className={`p-3 rounded-lg cursor-pointer transition-colors border ${selectedPatientId === p.subject_id ? 'bg-blue-50 border-blue-200' : 'bg-white border-transparent hover:bg-slate-50 hover:border-slate-200'}`}
            >
              <div className="flex justify-between items-center">
                <div className="font-semibold text-slate-700 flex items-center gap-2">
                  <User size={16} className={selectedPatientId === p.subject_id ? 'text-blue-500' : 'text-slate-400'} />
                  Patient {p.subject_id}
                </div>
              </div>
              <div className="text-xs text-slate-500 mt-1 pl-6">
                {p.demographics?.age} yo {p.demographics?.gender} • {p.demographics?.race}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full overflow-hidden bg-slate-50">

        {/* Header */}
        <div className="h-16 bg-white border-b border-slate-200 flex items-center px-6 shadow-sm justify-between shrink-0">
          <h2 className="text-lg font-semibold text-slate-700 m-0">
            {patientDetails ? `Patient Dashboard: ${patientDetails.subject_id}` : 'Select a patient'}
          </h2>
          {patientDetails && (
            <div className="flex gap-4 text-sm text-slate-600 bg-slate-100 px-4 py-1.5 rounded-full border border-slate-200">
              <span>Age: <strong className="text-slate-800">{patientDetails.demographics?.age}</strong></span>
              <span>Gender: <strong className="text-slate-800">{patientDetails.demographics?.gender}</strong></span>
              <span>Race: <strong className="text-slate-800 capitalize">{patientDetails.demographics?.race}</strong></span>
            </div>
          )}
        </div>

        {/* Dashboard Content */}
        <div className="flex-1 overflow-y-auto p-6 flex flex-col xl:flex-row gap-6">

          {/* Left Column: Input & Pipeline */}
          <div className="w-full xl:w-1/3 flex flex-col gap-6">

            {/* Vitals Input Form */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 text-left">
              <div className="flex items-center gap-2 mb-4 border-b border-slate-100 pb-3">
                <FileText className="text-blue-500" size={20} />
                <h3 className="font-semibold text-slate-800 m-0">Log Clinical Vitals</h3>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-5">
                <div>
                  <label className="text-xs font-semibold text-slate-500 mb-1 flex items-center gap-1">
                    <Activity size={12} /> Heart Rate (bpm)
                  </label>
                  <input type="number" name="hr" value={vitals.hr} onChange={handleVitalChange}
                    className="w-full border border-slate-300 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none" />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-500 mb-1 flex items-center gap-1">
                    <Wind size={12} /> Resp Rate (br/min)
                  </label>
                  <input type="number" name="rr" value={vitals.rr} onChange={handleVitalChange}
                    className="w-full border border-slate-300 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none" />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-500 mb-1 flex items-center gap-1">
                    <Thermometer size={12} /> Temp (°C)
                  </label>
                  <input type="number" name="temp" value={vitals.temp} step="0.1" onChange={handleVitalChange}
                    className="w-full border border-slate-300 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none" />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-500 mb-1 flex items-center gap-1">
                    <Droplet size={12} /> Lactate (mmol/L)
                  </label>
                  <input type="number" name="lactate" value={vitals.lactate} step="0.1" onChange={handleVitalChange}
                    className="w-full border border-slate-300 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none" />
                </div>
              </div>

              <button
                onClick={runAgents}
                disabled={isProcessing || !patientDetails}
                className={`w-full py-2.5 rounded-lg font-medium text-white flex justify-center items-center gap-2 transition-all ${isProcessing || !patientDetails ? 'bg-blue-300 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 shadow-sm hover:shadow'}`}
              >
                {isProcessing ? (
                  <><Clock size={18} className="animate-spin" /> Analyzing...</>
                ) : (
                  <><Activity size={18} /> Run Agent Framework</>
                )}
              </button>
            </div>

            {/* Agent Pipeline Visualizer */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 text-left">
               <h3 className="font-semibold text-slate-800 mb-4 border-b border-slate-100 pb-3 m-0">AI Agent Pipeline</h3>
               <div className="space-y-4 relative before:absolute before:inset-0 before:ml-[15px] before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-slate-300 before:to-transparent">

                  {/* Perceptor */}
                  <div className={`relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active`}>
                    <div className={`flex items-center justify-center w-8 h-8 rounded-full border-2 bg-white z-10 shrink-0 shadow-sm
                      ${activeStep >= 1 ? 'border-blue-500 text-blue-500' : 'border-slate-300 text-slate-300'}`}>
                      <Activity size={14} />
                    </div>
                    <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] ml-4 md:ml-0 md:mr-4 p-3 rounded-lg border border-slate-200 bg-slate-50">
                      <div className={`font-semibold text-sm ${activeStep >= 1 ? 'text-slate-800' : 'text-slate-400'}`}>1. Perceptor</div>
                      <div className="text-xs text-slate-500 mt-1">NLP & Sepsis-3 screening</div>
                    </div>
                  </div>

                  {/* Planner */}
                  <div className={`relative flex items-center justify-between md:justify-normal md:even:flex-row group`}>
                    <div className={`flex items-center justify-center w-8 h-8 rounded-full border-2 bg-white z-10 shrink-0 shadow-sm
                      ${activeStep >= 2 ? 'border-purple-500 text-purple-500' : 'border-slate-300 text-slate-300'}`}>
                      <FileText size={14} />
                    </div>
                    <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] ml-4 md:ml-4 p-3 rounded-lg border border-slate-200 bg-slate-50">
                      <div className={`font-semibold text-sm ${activeStep >= 2 ? 'text-slate-800' : 'text-slate-400'}`}>2. Planner</div>
                      <div className="text-xs text-slate-500 mt-1">RAG Treatment Plan</div>
                    </div>
                  </div>

                  {/* Executor */}
                  <div className={`relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group`}>
                    <div className={`flex items-center justify-center w-8 h-8 rounded-full border-2 bg-white z-10 shrink-0 shadow-sm
                      ${activeStep >= 3 ? 'border-amber-500 text-amber-500' : 'border-slate-300 text-slate-300'}`}>
                      <CheckCircle size={14} />
                    </div>
                    <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] ml-4 md:ml-0 md:mr-4 p-3 rounded-lg border border-slate-200 bg-slate-50">
                      <div className={`font-semibold text-sm ${activeStep >= 3 ? 'text-slate-800' : 'text-slate-400'}`}>3. Executor</div>
                      <div className="text-xs text-slate-500 mt-1">FHIR Orders Mock</div>
                    </div>
                  </div>

                  {/* Verifier */}
                  <div className={`relative flex items-center justify-between md:justify-normal md:even:flex-row group`}>
                    <div className={`flex items-center justify-center w-8 h-8 rounded-full border-2 bg-white z-10 shrink-0 shadow-sm
                      ${activeStep >= 4 ? 'border-emerald-500 text-emerald-500' : 'border-slate-300 text-slate-300'}`}>
                      <Activity size={14} />
                    </div>
                    <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] ml-4 md:ml-4 p-3 rounded-lg border border-slate-200 bg-slate-50">
                      <div className={`font-semibold text-sm ${activeStep >= 4 ? 'text-slate-800' : 'text-slate-400'}`}>4. Verifier</div>
                      <div className="text-xs text-slate-500 mt-1">SHAP-proxy explains</div>
                    </div>
                  </div>

                  {/* Evaluator */}
                  <div className={`relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group`}>
                    <div className={`flex items-center justify-center w-8 h-8 rounded-full border-2 bg-white z-10 shrink-0 shadow-sm
                      ${activeStep >= 5 ? 'border-indigo-500 text-indigo-500' : 'border-slate-300 text-slate-300'}`}>
                      <Brain size={14} />
                    </div>
                    <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] ml-4 md:ml-0 md:mr-4 p-3 rounded-lg border border-slate-200 bg-slate-50">
                      <div className={`font-semibold text-sm ${activeStep >= 5 ? 'text-slate-800' : 'text-slate-400'}`}>5. Evaluator</div>
                      <div className="text-xs text-slate-500 mt-1">LLM as Judge</div>
                    </div>
                  </div>

               </div>
            </div>
          </div>

          {/* Right Column: Results View */}
          <div className="w-full xl:w-2/3 flex flex-col gap-6 text-left">
            {results ? (
              <>
                {/* Alert Status */}
                <div className={`rounded-xl p-6 border shadow-sm flex items-start gap-4 ${results.alert_triggered ? 'bg-red-50 border-red-200' : 'bg-emerald-50 border-emerald-200'}`}>
                  {results.alert_triggered ? (
                    <AlertTriangle className="text-red-600 mt-1 shrink-0" size={32} />
                  ) : (
                    <CheckCircle className="text-emerald-600 mt-1 shrink-0" size={32} />
                  )}
                  <div>
                    <h2 className={`text-xl font-bold m-0 mb-1 ${results.alert_triggered ? 'text-red-800' : 'text-emerald-800'}`}>
                      {results.alert_triggered ? 'Sepsis Alert Triggered' : 'No Sepsis Alert'}
                    </h2>
                    {results.alert_triggered && results.alerts.length > 0 && (
                      <div className="mt-2 text-sm text-red-700">
                        <p className="font-semibold mb-1">Sepsis-3 Criteria Matched:</p>
                        <ul className="list-disc pl-5 space-y-1 m-0">
                          {results.alerts[0].reasons.map((reason, i) => (
                            <li key={i}>{reason}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>

                {results.alert_triggered && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Clinical Note & Extraction */}
                    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 md:col-span-2">
                      <h3 className="font-semibold text-slate-800 mb-3 flex items-center gap-2 pb-2 border-b border-slate-100 m-0">
                        <Search size={18} className="text-blue-500" /> Clinical Notes & Entity Extraction
                      </h3>
                      <div className="flex flex-col md:flex-row gap-6">
                        <div className="w-full md:w-2/3">
                          <p className="text-xs font-semibold text-slate-500 mb-2">Unstructured Note</p>
                          <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 text-slate-700 text-sm max-h-48 overflow-y-auto whitespace-pre-wrap">
                            {results.clinical_note || "No clinical note available for this encounter."}
                          </div>
                        </div>
                        <div className="w-full md:w-1/3">
                          <p className="text-xs font-semibold text-slate-500 mb-2">MedCAT Extracted Entities</p>
                          {results.extracted_entities && results.extracted_entities.length > 0 ? (
                            <div className="flex flex-wrap gap-2">
                              {results.extracted_entities.map((e, idx) => (
                                <span key={idx} className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-md border border-blue-200">
                                  {e.name}
                                </span>
                              ))}
                            </div>
                          ) : (
                            <div className="text-sm text-slate-500 italic bg-slate-50 p-3 rounded-lg border border-slate-200">No matching clinical entities found in note.</div>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Cellular Network Graph */}
                    {results.cellular_data && results.cellular_data.nodes && (
                      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 md:col-span-2">
                        <h3 className="font-semibold text-slate-800 mb-3 flex items-center gap-2 pb-2 border-b border-slate-100 m-0">
                          <Share2 size={18} className="text-indigo-500" /> Cellular Data Network (Genes, Proteins, Pathways)
                        </h3>
                        <div className="flex flex-col md:flex-row gap-6">
                          <div className="w-full md:w-2/3 h-64 bg-slate-900 rounded-lg overflow-hidden border border-slate-300">
                            <ForceGraph2D
                              graphData={results.cellular_data}
                              width={600}
                              height={256}
                              nodeLabel="id"
                              nodeColor={(node) => getHeatColor(node.heat || 0)}
                              nodeVal={(node) => (node.type === 'pathway' ? 8 : 4)}
                              linkColor={() => 'rgba(255,255,255,0.2)'}
                              backgroundColor="#0f172a"
                            />
                          </div>
                          <div className="w-full md:w-1/3">
                            <p className="text-xs font-semibold text-slate-500 mb-2">Heat Signatures Analysis</p>
                            <div className="space-y-3">
                              {results.cellular_data.nodes.sort((a,b)=>(b.heat||0)-(a.heat||0)).slice(0, 5).map((node, idx) => (
                                <div key={idx} className="flex items-center gap-3">
                                  <div className="w-16 text-right text-xs font-semibold text-slate-600 truncate" title={node.id}>{node.id}</div>
                                  <div className="flex-1 bg-slate-100 rounded-full h-2.5 overflow-hidden">
                                    <div
                                      className="h-full rounded-full"
                                      style={{ width: `${Math.min(100, (node.heat || 0) * 100)}%`, backgroundColor: getHeatColor(node.heat || 0) }}
                                    ></div>
                                  </div>
                                  <div className="w-12 text-xs text-slate-500">{((node.heat || 0)*100).toFixed(0)}%</div>
                                </div>
                              ))}
                            </div>
                            <div className="mt-4 text-xs text-slate-500">
                              <span className="inline-block w-3 h-3 rounded-full bg-blue-500 mr-1"></span> Normal
                              <span className="inline-block w-3 h-3 rounded-full bg-red-500 mr-1 ml-3"></span> Elevated Heat
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Treatment Plan */}
                    <div className="bg-white rounded-xl shadow-sm border border-purple-100 p-5">
                      <h3 className="font-semibold text-purple-800 mb-3 flex items-center gap-2 pb-2 border-b border-purple-50 m-0">
                        <FileText size={18} /> Generated Treatment Plan (w/ Therapeutics)
                      </h3>
                      <ul className="space-y-2 m-0 p-0 list-none">
                        {results.plan.map((item, i) => (
                          <li key={i} className="flex gap-2 text-sm text-slate-700 items-start">
                            <ChevronRight size={16} className="text-purple-500 shrink-0 mt-0.5" />
                            <span>{item}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Evaluator Assessment */}
                    <div className="bg-white rounded-xl shadow-sm border border-indigo-100 p-5">
                      <h3 className="font-semibold text-indigo-800 mb-3 flex items-center gap-2 pb-2 border-b border-indigo-50 m-0">
                        <Brain size={18} /> LLM as Judge: Plan Evaluation
                      </h3>
                      
                      {results.evaluation ? (
                        <div className="space-y-4">
                          <div className="flex items-center gap-4">
                            <div className="text-center">
                              <div className={`text-2xl font-bold ${results.evaluation.score >= 80 ? 'text-indigo-600' : 'text-amber-600'}`}>
                                {results.evaluation.score}/100
                              </div>
                              <div className="text-xs text-slate-500">Score</div>
                            </div>
                            <div className="flex-1">
                              <div className="text-sm font-semibold text-slate-700 mb-1">Alignment: {results.evaluation.alignment}</div>
                              <div className="w-full bg-slate-100 rounded-full h-2">
                                <div className={`h-2 rounded-full ${results.evaluation.score >= 80 ? 'bg-indigo-500' : 'bg-amber-500'}`} style={{ width: `${results.evaluation.score}%` }}></div>
                              </div>
                            </div>
                          </div>
                          <div className="text-sm text-slate-700 bg-indigo-50 p-3 rounded-lg border border-indigo-100">
                            <strong>Judge Feedback:</strong> {results.evaluation.feedback}
                          </div>
                        </div>
                      ) : (
                        <div className="text-sm text-slate-500">Evaluation pending...</div>
                      )}
                    </div>

                    {/* SHAP Explanation */}
                    <div className="bg-white rounded-xl shadow-sm border border-emerald-100 p-5 md:col-span-2">
                      <h3 className="font-semibold text-emerald-800 mb-3 flex items-center gap-2 pb-2 border-b border-emerald-50 m-0">
                        <Activity size={18} /> Verifier Explainability (SHAP-proxy)
                      </h3>
                      <div className="flex flex-col md:flex-row gap-6 items-center">
                        <div className="w-full md:w-1/2">
                          <pre className="text-xs bg-slate-50 p-4 rounded-lg border border-slate-200 overflow-x-auto text-slate-700 m-0">
                            {results.explanation}
                          </pre>
                        </div>
                        <div className="w-full md:w-1/2 flex flex-col gap-3">
                          {Object.entries(results.shap_importance).sort((a,b)=>b[1]-a[1]).map(([feature, importance]) => (
                            <div key={feature} className="flex items-center gap-3">
                              <div className="w-16 text-right text-xs font-semibold text-slate-600">{feature}</div>
                              <div className="flex-1 bg-slate-100 rounded-full h-3 overflow-hidden">
                                <div
                                  className="bg-emerald-500 h-full rounded-full"
                                  style={{ width: `${Math.min(100, importance * 100)}%` }}
                                ></div>
                              </div>
                              <div className="w-12 text-xs text-slate-500">{(importance*100).toFixed(1)}%</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-slate-400 p-10 bg-white rounded-xl border border-dashed border-slate-300">
                <Activity size={48} className="mb-4 text-slate-300 opacity-50" />
                <p className="text-lg font-medium text-slate-500 mb-2">Awaiting Clinical Evaluation</p>
                <p className="text-sm text-center max-w-md">
                  Select a patient from the list, update their vitals in the form, and run the agent framework to generate a personalized treatment plan and analysis.
                </p>
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}
