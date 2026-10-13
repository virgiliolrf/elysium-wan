// Experiment judgment: expected vs observed, decided by arithmetic the human can audit, not by an agent's mood.
// metrics: {name: {expected, observed, higherIsBetter}}; tolerancePct: how far observed may fall short of expected and
// still count; primary: the metric the falsification criterion names.
//   VALIDATED            every metric meets its prediction within tolerance
//   PARTIALLY VALIDATED  the primary metric meets it, some other metric does not
//   REFUTED              the primary metric misses its prediction (or the falsification criterion fired)
//   INCONCLUSIVE         a metric is missing or not a number — the run did not measure what it promised
const classifyExperiment = ({ metrics, tolerancePct = 10, primary, falsified = false }) => {
  const rows = Object.entries(metrics ?? {}).map(([name, m]) => {
    const e = Number(m?.expected)
    const o = Number(m?.observed)
    if (!Number.isFinite(e) || !Number.isFinite(o) || m?.observed === null || m?.observed === undefined) return { name, met: null }
    const slack = Math.abs(e) * (tolerancePct / 100)
    const met = m.higherIsBetter === false ? o <= e + slack : o >= e - slack
    return { name, expected: e, observed: o, met, deltaPct: e ? Math.round(((o - e) / Math.abs(e)) * 1000) / 10 : null }
  })
  if (!rows.length || !rows.some(r => r.name === primary)) return { classification: 'INCONCLUSIVE', rows, why: `the primary metric ${primary} was not measured` }
  if (rows.some(r => r.met === null)) return { classification: 'INCONCLUSIVE', rows, why: `not measured: ${rows.filter(r => r.met === null).map(r => r.name).join(', ')}` }
  const p = rows.find(r => r.name === primary)
  if (falsified || !p.met) return { classification: 'REFUTED', rows, why: falsified ? 'the falsification criterion fired' : `${primary} missed its prediction (${p.observed} vs ${p.expected})` }
  if (rows.every(r => r.met)) return { classification: 'VALIDATED', rows, why: 'every metric met its prediction' }
  return { classification: 'PARTIALLY VALIDATED', rows, why: `${primary} met; missed: ${rows.filter(r => !r.met).map(r => r.name).join(', ')}` }
}

// Local -> paid promotion checklist; every item must carry evidence (a boolean the script derived, not a claim).
const PROMOTION_ITEMS = ['hypothesisDefined', 'architectureApproved', 'localPrototypeWorks', 'localCorrectness', 'testsPass', 'failureCasesTested', 'improvementDemonstrated']
const promotionChecklist = ({ hypothesis, adrApproved, prototypeOk, correctnessOk, ladderGreen, failureOk, classification }) => ({
  hypothesisDefined: typeof hypothesis === 'string' && hypothesis.trim().length >= 20,
  architectureApproved: adrApproved === true,
  localPrototypeWorks: prototypeOk === true,
  localCorrectness: correctnessOk === true,
  testsPass: ladderGreen === true,
  failureCasesTested: failureOk === true,
  improvementDemonstrated: ['VALIDATED', 'PARTIALLY VALIDATED'].includes(classification),
})
