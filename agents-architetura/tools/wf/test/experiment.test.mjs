import { test } from 'node:test'
import assert from 'node:assert/strict'
import { loadLib } from '../sim.mjs'

const { classifyExperiment, promotionChecklist, PROMOTION_ITEMS } = loadLib(['experiment'])
const m = (expected, observed, higherIsBetter = true) => ({ expected, observed, higherIsBetter })

test('classification is arithmetic against the prediction, with tolerance and direction', () => {
  assert.equal(classifyExperiment({ primary: 'speedup', metrics: { speedup: m(4, 3.8), stepTimeSec: m(2.5, 2.6, false) } }).classification, 'VALIDATED')
  const partial = classifyExperiment({ primary: 'speedup', metrics: { speedup: m(4, 4.2), costUsd: m(10, 14, false) } })
  assert.deepEqual([partial.classification, partial.why], ['PARTIALLY VALIDATED', 'speedup met; missed: costUsd'])
  assert.equal(classifyExperiment({ primary: 'speedup', metrics: { speedup: m(4, 2.1) } }).classification, 'REFUTED')
  assert.equal(classifyExperiment({ primary: 'speedup', metrics: { speedup: m(4, 4) }, falsified: true }).classification, 'REFUTED')
  assert.equal(classifyExperiment({ primary: 'speedup', metrics: { speedup: m(4, null) } }).classification, 'INCONCLUSIVE')
  assert.equal(classifyExperiment({ primary: 'speedup', metrics: { other: m(1, 1) } }).classification, 'INCONCLUSIVE')
})

test('promotion checklist derives every item; nothing is assumed', () => {
  const all = promotionChecklist({ hypothesis: 'outer-step sync cuts step time 4x over 80 ms RTT', adrApproved: true, prototypeOk: true, correctnessOk: true, ladderGreen: true, failureOk: true, classification: 'VALIDATED' })
  assert.deepEqual(Object.keys(all), PROMOTION_ITEMS)
  assert.ok(Object.values(all).every(Boolean))
  const some = promotionChecklist({ hypothesis: 'x', adrApproved: true, prototypeOk: true, classification: 'REFUTED' })
  assert.deepEqual(Object.entries(some).filter(([, v]) => !v).map(([k]) => k), ['hypothesisDefined', 'localCorrectness', 'testsPass', 'failureCasesTested', 'improvementDemonstrated'])
})
