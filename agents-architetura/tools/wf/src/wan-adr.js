export const meta = {
  name: 'wan-adr',
  description: 'A major architecture change, decided the way an architecture team decides, thesis-style: several researchers in parallel, one per angle, with web access (papers, systems, implementations — each traced idea -> mechanism -> our bottleneck -> expected effect -> trade-offs -> experiment) -> the first-principles architect (Opus) writes one ADR from all of it, from the template with a real cost model, alternatives including doing nothing, mandatory falsification criteria and a local experiment before any paid cloud run, with the design/spec amendments written out -> 3 adversarial lenses looped -> the ARCHITECTURE DECISION REQUIRED presentation -> STOP. Nothing proceeds until a human grants the ADR and the amended design (silence is not approval).',
  whenToUse: 'When a needed change is MAJOR (training architecture, scheduler, synchronization, process responsibilities, protocols, training algorithm, checkpoint/recovery, topology, a core technology, major infrastructure, consistency or fault-tolerance semantics). wan-implement calls it when a spec conflict needs one. args: {"problem": "...", "id": "short-slug", "evidence": "...", "angles": ["<research angle>", ...] (optional, 2..8; default: five angles)}; or {"revise": "docs/design/adr/ADR-NNNN-<slug>.md", "references": ["<paper or fact the human brings>", ...]} to review an existing ADR against new references (no research, no draft); or {"split": "docs/design/adr/ADR-NNNN-<slug>.md", "references": [...]} to split an existing ADR into one-decision ADRs and review each. An ADR over its size budget, or whose review stalls, is split automatically.',
  phases: [
    { title: 'Gate', detail: 'Design approved; next ADR number' },
    { title: 'Research', detail: 'Researchers in parallel, one per angle (web): candidate mechanisms traced to our bottleneck' },
    { title: 'Draft', detail: 'Architect (Opus, first principles): ADR from all the research + the design/spec amendments (specs amended on the record)' },
    { title: 'Split', detail: 'An ADR over 400 lines, or whose review stalls, becomes 2-4 one-decision ADRs plus research notes; each is reviewed on its own' },
    { title: 'Review', detail: 'first-principles, alternatives-and-risk, consistency, frontier: reviewers with web access file findings AND new ideas; the architect fixes and adopts or rejects every idea with its reason; loop <= 5' },
    { title: 'Present', detail: 'Log the decision block; STOP for the human grant' },
  ],
}

// @include hygiene
// @include schemas
// @include dag
// @include loop
// @include project

// Revise an existing ADR with what the human brings (papers, systems, facts): Research and Draft are skipped; the reviewers,
// who have the web, verify each reference and the architect cites, compares or rejects it. Split starts from an existing ADR
// too, and cuts it into one-decision ADRs before the review.
const REVISE = typeof args?.revise === 'string' ? args.revise : null
const SPLIT = typeof args?.split === 'string' ? args.split : null
if (REVISE !== null && SPLIT !== null) throw new Error('args.revise and args.split are exclusive')
const ADR_PATH_RE = /^docs\/design\/adr\/(ADR-\d{4})-([a-z0-9-]{1,40})\.md$/
const revised = (REVISE ?? SPLIT) === null ? null : (REVISE ?? SPLIT).match(ADR_PATH_RE)
if ((REVISE ?? SPLIT) !== null && !revised) throw new Error(`args.${REVISE !== null ? 'revise' : 'split'} must be an existing ADR path like docs/design/adr/ADR-0001-slug.md (got ${JSON.stringify(clean(REVISE ?? SPLIT, 120))})`)
// "Divide always" (the human's rule, 2026-10-08): an ADR is one decision a human can grant or refuse, within a size budget.
// The hetero ADR reached 2006 lines; its review went 17 -> 17 -> 19 majors while each fix round got slower.
const MAX_ADR_LINES = 400
const MAX_SPLIT_DEPTH = 2
// From this round on, a fix corrects and never redesigns: new ideas are recorded as follow-ups and the decision and its
// mechanism stay. Rounds 1-2 may adopt ideas. The hetero ADR-0002 adopted a reviewer's idea in round 2 (a residue radix tree
// replaced the group-order fold), and round 3 filed the text the redesign left behind in four documents: 7 -> 12 majors.
const FREEZE_ROUND = 3
const references = Array.isArray(args?.references) ? args.references.slice(0, 20).map(r => clean(String(r), 600)) : []
if (!revised && (typeof args?.problem !== 'string' || args.problem.trim().length < 20)) throw new Error('args.problem (the problem that needs an architecture decision) is required')
const slug = revised ? revised[2] : token(args.id ?? args.problem, 40).toLowerCase()
const NEXT = { type: 'object', required: ['next'], properties: { next: { type: 'string', description: 'ADR-NNNN: one more than the highest existing number (ADR-0001 if none)' } } }
const RESEARCH = {
  type: 'object',
  required: ['bottleneck', 'options'],
  properties: {
    bottleneck: { type: 'string', description: 'the dominant term of the cost model and the evidence for it' },
    options: {
      type: 'array',
      items: {
        type: 'object',
        required: ['idea', 'source', 'mechanism', 'effectOnBottleneck', 'tradeoffs', 'experiment'],
        properties: { idea: { type: 'string' }, source: { type: 'string', description: 'paper / system / implementation, cited' }, mechanism: { type: 'string' }, effectOnBottleneck: { type: 'string' }, tradeoffs: { type: 'string' }, experiment: { type: 'string', description: 'the smallest local experiment that would refute it' } },
      },
    },
  },
}
// The reference PR's habit applied to research: several angles in parallel instead of one researcher's view.
const DEFAULT_ANGLES = [
  ['numerics', 'floating-point arithmetic and reproducibility: what each operator computes on each kind of hardware, exact and correctly rounded arithmetic, deterministic reductions, what makes two runs bit-identical'],
  ['systems', 'existing systems and papers for this problem: distributed and heterogeneous training systems, how they partition, schedule and synchronize, what they measured, what failed'],
  ['hardware', 'the hardware and its software stacks as they really behave (vendor documentation, compiler and runtime behavior, published measurements): what can and cannot be controlled'],
  ['performance', 'the cost model: where step time goes in this system (compute, communication, pipeline bubbles, synchronization), and what each candidate mechanism does to every term, with numbers'],
  ['contrarian', 'the strongest case against the obvious answer: different framings of the requirement, mechanisms from other fields, and proofs that a requirement is impossible as stated together with the strongest achievable alternative'],
]
const angles = Array.isArray(args?.angles) && args.angles.length >= 2 && args.angles.length <= 8
  ? args.angles.map((a, i) => [`angle-${i + 1}`, clean(String(a), 400)])
  : DEFAULT_ANGLES
// Reviewers return findings AND ideas: the frontier lens and every other lens may bring a mechanism the draft missed.
const REVIEW_WITH_IDEAS = {
  ...FINDINGS,
  properties: {
    ...FINDINGS.properties,
    ideas: {
      type: 'array',
      description: 'new mechanisms, papers or framings the draft did not consider; an idea that would materially change the decision is ALSO filed as a major finding',
      items: {
        type: 'object',
        required: ['idea', 'source', 'why'],
        properties: { idea: { type: 'string' }, source: { type: 'string', description: 'paper / system / derivation, cited' }, why: { type: 'string', description: 'which term or constraint it improves, and by how much' } },
      },
    },
  },
}
const PRESENTATION = { type: 'object', required: ['block'], properties: { block: { type: 'string', description: 'the ARCHITECTURE DECISION REQUIRED block, verbatim' } } }

phase('Gate')
if (revised) {
  // An ADR in progress has already written its pending amendments into the design, so the working copy is unapproved by
  // construction: the committed design it started from must carry a human grant. The amendments are granted at the end.
  const base = await agent(common('You run ALONE — you MAY run this one command. Change nothing.') + `
Run exactly: node agents-architetura/tools/factory/approved-base.mjs ${DESIGN} --json
Return the JSON it prints EXACTLY as printed.`,
    { label: 'gate:approval-base', phase: 'Gate', schema: APPROVAL_CHECK, agentType: ROLE.verifier })
  if (!approvedAll(base, [DESIGN])) throw new Error(`the committed ${DESIGN} is not granted by a human (${clean(base?.docs?.[0]?.reason ?? 'no result', 200)}) — grant the design this ADR started from first`)
} else await requireApprovedDesign('Gate')
const n = revised ? { next: revised[1] } : await agent(common('READ-ONLY.') + `
List ${DESIGN_DIR}/adr/ and return the next ADR id (ADR-NNNN, four digits): one more than the highest existing number, ADR-0001 if none.`,
  { label: 'adr:next', phase: 'Gate', schema: NEXT, agentType: ROLE.readonly })
const id = /^ADR-\d{4}$/.test(n?.next ?? '') ? n.next : null
if (!id) throw new Error('could not determine the next ADR number')
const ADR = `${DESIGN_DIR}/adr/${id}-${slug}.md`
log(`ADR: ${ADR}${revised ? ` — ${SPLIT ? 'splitting' : 'revising'} with ${references.length} reference(s): Research and Draft skipped` : ''}`)

const ownFor = adr => `OWNED PATHS: ${adr}, ${DESIGN}, ${DESIGN_HISTORY}, ${DESIGN_DIR}/SPEC-*.md and their -history.md files (only).`
const OWN = ownFor(ADR)
const DRAFT_RULES = `
- Use the template agents-architetura/docs/templates/ADR.md; every section is mandatory; fill in the ARCHITECTURE DECISION REQUIRED block at the top.
- Mathematical Model: the decomposition this change actually moves (e.g. T_total = T_compute + T_communication + T_synchronization +
  T_scheduler + T_overhead) with the variables that matter, evaluated with numbers; say which term moves, by how much, and when the
  gain disappears. No decorative mathematics.
- Falsification Criteria: the experiment and observed result that would prove the idea wrong. Local Experiment before any Vast.ai cost.
- Write the concrete amendments into ${DESIGN} (and any spec) so the human approves exact text; record each spec amendment on the record:
  node agents-architetura/tools/factory/freeze.mjs amend <spec> --reason "<why>" --by ${id}
- Do NOT write a Status line claiming approval. Recommendation is PROCEED or DO NOT PROCEED with the reason.`

if (!revised) {
  phase('Research')
  const surveys = await parallel(angles.map(([key, angle]) => () => agent(common('READ-ONLY on the repository. You MAY use web search and fetch.') + `
Research the problem below for ${id} from ONE angle: ${angle}. Other researchers cover the other angles; go deep on yours.
First locate the actual bottleneck in this system (${DESIGN}, ${DESIGN_DIR}/CURRENT_ARCHITECTURE.md, the code, any measurements) and
state the evidence. Then find the strongest candidate mechanisms — papers, existing systems, implementations, cited — and trace each:
idea -> mechanism -> effect on OUR bottleneck -> trade-offs -> the smallest local experiment that would refute it. Include "do
nothing / tune what exists". Never recommend a mechanism because a paper uses it.
${fence('problem', { problem: args.problem, evidence: args.evidence ?? null, from: args.from ?? null }, 8000)}`,
    { label: `research:${key}`, phase: 'Research', schema: RESEARCH, agentType: ROLE.researcher })))
  const research = angles.map(([key], i) => surveys[i] ? { angle: key, ...surveys[i] } : null).filter(Boolean)
  if (!research.some(r => r.options?.length)) throw new Error('research produced no options')
  log(`Research: ${research.length}/${angles.length} angles reported, ${research.reduce((n, r) => n + (r.options?.length ?? 0), 0)} options`)

  phase('Draft')
  await agent(common(OWN) + DRAFT_RULES + `
Write ${ADR} for the problem below from ALL the research (one survey per angle). Derive the architecture from first principles:
hard constraints are constraints — prove any that cannot be met and design the strongest achievable alternative instead of
breaking another constraint. Add an "Ideas considered" section (adopted or rejected, each with its reason and number) and a
short changelog per round.
${fence('problem', { problem: args.problem, evidence: args.evidence ?? null }, 6000)}
${fence('research', research, 60000)}`,
    { label: 'draft', phase: 'Draft', schema: RESULT, agentType: ROLE.architect })
}

const DOC_LINES = {
  type: 'object',
  required: ['files'],
  properties: { files: { type: 'array', items: { type: 'object', required: ['path', 'exists', 'lines'], properties: { path: { type: 'string' }, exists: { type: 'boolean' }, lines: { type: 'integer' } } } } },
}
const SPLIT_RESULT = {
  type: 'object',
  required: ['adrs'],
  properties: {
    adrs: { type: 'array', items: { type: 'object', required: ['path', 'decision'], properties: { path: { type: 'string', description: 'docs/design/adr/ADR-NNNN-<slug>.md' }, decision: { type: 'string', description: 'the one decision this ADR asks the human to grant' } } } },
    notes: { type: 'array', items: { type: 'string' }, description: 'docs/research/*.md files written for the derivations and evidence' },
  },
}
// The script decides sizes: a tool counts the lines, an agent only relays its JSON.
const linesOf = async (paths, label) => {
  const r = await agent(common('You run ALONE — you MAY run this one command. Change nothing.') + `
Run exactly: node agents-architetura/tools/factory/doc-lines.mjs ${paths.join(' ')} --json
Return the JSON it prints EXACTLY as printed.`,
    { label: `size:${label}`, phase: 'Split', schema: DOC_LINES, agentType: ROLE.verifier })
  return new Map((r?.files ?? []).filter(f => paths.includes(f?.path)).map(f => [f.path, f.exists ? f.lines : 0]))
}
const splitAdr = async (path, why, open = []) => {
  phase('Split')
  const [, num, base] = path.match(ADR_PATH_RE)
  const r = await agent(common(`OWNED PATHS: ${DESIGN_DIR}/adr/, docs/research/, ${DESIGN}, ${DESIGN_HISTORY}, ${DESIGN_DIR}/SPEC-*.md and their -history.md files (only).`) + DRAFT_RULES + `
${path} is ${why}. Split it into 2 to 4 ADRs, each ONE decision a human can grant or refuse on its own, each at most
${MAX_ADR_LINES} lines and with every template section. The first decision stays at ${path}. The others take the next free
numbers in ${DESIGN_DIR}/adr/ (list it; never reuse a number), each named for its decision. Long derivations, surveys and
evidence tables move into docs/research/<slug>.md: notes the ADRs cite, not decisions. Keep every number, falsification
criterion and amendment; attribute each design or spec amendment to the ADR that decides it (freeze.mjs amend ... --by that
ADR's id). Cross-reference the ADRs ("depends on ADR-NNNN"). Change no decision: an open finding goes under Risks in the ADR
it concerns.
${fence('open_findings', open.slice(0, 30), 12000)}`,
    { label: `split:${num}`, phase: 'Split', schema: SPLIT_RESULT, agentType: ROLE.architect })
  const parts = (r?.adrs ?? []).map(a => String(a?.path ?? ''))
  const numbers = parts.map(x => x.match(ADR_PATH_RE)?.[1])
  if (parts.length < 2 || parts.length > 4 || numbers.some(x => !x) || new Set(numbers).size !== parts.length || parts[0] !== path) {
    throw new Error(`the split of ${path} returned ${JSON.stringify(clean(JSON.stringify(parts), 300))}: 2-4 ADR paths, the first ${path}, distinct numbers — resume this run`)
  }
  const sizes = await linesOf(parts, `split-${num}`)
  const missing = parts.filter(x => !sizes.get(x))
  if (missing.length) throw new Error(`the split of ${path} did not write ${missing.join(', ')} — resume this run`)
  const over = parts.filter(x => sizes.get(x) > MAX_ADR_LINES)
  if (over.length) log(`Split of ${base}: ${over.map(x => `${x} (${sizes.get(x)} lines)`).join(', ')} still over ${MAX_ADR_LINES} lines — reviewed anyway; a stalled review splits again`)
  log(`Split ${path} (${why}) into ${parts.join(', ')}`)
  return parts
}

const LENSES = [
  ['first-principles', 'Is the bottleneck real (evidence) and the derivation a mechanism rather than an appeal to authority? Does the math say which term moves and by how much, with numbers? Are the falsification criteria observable and the local experiment able to refute the idea before any paid run? Is the cost estimate derived?'],
  ['alternatives-and-risk', 'Were genuinely different alternatives — including doing nothing or tuning what exists — compared on the same model? Migration impact, rollback, risks: honest and complete? Does it preserve what already works?'],
  ['consistency', `Do the amendments in ${DESIGN} and the specs implement exactly what the ADR decides — nothing more (scope creep) and nothing less? Are spec amendments recorded with "--by" this ADR's id?`],
  ['frontier', 'Search the web for what the draft missed: newer papers, systems and hardware facts, mechanisms from other fields, a reframing that meets every hard constraint more cheaply. Bring them as ideas; one that would materially change the decision is also a major finding.'],
]
// One ADR's review: the lenses with ideas, the architect's fixes, then the references and the last round's ideas recorded.
const reviewAdr = async adr => {
  const adrId = adr.match(ADR_PATH_RE)[1]
  const ideasByRound = new Map()
  const review = await reviewUntilClean({
    label: adrId,
    lenses: LENSES,
    // the ADR, the design and the specs are this change; research notes and everything else are not
    ownsFile: f => [adr, DESIGN, DESIGN_HISTORY].includes(f) || f.startsWith(`${DESIGN_DIR}/SPEC-`),
    review: (name, lens, round) => agent(common('READ-ONLY adversarial reviewer. You MAY use web search and fetch.') + `
Review ${adr}, the research notes it cites and the amendments it made (round ${round}, lens ${name}). ${lens}
It decides ONE thing; other ADRs it names decide the rest — review this decision, not theirs.
${round >= FREEZE_ROUND ? `The decision and its mechanism are frozen from round ${FREEZE_ROUND}: file a finding only for a defect in the decision as written
(wrong, inconsistent, unsupported or unsafe), never "a better mechanism exists". Return a better mechanism as an idea; it becomes a
follow-up.` : 'Besides findings, return "ideas": mechanisms, papers or framings the draft has not considered (cited). Think beyond the draft.'}${references.length ? `
The human brought these references. VERIFY each one on the web: does it exist, and do its claims and numbers match the source?
A false or unverifiable reference is a finding. A real one that the ADR must cite, compare against or answer is an idea, or a
major finding when it would change the decision.
${fence('references', references, 12000)}` : ''}`,
      { label: `review:${adrId}:${name}:r${round}`, phase: 'Review', schema: REVIEW_WITH_IDEAS, agentType: ROLE.reviewer }).then(r => {
      if (r?.ideas?.length) ideasByRound.set(round, [...(ideasByRound.get(round) ?? []), ...r.ideas.map(x => ({ lens: name, ...x }))])
      return r
    }),
    fix: (findings, round) => agent(common(ownFor(adr)) + DRAFT_RULES + `
Round ${round}. Fix these findings in ${adr} at their cause, blocker and major first (a minor is cheap: fix it too unless it
conflicts); refute false ones in the ADR's Risks section with evidence. Keep the ADR within ${MAX_ADR_LINES} lines: move a long
derivation into docs/research/ rather than growing the decision. ${round >= FREEZE_ROUND ? `From round ${FREEZE_ROUND} the decision and
its mechanism are frozen: correct the text, the numbers and the amendments, and change no mechanism. Record every reviewer idea
under "Ideas considered" as a follow-up (with its reason and number), without adopting it. A finding that can only be fixed by a
new mechanism goes under Risks as open, for the human.` : `Then evaluate every idea the reviewers brought: adopt it and
redo the derivation it touches, or reject it with the reason and the number that ranks it lower.`} Record both under
"Ideas considered" and add this round to the changelog.${references.length ? ` Place every reference the reviewers verified in
the ADR (related work or Ideas considered) with your verdict and its numbers; name the ones that could not be verified and
leave them out.` : ''}
${fence('reviewer_findings', findings, 20000)}
${fence('reviewer_ideas', ideasByRound.get(round) ?? [], 12000)}`,
      { label: `fix:${adrId}:r${round}`, phase: 'Review', schema: RESULT, agentType: ROLE.architect }),
  })
  // The round that converged may still have brought ideas: the architect records its verdict on each (no design change
  // without another review, so an idea that would change the decision must have been filed as a major finding).
  const lastIdeas = ideasByRound.get(review.rounds) ?? []
  if (revised && references.length && !review.stalled) {
    // the references are placed whether or not a fix round ran: a review that filed them only as ideas, or converged at
    // once, must not leave them out of the ADR
    await agent(common(`OWNED PATHS: ${adr} (only).`) + `
Place the human's references that concern this ADR's decision in ${adr}: every one the reviewers verified goes into the related
work or "Ideas considered" with your verdict and its numbers; name the ones that could not be verified and leave them out.
Record your verdict on every reviewer idea below as well. Do not change the decision itself: a reference that would change it
must already have been filed as a finding.
${fence('references', references, 12000)}
${fence('reviewer_ideas', [...ideasByRound.values()].flat(), 16000)}`,
      { label: `references:place:${adrId}`, phase: 'Review', schema: RESULT, agentType: ROLE.architect })
  } else if (review.converged && lastIdeas.length) {
    await agent(common(`OWNED PATHS: ${adr} (only).`) + `
The review converged. Record your verdict on each idea of its last round under "Ideas considered" in ${adr} (adopt later as a
follow-up, or reject, each with the reason); do not change the decision itself.
${fence('reviewer_ideas', lastIdeas, 12000)}`,
      { label: `ideas:record:${adrId}`, phase: 'Review', schema: RESULT, agentType: ROLE.architect })
  }
  return { adr, review }
}
// A review or split that threw is reported, never counted as a decision.
const failedReview = adr => {
  log(`${adr}: its review or split failed — resume this run`)
  return { adr, failed: true, review: { converged: false, rounds: 0, findingsPerRound: ['failed'] } }
}
// Review an ADR; a stalled review splits it (up to MAX_SPLIT_DEPTH levels) and reviews the parts on their own.
const decide = async (adr, depth) => {
  phase('Review')
  const r = await reviewAdr(adr)
  if (!r.review.stalled || depth >= MAX_SPLIT_DEPTH) return [r]
  const parts = await splitAdr(adr, `stalled under review (${r.review.findingsPerRound.join(' -> ')} blocker/major findings)`, r.review.open ?? [])
  return (await parallel(parts.map(part => () => decide(part, depth + 1)))).flatMap((x, i) => x ?? [failedReview(parts[i])])
}

let adrs = [ADR]
if (SPLIT) adrs = await splitAdr(ADR, 'to be split, as the human asked')
else {
  const size = (await linesOf([ADR], id)).get(ADR) ?? 0
  if (!size) throw new Error(`${ADR} does not exist — ${revised ? 'check the path' : 'the draft wrote nothing; resume this run'}`)
  if (size > MAX_ADR_LINES) adrs = await splitAdr(ADR, `${size} lines, over the ${MAX_ADR_LINES}-line budget`)
}
const decided = (await parallel(adrs.map(a => () => decide(a, SPLIT || adrs.length > 1 ? 1 : 0)))).flatMap((x, i) => x ?? [failedReview(adrs[i])])

phase('Present')
const blocks = await parallel(decided.map(d => () => agent(common('READ-ONLY.') + `
Return the "ARCHITECTURE DECISION REQUIRED" block at the top of ${d.adr}, verbatim.`,
  { label: `present:${d.adr.match(ADR_PATH_RE)[1]}`, phase: 'Present', schema: PRESENTATION, agentType: ROLE.readonly })))
decided.forEach((d, i) => log(clean(blocks[i]?.block ?? `(no decision block found in ${d.adr})`, 6000)))
const command = `node agents-architetura/tools/factory/approve.mjs grant ${decided.map(d => d.adr).join(' ')} ${DESIGN}`
log(`Awaiting explicit human approval: ${command}`)
const first = decided.find(d => d.adr === ADR) ?? decided[0]
return {
  ok: decided.every(d => d.review.converged),
  awaitingHuman: true,
  adr: ADR,
  id,
  revised: !!REVISE,
  split: decided.length > 1,
  references: references.length,
  review: { converged: first.review.converged, rounds: first.review.rounds },
  adrs: decided.map(d => ({ adr: d.adr, converged: d.review.converged, rounds: d.review.rounds, stalled: !!d.review.stalled, failed: !!d.failed, findingsPerRound: d.review.findingsPerRound, open: d.review.open ?? [], house: d.review.house ?? [] })),
  command,
}
