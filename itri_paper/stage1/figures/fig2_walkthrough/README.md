# Walkthrough acceptance dependency

The experiment brief requires a **validated chest-pain script** for Figure 2.
No current script has the required physician approvals. Therefore no dialogue,
FHIR snippet, or clinician screenshot is labelled validated or publication-ready.

After both reviewers approve the current script hash and its expected path:

1. Run that exact script through E2 and retain its session/source hashes.
2. Export the synthetic dialogue excerpt, extracted facts/evidence, and captured
   QuestionnaireResponse, eligible Observation, and Composition.
3. Capture the actual clinician view against the same isolated synthetic record.
4. Include the run identifier and review hashes alongside the separate files.

Do not substitute a generated interface mockup or an unreviewed smoke session for
the requested clinical walkthrough. The source experiment artifacts remain under
the immutable `experiments/results/` run directories.
