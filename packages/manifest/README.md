# Memory manifest

The manifest is the contract between the processing pipeline and the viewer.
`memory.schema.json` is the source of truth. `index.ts` mirrors it for the web app,
and the workers validate against the JSON Schema before publishing.

Bump `version` whenever a memory is reprocessed so old links keep working.
