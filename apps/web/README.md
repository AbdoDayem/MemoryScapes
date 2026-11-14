# MemoryScapes web

Next.js app with three surfaces:

| Route | Component | Purpose |
|---|---|---|
| `/` | `components/globe/MemoryGlobe.tsx` | Globe of public memories |
| `/upload` | `components/upload/UploadForm.tsx` | Upload video or panoramas plus audio |
| `/memories/[id]` | `components/viewer/MemoryViewer.tsx` | Walk through a memory with spatial audio |

`components/audio/SpatialAudioEngine.ts` places each audio source from the manifest
in 3D with Web Audio HRTF panning.

```bash
npm install        # from the repo root
npm run dev:web
```
