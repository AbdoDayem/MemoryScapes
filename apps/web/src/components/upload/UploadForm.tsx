"use client";

/**
 * Upload flow:
 * 1. POST /memories -> presigned URLs
 * 2. Chunked multipart upload straight to storage
 * 3. POST /memories/:id/process
 * 4. Subscribe to /memories/:id/events (SSE) for stage progress
 *
 * Client-side checks: duration, resolution, audio present. Show capture guidance
 * (move slowly, orbit, avoid rotating in place).
 */
export default function UploadForm() {
  return <form>Upload form (not implemented)</form>;
}
