import MemoryViewer from "@/components/viewer/MemoryViewer";

/** Walk through one memory: splat + spatial soundscape. */
export default async function MemoryPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <MemoryViewer memoryId={id} />;
}
