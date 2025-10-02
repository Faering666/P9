import SatelliteViewer from '@/components/SatelliteViewer';

export default function Home() {
    return (
        <main className="min-h-screen p-8">
            <div className="max-w-6xl mx-auto">
                <h1 className="text-4xl font-bold text-center mb-8">
                    Satellite Image Viewer
                </h1>
                <SatelliteViewer />
            </div>
        </main>
    )
}