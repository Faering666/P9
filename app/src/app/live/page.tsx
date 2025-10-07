import LiveFeed from '@/components/LiveFeed';

export default function LivePage() {
    return (
        <main className="min-h-screen p-4">
            <div className="max-w-full mx-auto">
                <h1 className="text-4xl font-bold text-center mb-6">
                    Live Feed
                </h1>
                <LiveFeed />
            </div>
        </main>
    )
}