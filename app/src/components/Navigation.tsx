'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex gap-4 h-14 items-center">
          <Link
            href="/"
            className={`px-4 py-2 rounded-lg transition-colors ${
              pathname === '/'
                ? 'bg-blue-500 text-white'
                : 'text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            Satellite Viewer
          </Link>
          <Link
            href="/live"
            className={`px-4 py-2 rounded-lg transition-colors ${
              pathname === '/live'
                ? 'bg-blue-500 text-white'
                : 'text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            Live Feed
          </Link>
        </div>
      </div>
    </nav>
  );
}