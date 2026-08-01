import type { Metadata, Viewport } from 'next';
import './globals.css';
import { TooltipProvider } from '@/components/ui/tooltip';
import { SiteFooter, SiteHeader } from '@/components/site-header';
import { StudioProvider } from '@/components/studio-provider';

export const metadata: Metadata = {
  title: 'Llama Local Lab — Zero-Cost Llama Evaluation Studio',
  description:
    'Plan quantization, generate GGUF conversion commands, model inference cost, and run Llama in your browser. No server, no API key, no data leaves your machine.',
  applicationName: 'Llama Local Lab',
  keywords: [
    'Llama 4',
    'Llama 4 Scout',
    'Llama 4 Maverick',
    'Llama 3.3',
    'GGUF',
    'quantization',
    'llama.cpp',
    'WebGPU',
    'WebLLM',
    'inference cost',
  ],
  authors: [{ name: 'Llama Local Lab contributors' }],
  openGraph: {
    title: 'Llama Local Lab',
    description:
      'Which quant runs on my GPU? How much quality do I lose? What does inference actually cost? Answer all three in the browser.',
    type: 'website',
  },
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  themeColor: '#0064E0',
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-background font-sans">
        <TooltipProvider delayDuration={200} skipDelayDuration={300}>
          <StudioProvider>
            <a
              href="#studio"
              className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-primary focus:px-4 focus:py-2 focus:text-primary-foreground"
            >
              Skip to the studio
            </a>
            <SiteHeader />
            <main>{children}</main>
            <SiteFooter />
          </StudioProvider>
        </TooltipProvider>
      </body>
    </html>
  );
}
