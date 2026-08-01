'use client';

import * as React from 'react';
import {
  Boxes,
  ClipboardCheck,
  Layers,
  MessageSquareCode,
  Receipt,
  TerminalSquare,
} from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useStudio, type StudioTab } from '@/components/studio-provider';
import { ModelSelectorPanel } from '@/components/features/model-selector';
import { QuantPlannerPanel } from '@/components/features/quant-planner';
import { GgufGeneratorPanel } from '@/components/features/gguf-generator';
import { CostCalculatorPanel } from '@/components/features/cost-calculator';
import { PlaygroundPanel } from '@/components/features/playground';
import { EvalSuitePanel } from '@/components/features/eval-suite';

const TABS: {
  id: StudioTab;
  label: string;
  short: string;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  { id: 'models', label: 'Models', short: '01', icon: Boxes },
  { id: 'quant', label: 'Quantization', short: '02', icon: Layers },
  { id: 'gguf', label: 'GGUF commands', short: '03', icon: TerminalSquare },
  { id: 'cost', label: 'Cost', short: '04', icon: Receipt },
  { id: 'playground', label: 'Playground', short: '05', icon: MessageSquareCode },
  { id: 'eval', label: 'Evaluation', short: '06', icon: ClipboardCheck },
];

export function StudioTabs() {
  const { tab, goTo } = useStudio();

  return (
    <div id="studio" className="scroll-mt-14">
      <Tabs value={tab} onValueChange={(v) => goTo(v as StudioTab)} className="w-full">
        <div className="sticky top-14 z-30 border-b bg-background/90 backdrop-blur supports-[backdrop-filter]:bg-background/75">
          <div className="container">
            <TabsList className="h-auto w-full justify-start overflow-x-auto scrollbar-thin py-1.5">
              {TABS.map(({ id, label, short, icon: Icon }) => (
                <TabsTrigger key={id} value={id} className="gap-2">
                  <Icon className="h-4 w-4" />
                  <span className="font-mono text-[10px] text-muted-foreground">{short}</span>
                  <span>{label}</span>
                </TabsTrigger>
              ))}
            </TabsList>
          </div>
        </div>

        <div className="container py-8 sm:py-10">
          <TabsContent value="models" className="mt-0 animate-fade-in">
            <ModelSelectorPanel />
          </TabsContent>
          <TabsContent value="quant" className="mt-0 animate-fade-in">
            <QuantPlannerPanel />
          </TabsContent>
          <TabsContent value="gguf" className="mt-0 animate-fade-in">
            <GgufGeneratorPanel />
          </TabsContent>
          <TabsContent value="cost" className="mt-0 animate-fade-in">
            <CostCalculatorPanel />
          </TabsContent>
          <TabsContent value="playground" className="mt-0 animate-fade-in">
            <PlaygroundPanel />
          </TabsContent>
          <TabsContent value="eval" className="mt-0 animate-fade-in">
            <EvalSuitePanel />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
