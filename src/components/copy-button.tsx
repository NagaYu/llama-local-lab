'use client';

import * as React from 'react';
import { Check, Copy } from 'lucide-react';
import { Button, type ButtonProps } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { copyText } from '@/lib/utils';

interface CopyButtonProps extends Omit<ButtonProps, 'onClick' | 'children' | 'value'> {
  /** The text to copy, or a thunk for content that is expensive to build. */
  value: string | (() => string);
  label?: string;
  /** Show the label text next to the icon instead of icon-only. */
  showLabel?: boolean;
}

export function CopyButton({
  value,
  label = 'Copy',
  showLabel = false,
  className,
  variant = 'ghost',
  size,
  ...props
}: CopyButtonProps) {
  const [copied, setCopied] = React.useState(false);
  const [failed, setFailed] = React.useState(false);
  const timer = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  React.useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  const onCopy = async () => {
    const text = typeof value === 'function' ? value() : value;
    const ok = await copyText(text);
    setCopied(ok);
    setFailed(!ok);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      setCopied(false);
      setFailed(false);
    }, 1800);
  };

  return (
    <Button
      type="button"
      variant={variant}
      size={size ?? (showLabel ? 'sm' : 'icon-sm')}
      onClick={onCopy}
      aria-label={label}
      className={cn(copied && 'text-success', failed && 'text-danger', className)}
      {...props}
    >
      {copied ? <Check /> : <Copy />}
      {showLabel && <span>{failed ? 'Press ⌘C' : copied ? 'Copied' : label}</span>}
    </Button>
  );
}
