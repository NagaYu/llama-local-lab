/**
 * Llama-native function calling.
 *
 * WebLLM's OpenAI-compatible `tools` parameter only works for the handful of
 * models in its `functionCallingModelIds` list — none of which are Llama 3.2.
 * Meta's own approach is prompt-level: you describe the functions in the system
 * message and the model replies with a JSON object. That works on every Llama
 * checkpoint, so it is what this studio does, and it is also what you would
 * ship if you served these weights yourself.
 */

export const DEFAULT_TOOLS_JSON = `[
  {
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "description": "Get the current weather for a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
  }
]`;

export interface ParsedTools {
  tools: unknown[];
  error: string | null;
}

export function parseTools(json: string): ParsedTools {
  const trimmed = json.trim();
  if (!trimmed) return { tools: [], error: null };
  try {
    const parsed = JSON.parse(trimmed);
    if (!Array.isArray(parsed)) {
      return { tools: [], error: 'Tools must be a JSON array.' };
    }
    for (const [i, entry] of parsed.entries()) {
      const fn = (entry as { function?: { name?: string } })?.function;
      if (!fn?.name) {
        return { tools: [], error: `Entry ${i} is missing "function.name".` };
      }
    }
    return { tools: parsed, error: null };
  } catch (err) {
    return { tools: [], error: err instanceof Error ? err.message : 'Invalid JSON.' };
  }
}

/**
 * Build the tool-calling preamble in the format Meta documents for Llama 3.1+.
 * Appended to the user's own system prompt rather than replacing it.
 */
export function buildToolSystemPrompt(tools: unknown[]): string {
  if (tools.length === 0) return '';
  const rendered = tools
    .map((t) => JSON.stringify((t as { function?: unknown }).function ?? t, null, 2))
    .join('\n\n');

  return [
    'You have access to the following functions:',
    '',
    rendered,
    '',
    'If you choose to call a function, reply with ONLY a JSON object in this exact form and nothing else:',
    '{"name": "function_name", "parameters": {"argument": "value"}}',
    '',
    'Do not add any prose, explanation or markdown fences around the JSON.',
    'If no function is relevant, answer the user normally in plain text.',
  ].join('\n');
}

export interface ToolCall {
  name: string;
  arguments: string;
}

/**
 * Pull a tool call out of a model response.
 *
 * Handles the `<|python_tag|>` prefix Llama 3.1 emits for built-in tools, bare
 * JSON, and JSON wrapped in a markdown fence — all three show up in practice at
 * low quantization levels.
 */
export function extractToolCall(text: string, knownNames: string[]): ToolCall | null {
  let candidate = text.trim();

  const pythonTag = candidate.indexOf('<|python_tag|>');
  if (pythonTag !== -1) {
    candidate = candidate.slice(pythonTag + '<|python_tag|>'.length).trim();
  }

  const fence = candidate.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fence) candidate = fence[1].trim();

  const start = candidate.indexOf('{');
  const end = candidate.lastIndexOf('}');
  if (start === -1 || end <= start) return null;

  try {
    const parsed = JSON.parse(candidate.slice(start, end + 1)) as {
      name?: string;
      parameters?: unknown;
      arguments?: unknown;
    };
    if (!parsed.name) return null;
    if (knownNames.length > 0 && !knownNames.includes(parsed.name)) return null;
    const args = parsed.parameters ?? parsed.arguments ?? {};
    return { name: parsed.name, arguments: JSON.stringify(args, null, 2) };
  } catch {
    return null;
  }
}

export function toolNames(tools: unknown[]): string[] {
  return tools
    .map((t) => (t as { function?: { name?: string } })?.function?.name)
    .filter((n): n is string => Boolean(n));
}
