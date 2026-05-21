import { Injectable } from '@nestjs/common';
import { Agent } from '@mastra/core/agent';
import { createOpenAI } from '@ai-sdk/openai';
import { LlmService } from '../llm/llm.service';
import { SearchService } from '../index/search.service';
import { buildSystemPrompt, ChatContext } from './prompt.builder';
import { createSearchFilesTool } from './tools/search-files.tool';
import { createReadFileTool } from './tools/read-file.mastra';
import { createSearchArticlesTool } from './tools/search-articles.mastra';
import { createReadArticleTool } from './tools/read-article.tool';
import { join } from 'path';

export interface MastraStreamEvent {
  type: 'tool_call' | 'tool_result' | 'text_delta' | 'steps' | 'done' | 'error';
  [key: string]: any;
}

@Injectable()
export class MastraService {
  private readonly sourceRoot: string;
  private readonly articlesRoot: string;

  constructor(
    private readonly llmService: LlmService,
    private readonly searchService: SearchService,
  ) {
    this.sourceRoot = join(__dirname, '..', 'claude-code-source');
    this.articlesRoot = join(__dirname, '..', 'articles');
  }

  async *run(
    conversationMessages: { role: string; content: string }[],
    context: ChatContext,
  ): AsyncIterable<MastraStreamEvent> {
    // Get model config from DB and build a typed AI SDK provider
    const config = await this.llmService.getChatProviderConfig();
    const openaiProvider = createOpenAI({
      apiKey: config.apiKey,
      ...(config.baseUrl ? { baseURL: config.baseUrl } : {}),
    });
    // Use .chat() to force Chat Completions API, not Responses API
    const model = openaiProvider.chat(config.model);

    // Create tools
    const tools = {
      search_files: createSearchFilesTool(this.searchService),
      read_file: createReadFileTool(this.sourceRoot),
      search_articles: createSearchArticlesTool(this.searchService),
      read_article: createReadArticleTool(this.articlesRoot),
    };

    // Create agent
    const agent = new Agent({
      id: 'claude-harness-assistant',
      name: 'Claude Harness Assistant',
      instructions: buildSystemPrompt(context),
      model,
      tools,
    });

    // Build messages for Mastra
    const messages = conversationMessages.map((m) => ({
      role: m.role as 'user' | 'assistant' | 'system',
      content: m.content,
    })) as any;

    // Track steps for thinking process
    const steps: { tool: string; args: any; resultPreview: string }[] = [];
    let hasText = false;

    try {
      const result = await agent.stream(messages, { maxSteps: 15 });

      // Read the full stream from Mastra and translate events
      for await (const chunk of result.fullStream) {
        if (chunk.type === 'tool-call') {
          const payload = (chunk as any).payload;
          const toolCallEvent: MastraStreamEvent = {
            type: 'tool_call',
            name: payload.toolName,
            args: payload.args,
          };
          yield toolCallEvent;
        } else if (chunk.type === 'tool-result') {
          const payload = (chunk as any).payload;
          const resultStr = typeof payload.result === 'string'
            ? payload.result
            : JSON.stringify(payload.result);
          const preview = resultStr.length > 200 ? resultStr.slice(0, 200) + '...' : resultStr;
          steps.push({ tool: payload.toolName, args: payload.args, resultPreview: preview });
          const toolResultEvent: MastraStreamEvent = {
            type: 'tool_result',
            name: payload.toolName,
            result: preview,
          };
          yield toolResultEvent;
        } else if (chunk.type === 'text-delta') {
          const payload = (chunk as any).payload;
          hasText = true;
          yield { type: 'text_delta', delta: payload.text };
        }
      }

      // Emit thinking steps summary
      if (steps.length > 0) {
        yield { type: 'steps', steps };
      }

      // Fallback: if the agent exhausted max steps without producing text,
      // make a final LLM call to summarize what was found
      if (!hasText && steps.length > 0) {
        const toolSummary = steps
          .map((s) => `[${s.tool}(${JSON.stringify(s.args)})] → ${s.resultPreview}`)
          .join('\n');
        const fallbackMessages = [
          ...messages,
          {
            role: 'assistant' as const,
            content: `I searched but couldn't produce a response. Here are the tool results:\n${toolSummary}`,
          },
          {
            role: 'user' as const,
            content: 'Based on the tool results above, please provide your answer to the original question. If you could not find what you were looking for, explain what you found and suggest alternatives.',
          },
        ];

        const fallbackResult = await agent.stream(fallbackMessages);
        for await (const chunk of fallbackResult.fullStream) {
          if (chunk.type === 'text-delta') {
            const payload = (chunk as any).payload;
            yield { type: 'text_delta', delta: payload.text };
          }
        }
      }

      // Extract token usage from the stream result (AI SDK exposes inputTokens/outputTokens)
      const usage = await (result as any).usage;
      yield {
        type: 'done',
        usage: {
          inputTokens: usage?.inputTokens ?? usage?.promptTokens ?? 0,
          outputTokens: usage?.outputTokens ?? usage?.completionTokens ?? 0,
        },
      };
    } catch (error) {
      yield { type: 'error', message: (error as Error).message };
    }
  }
}
