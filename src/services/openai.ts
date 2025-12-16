import OpenAI, { AzureOpenAI } from "openai";
import {
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from "openai/resources";

import { kEnvs } from "../utils/env";
import { withDefault } from "../utils/base";
import { ChatCompletionCreateParamsBase } from "openai/resources/chat/completions";
import { Logger } from "../utils/log";
import { kProxyAgent } from "./proxy";
import { isNotEmpty } from "../utils/is";

export interface ChatOptions {
  user: string;
  system?: string;
  model?: ChatCompletionCreateParamsBase["model"];
  tools?: Array<ChatCompletionTool>;
  jsonMode?: boolean;
  requestId?: string;
  trace?: boolean;
  enableSearch?: boolean;
}

class OpenAIClient {
  traceInput = false;
  traceOutput = true;
  private _logger = Logger.create({ tag: "Open AI" });

  deployment?: string;

  private _client?: OpenAI;
  private _init() {
    this.deployment = kEnvs.AZURE_OPENAI_DEPLOYMENT;
    if (!this._client) {
      this._client = kEnvs.AZURE_OPENAI_API_KEY
        ? new AzureOpenAI({
            httpAgent: kProxyAgent,
            deployment: this.deployment,
          })
        : new OpenAI({ httpAgent: kProxyAgent });
    }
  }

  private _abortCallbacks: Record<string, VoidFunction> = {
    // requestId: abortStreamCallback
  };

  cancel(requestId: string) {
    this._init();
    if (this._abortCallbacks[requestId]) {
      this._abortCallbacks[requestId]();
      delete this._abortCallbacks[requestId];
    }
  }

  async chat(options: ChatOptions) {
    this._init();
    let {
      user,
      system,
      tools,
      jsonMode,
      requestId,
      trace = false,
      model = this.deployment ?? kEnvs.OPENAI_MODEL ?? "gpt-4o",
    } = options;
    if (trace && this.traceInput) {
      this._logger.log(
        `🔥 onAskAI\n🤖️ System: ${system ?? "None"}\n😊 User: ${user}`.trim()
      );
    }
    const systemMsg: ChatCompletionMessageParam[] = isNotEmpty(system)
      ? [{ role: "system", content: system! }]
      : [];
    let signal: AbortSignal | undefined;
    if (requestId) {
      const controller = new AbortController();
      this._abortCallbacks[requestId] = () => controller.abort();
      signal = controller.signal;
    }
    const chatCompletion = await this._client!.chat.completions.create(
      {
        model,
        tools,
        messages: [...systemMsg, { role: "user", content: user }],
        response_format: jsonMode ? { type: "json_object" } : undefined,
      },
      { signal }
    ).catch((e) => {
      this._logger.error("LLM 响应异常", e);
      return null;
    });
    if (requestId) {
      delete this._abortCallbacks[requestId];
    }
    const message = chatCompletion?.choices?.[0]?.message;
    if (trace && this.traceOutput) {
      this._logger.log(`✅ Answer: ${message?.content ?? "None"}`.trim());
    }
    return message;
  }

  async chatStream(
    options: ChatOptions & {
      onStream?: (text: string) => void;
    }
  ) {
    this._init();
    let {
      user,
      system,
      tools,
      jsonMode,
      requestId,
      onStream,
      trace = false,
      model = this.deployment ?? kEnvs.OPENAI_MODEL ?? "gpt-4o",
      enableSearch = kEnvs.QWEN_ENABLE_SEARCH,
    } = options;
    
    this._logger.log(`🔥 开始流式AI请求 - 请求ID: ${requestId || "无"}, 模型: ${model}`);
    
    if (trace && this.traceInput) {
      this._logger.log(
        `🔥 onAskAI\n🤖️ System: ${system ?? "None"}\n😊 User: ${user}`.trim()
      );
    }
    const systemMsg: ChatCompletionMessageParam[] = isNotEmpty(system)
      ? [{ role: "system", content: system! }]
      : [];
    
    this._logger.log(`🔥 发送请求到OpenAI - 工具数量: ${tools?.length || 0}, JSON模式: ${jsonMode ? "是" : "否"}, 搜索功能: ${enableSearch ? "是" : "否"}`);
    
    const stream = await this._client!.chat.completions.create({
      model,
      tools,
      stream: true,
      messages: [...systemMsg, { role: "user", content: user }],
      response_format: jsonMode ? { type: "json_object" } : undefined,
      ...(enableSearch && { enable_search: true })
    }).catch((e) => {
      this._logger.error("LLM 响应异常", e);
      return null;
    });
    if (!stream) {
      this._logger.error(`🔥 流式请求失败 - 请求ID: ${requestId || "无"}`);
      return;
    }
    if (requestId) {
      this._abortCallbacks[requestId] = () => stream.controller.abort();
    }
    
    this._logger.log(`🔥 开始接收流式响应 - 请求ID: ${requestId || "无"}`);
    
    let content = "";
    let chunkCount = 0;
    for await (const chunk of stream) {
      const text = chunk.choices[0]?.delta?.content || "";
      const aborted =
        requestId && !Object.keys(this._abortCallbacks).includes(requestId);
      if (aborted) {
        this._logger.log(`🔥 流式响应被中止 - 请求ID: ${requestId}`);
        content = "";
        break;
      }
      if (text) {
        onStream?.(text);
        content += text;
        chunkCount++;
      }
    }
    if (requestId) {
      delete this._abortCallbacks[requestId];
    }
    
    this._logger.log(`🔥 流式响应完成 - 请求ID: ${requestId || "无"}, 总块数: ${chunkCount}, 总字符数: ${content.length}`);
    
    if (trace && this.traceOutput) {
      this._logger.log(`✅ Answer: ${content ?? "None"}`.trim());
    }
    return withDefault(content, undefined);
  }
}

export const openai = new OpenAIClient();
