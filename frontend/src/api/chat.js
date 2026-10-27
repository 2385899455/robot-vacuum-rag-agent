// 所有模型调用由 Python 后端完成，浏览器仅传递问题和接收最终答案。
export async function sendChat(message) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 180_000);
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      if (response.status === 503)
        throw new Error("助手暂时忙碌或尚未就绪，请稍后重试。");
      if (response.status === 422)
        throw new Error("问题格式不正确，请输入 1–2000 个字符。");
      throw new Error(
        typeof data?.detail === "string"
          ? data.detail
          : "回答生成失败，请确认后端服务正常后重试。",
      );
    }
    if (typeof data?.answer !== "string" || !data.answer.trim()) {
      throw new Error("没有收到有效回答，请稍后重试。");
    }
    return data.answer;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("等待回答超时。后端可能仍在处理，请稍候再试。");
    }
    if (error instanceof TypeError) {
      throw new Error("暂时无法连接服务，请检查网络和后端是否启动。");
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

export async function checkHealth() {
  const response = await fetch("/health", {
    signal: AbortSignal.timeout(5000),
  });
  if (!response.ok) return false;
  const data = await response.json();
  return data.status === "ok" && data.agent_ready === true;
}
