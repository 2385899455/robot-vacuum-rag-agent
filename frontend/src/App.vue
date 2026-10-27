<script setup>
import { computed, nextTick, onMounted, ref } from "vue";
import { marked } from "marked";
import DOMPurify from "dompurify";
import { checkHealth, sendChat } from "./api/chat";

const input = ref("");
const messages = ref([]);
const loading = ref(false);
const connection = ref("checking");
const checking = ref(false);
const inputElement = ref(null);
const scrollElement = ref(null);
const notice = ref("");
const characterCount = computed(() => Array.from(input.value.trim()).length);
const canSend = computed(
  () =>
    !loading.value && characterCount.value > 0 && characterCount.value <= 2000,
);
const statusLabel = computed(
  () =>
    ({ checking: "正在连接", online: "服务已连接", offline: "服务未连接" })[
      connection.value
    ],
);
const suggestions = [
  {
    icon: "✳",
    title: "排查小故障",
    description: "吸力下降、无法回充，逐步找到原因",
    query: "扫地机器人吸力明显下降，应该怎么处理？",
    tone: "green",
  },
  {
    icon: "◷",
    title: "日常维护",
    description: "了解清洁方法，让每一次打扫更顺畅",
    query: "尘盒和滤网应该多久清理一次？",
    tone: "orange",
  },
  {
    icon: "⌂",
    title: "找到适合你的",
    description: "结合户型与生活习惯，了解选购要点",
    query: "小户型家庭选购扫地机器人时主要看哪些参数？",
    tone: "blue",
  },
  {
    icon: "▤",
    title: "查看使用报告",
    description: "体验个人使用情况与保养建议",
    query: "给我生成一个使用报告。",
    tone: "purple",
  },
];

function renderMarkdown(content) {
  // 模型文本不是可信 HTML；禁用图片等不必要内容，清理危险标签和 URL。
  return DOMPurify.sanitize(marked.parse(content, { breaks: true }), {
    ALLOWED_TAGS: [
      "p",
      "br",
      "strong",
      "em",
      "del",
      "h1",
      "h2",
      "h3",
      "h4",
      "h5",
      "h6",
      "ul",
      "ol",
      "li",
      "blockquote",
      "pre",
      "code",
      "a",
      "table",
      "thead",
      "tbody",
      "tr",
      "th",
      "td",
      "hr",
    ],
    ALLOWED_ATTR: ["href", "title", "start"],
  });
}

async function refreshHealth() {
  if (checking.value) return;
  checking.value = true;
  connection.value = "checking";
  try {
    connection.value = (await checkHealth()) ? "online" : "offline";
  } catch {
    connection.value = "offline";
  } finally {
    checking.value = false;
  }
}

async function scrollToBottom() {
  await nextTick();
  scrollElement.value?.scrollTo({
    top: scrollElement.value.scrollHeight,
    behavior: "smooth",
  });
}

async function requestAnswer(message) {
  if (loading.value) return;
  loading.value = true;
  message.error = "";
  message.status = "pending";
  notice.value = "";
  scrollToBottom();
  try {
    const answer = await sendChat(message.content);
    connection.value = "online";
    message.status = "sent";
    messages.value.push({
      id: crypto.randomUUID(),
      role: "assistant",
      content: answer,
    });
  } catch (error) {
    message.status = "failed";
    message.error = error.message;
    refreshHealth();
  } finally {
    loading.value = false;
    scrollToBottom();
    await nextTick();
    inputElement.value?.focus();
  }
}

function submit() {
  if (!canSend.value) return;
  messages.value.push({
    id: crypto.randomUUID(),
    role: "user",
    content: input.value.trim(),
    status: "pending",
    error: "",
  });
  input.value = "";
  resizeInput();
  requestAnswer(messages.value[messages.value.length - 1]);
}

function handleKeydown(event) {
  if (
    event.key === "Enter" &&
    !event.shiftKey &&
    !event.isComposing &&
    event.keyCode !== 229
  ) {
    event.preventDefault();
    submit();
  }
}

function chooseSuggestion(query) {
  input.value = query;
  inputElement.value?.focus();
  resizeInput();
}

function resetConversation() {
  if (loading.value) return;
  messages.value = [];
  input.value = "";
  notice.value = "";
  resizeInput();
  inputElement.value?.focus();
}

async function resizeInput() {
  await nextTick();
  if (!inputElement.value) return;
  inputElement.value.style.height = "auto";
  inputElement.value.style.height = `${Math.min(inputElement.value.scrollHeight, 150)}px`;
}

async function copyAnswer(content) {
  try {
    await navigator.clipboard.writeText(content);
    notice.value = "回答已复制";
  } catch {
    notice.value = "复制失败，请选中文字手动复制";
  }
}

onMounted(refreshHealth);
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <a class="brand" href="/" aria-label="净伴首页">
        <span class="brand-mark"
          ><svg viewBox="0 0 32 32" aria-hidden="true">
            <rect x="5" y="9" width="22" height="16" rx="7" />
            <path d="M16 9V5m-3 0h6M10 16h1m10 0h1M12 21h8" /></svg
        ></span>
        <span>净伴<span class="brand-sub">CLEANMATE</span></span>
      </a>
      <button class="new-chat" :disabled="loading" @click="resetConversation">
        <span>＋</span> 开始新对话 <span class="new-arrow">↗</span>
      </button>
      <div class="sidebar-section-label">你的清洁搭档</div>
      <div class="active-nav">
        <span>◉</span> 智能问答 <span class="nav-dot"></span>
      </div>
      <div class="sidebar-description">
        从一次小小的提问开始，<br />让家的日常更轻松。
      </div>
      <div class="sidebar-bottom">
        <div class="demo-card">
          <span class="demo-icon">ⓘ</span>
          <div>
            <strong>关于这个演示</strong>
            <p>天气预报来自高德服务；定位会将访问 IP 交给高德推测城市，可能不准确。个人使用记录仍为模拟数据。</p>
          </div>
        </div>
        <div class="sidebar-footer">
          <span class="avatar-small">净</span>
          <div>扫地机器人助手<small>让清洁，更有条理</small></div>
          <span class="footer-spark">✳</span>
        </div>
      </div>
    </aside>

    <main class="main-panel">
      <header class="topbar">
        <div class="page-title">
          智能问答 <span>为每个清洁问题，找到下一步</span>
        </div>
        <div class="topbar-actions">
          <button
            class="mobile-new"
            :disabled="loading"
            @click="resetConversation"
            aria-label="开始新对话"
          >
            ＋
          </button>
          <button
            class="connection"
            :class="connection"
            :disabled="checking"
            @click="refreshHealth"
            title="点击检查本地后端连接"
          >
            <span class="status-dot"></span>{{ statusLabel
            }}<span class="refresh-icon">↻</span>
          </button>
        </div>
      </header>

      <div v-if="connection === 'offline'" class="offline-banner" role="status">
        暂时无法连接助手服务，请确认后端已启动。<button
          @click="refreshHealth"
          :disabled="checking"
        >
          重新连接
        </button>
      </div>

      <section
        ref="scrollElement"
        class="conversation"
        :class="{ 'has-messages': messages.length }"
        aria-label="聊天记录"
      >
        <div v-if="!messages.length" class="welcome">
          <div class="welcome-badge">
            <span>✦</span> 给日常清洁，一点聪明的帮助
          </div>
          <h1>你好，今天有什么<br /><span>清洁问题？</span></h1>
          <p class="welcome-description">
            使用技巧、故障排查，或是一份保养建议。<br />告诉我你的问题，我们一起找到答案。
          </p>
          <div class="suggestions">
            <button
              v-for="item in suggestions"
              :key="item.title"
              class="suggestion"
              @click="chooseSuggestion(item.query)"
            >
              <span class="suggestion-icon" :class="item.tone">{{
                item.icon
              }}</span
              ><span class="suggestion-arrow">↗</span>
              <strong>{{ item.title }}</strong
              ><span class="suggestion-description">{{
                item.description
              }}</span>
            </button>
          </div>
          <div class="welcome-footnote">
            <span>✧</span> 描述设备现象和使用环境，可以帮助我更好地理解问题
          </div>
        </div>

        <div
          v-else
          class="message-list"
          aria-live="polite"
          aria-relevant="additions text"
        >
          <div class="conversation-start">新的提问，新的思路</div>
          <article
            v-for="message in messages"
            :key="message.id"
            class="message"
            :class="message.role"
          >
            <template v-if="message.role === 'user'">
              <div class="user-message">{{ message.content }}</div>
              <div v-if="message.error" class="message-error" role="alert">
                <span>{{ message.error }}</span
                ><button :disabled="loading" @click="requestAnswer(message)">
                  重试
                </button>
              </div>
            </template>
            <template v-else>
              <div class="assistant-label">
                <span class="assistant-avatar">✳</span> 净伴
                <span>清洁助手</span>
              </div>
              <div
                class="markdown-content"
                v-html="renderMarkdown(message.content)"
              ></div>
              <button class="copy-button" @click="copyAnswer(message.content)">
                ▢ 复制回答
              </button>
            </template>
          </article>
          <div v-if="loading" class="loading-message" role="status">
            <span class="assistant-avatar">✳</span
            ><span class="loading-dots"><i></i><i></i><i></i></span
            ><span>正在整理回答，请稍候…</span>
          </div>
        </div>
      </section>

      <footer class="composer-area">
        <div v-if="notice" class="toast" role="status">
          {{ notice }}
          <button @click="notice = ''" aria-label="关闭提示">×</button>
        </div>
        <form class="composer" @submit.prevent="submit">
          <label for="question" class="sr-only">输入你的问题</label>
          <textarea
            id="question"
            ref="inputElement"
            v-model="input"
            rows="1"
            placeholder="问问净伴，例如：主刷不转了该怎么办？"
            @keydown="handleKeydown"
            @input="resizeInput"
            :aria-invalid="characterCount > 2000"
            aria-describedby="input-hint"
          ></textarea>
          <div class="composer-toolbar">
            <span id="input-hint" class="input-hint">{{
              loading
                ? "回答生成中，可先编辑下一个问题"
                : "Enter 发送 · Shift + Enter 换行"
            }}</span>
            <div class="send-controls">
              <span
                class="character-count"
                :class="{ over: characterCount > 2000 }"
                >{{ characterCount }} / 2000</span
              ><button
                class="send-button"
                type="submit"
                :disabled="!canSend"
                :aria-label="loading ? '正在生成回答' : '发送消息'"
              >
                <span v-if="loading" class="spinner"></span
                ><svg v-else viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M12 19V5m-6 6 6-6 6 6" />
                </svg>
              </button>
            </div>
          </div>
        </form>
        <p class="composer-note">
          每次提问独立回答，请在问题中提供必要背景。聊天记录仅保留在当前页面。
          <span class="mobile-demo-note">天气查询使用高德；定位会向高德发送访问 IP，结果可能不准确。使用记录为模拟数据。</span>
        </p>
      </footer>
    </main>
  </div>
</template>
