/* Duckquill의 코드 상자에 언어 이름과 접근 가능한 복사 버튼을 붙입니다.
   문법 색상은 Zola가 생성합니다. 이 파일은 코드 실행이나 내용 수정은 하지 않습니다. */
(() => {
  "use strict";

  function initializeCodeBlocks() {
    const languages = {
      python: "Python", py: "Python", rust: "Rust", rs: "Rust",
      javascript: "JavaScript", js: "JavaScript", typescript: "TypeScript",
      html: "HTML", css: "CSS", json: "JSON", toml: "TOML", yaml: "YAML",
      bash: "Bash", sh: "Shell", zsh: "Zsh", sql: "SQL", text: "Text"
    };
    const blocks = document.querySelectorAll("pre.giallo, pre[class^='language-']");

    blocks.forEach((block) => {
      if (block.closest(".crt, .pre-container")) return;
      const code = block.querySelector("code");
      if (!code) return;

      const language = code.dataset.lang || block.dataset.lang || "text";
      const filename = code.dataset.name || block.dataset.name;
      const title = document.createElement("span");
      title.className = "code-language";
      title.textContent = filename || languages[language.toLowerCase()] || language;

      const button = document.createElement("button");
      button.type = "button";
      button.title = "코드 복사";
      button.setAttribute("aria-label", `${title.textContent} 코드 복사`);
      const icon = document.createElement("i");
      icon.className = "icon";
      icon.setAttribute("aria-hidden", "true");
      const label = document.createElement("span");
      label.textContent = "복사";
      button.append(icon, label);

      const status = document.createElement("span");
      status.className = "code-status";
      status.setAttribute("role", "status");
      status.setAttribute("aria-live", "polite");
      const header = document.createElement("div");
      header.className = "header";
      header.append(title, button, status);
      const container = document.createElement("div");
      container.className = "pre-container";
      block.before(container);
      container.append(header, block);

      // 키보드로도 긴 코드 영역에 진입해 좌우로 스크롤할 수 있습니다.
      block.tabIndex = 0;
      block.setAttribute("aria-label", `${title.textContent} 코드`);

      button.addEventListener("click", async () => {
        // 줄 번호는 화면에만 표시하고 실제로 복사하는 코드에서 제외합니다.
        const copy = code.cloneNode(true);
        copy.querySelectorAll(".giallo-ln").forEach((number) => number.remove());
        const text = copy.textContent || "";
        button.disabled = true;
        try {
          if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
          await navigator.clipboard.writeText(text);
          label.textContent = "복사됨";
          status.textContent = "코드를 복사했습니다.";
        } catch (_) {
          label.textContent = "복사 실패";
          status.textContent = "클립보드에 접근하지 못했습니다. 코드를 선택해 직접 복사해 주세요.";
        } finally {
          window.setTimeout(() => {
            button.disabled = false;
            label.textContent = "복사";
          }, 1600);
        }
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeCodeBlocks, { once: true });
  } else {
    initializeCodeBlocks();
  }
})();
