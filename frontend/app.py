"""
猫狗分类器 — Gradio 前端界面
=============================

基于 service 层的美观推理界面：
  - 文件选择、拖拽上传、剪贴板粘贴
  - 上传后自动预测
  - 支持模型切换（SEResNet18 / SEResNet34）
  - 展示预测类别、猫狗置信度、推理耗时、模型名称

Usage:
    python frontend/app.py
"""

import time
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from service import get_model_service, LABEL_MAP

# ============================================================================
# 模型缓存 — 懒加载，避免重复加载相同模型
# ============================================================================
_model_cache: dict = {}


def _get_or_load_model(model_name: str):
    """获取或加载模型服务实例（带缓存）。"""
    if model_name not in _model_cache:
        _model_cache[model_name] = get_model_service(model=model_name)
    return _model_cache[model_name]


# ============================================================================
# 预测函数
# ============================================================================

def predict(image, model_name: str):
    """
    对上传图像执行预测，返回 HTML 结果（内含状态信息）。

    Args:
        image: PIL Image，来自 gr.Image(type="pil")
        model_name: "seresnet18" | "seresnet34"
    """
    if image is None:
        return _render_empty()

    try:
        svc = _get_or_load_model(model_name)

        t_start = time.perf_counter()
        result = svc.predict(image)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        label = result["label"]
        probs = result["probabilities"]
        cat_conf = probs["cat"]
        dog_conf = probs["dog"]

        model_display = "SE-ResNet18" if model_name == "seresnet18" else "SE-ResNet34"

        return _render_result(label, cat_conf, dog_conf, elapsed_ms, model_display)

    except FileNotFoundError as e:
        return _render_error(f"模型权重文件未找到：{e}")
    except Exception as e:
        return _render_error(str(e))


# ============================================================================
# HTML 渲染
# ============================================================================

def _render_result(
    label: str,
    cat_conf: float,
    dog_conf: float,
    elapsed_ms: float,
    model_name: str,
) -> str:
    """渲染预测结果卡片 HTML。"""
    if label == "cat":
        accent = "#f59e0b"
        emoji = "🐱"
        cn_label = "猫"
    else:
        accent = "#3b82f6"
        emoji = "🐶"
        cn_label = "狗"

    cat_pct = cat_conf * 100
    dog_pct = dog_conf * 100
    main_conf = cat_conf if label == "cat" else dog_conf

    return f"""
<div class="result-card">
  <div class="prediction-header">
    <div class="prediction-emoji">{emoji}</div>
    <div class="prediction-label">
      <div class="label-main" style="color:{accent};">{cn_label} <span class="label-en">({label})</span></div>
      <div class="label-confidence">置信度 <strong>{main_conf:.2%}</strong></div>
    </div>
  </div>

  <div class="confidence-bars">
    <div class="bar-row">
      <div class="bar-label">🐱 Cat</div>
      <div class="bar-track">
        <div class="bar-fill bar-cat" style="width:{cat_pct:.1f}%">{cat_pct:.1f}%</div>
      </div>
    </div>
    <div class="bar-row">
      <div class="bar-label">🐶 Dog</div>
      <div class="bar-track">
        <div class="bar-fill bar-dog" style="width:{dog_pct:.1f}%">{dog_pct:.1f}%</div>
      </div>
    </div>
  </div>

  <div class="meta-info">
    <div class="meta-item">
      <span class="meta-icon">⚡</span>
      <span class="meta-text">推理耗时 <strong>{elapsed_ms:.1f} ms</strong></span>
    </div>
    <div class="meta-item">
      <span class="meta-icon">🧠</span>
      <span class="meta-text">模型 <strong>{model_name}</strong></span>
    </div>
  </div>

  <div class="status-line status-success">✅ 预测完成</div>
</div>
"""


def _render_empty() -> str:
    """空状态占位卡片。"""
    return """
<div class="result-card empty-state">
  <div class="empty-emoji">🐱&nbsp;&nbsp;❓&nbsp;&nbsp;🐶</div>
  <div class="empty-text">上传一张猫咪或狗狗的照片<br>AI 将自动识别并分类</div>
  <div class="empty-hint">支持拖拽上传 · 点击选择 · Ctrl+V 粘贴</div>
  <div class="status-line status-ready">🟢 模型就绪，请上传图片</div>
</div>
"""


def _render_error(msg: str) -> str:
    """错误状态卡片。"""
    return f"""
<div class="result-card error-state">
  <div class="error-emoji">⚠️</div>
  <div class="error-text">{msg}</div>
  <div class="status-line status-error">❌ 预测失败</div>
</div>
"""


# ============================================================================
# 自定义 CSS
# ============================================================================

CUSTOM_CSS = """
/* ===== 全局 ===== */
.gradio-container {
    max-width: 1120px !important;
    margin: 0 auto !important;
}

/* ===== 标题 ===== */
.app-header {
    text-align: center;
    padding: 20px 0 4px 0;
}
.app-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #f59e0b 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-header p {
    color: #6b7280;
    font-size: 0.92rem;
    margin: 4px 0 0 0;
}

/* ===== 用 CSS Grid 锁定左右双栏 — 不受内容高度变化影响 ===== */
.main-row {
    display: grid !important;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    align-items: start;
}
/* 覆盖 Gradio 默认 flex 行为，防止折叠 */
.main-row > * {
    min-width: 0 !important;
    width: 100% !important;
}

/* 去掉柱状图动画 — 避免过渡期间触发 layout reflow */
.bar-fill {
    transition: none !important;
}

/* ===== 结果卡片 ===== */
.result-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 28px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 6px 18px rgba(0,0,0,0.04);
    border: 1px solid #e5e7eb;
    min-height: 320px;
}

.prediction-header {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 24px;
}
.prediction-emoji {
    font-size: 3.2rem;
    line-height: 1;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.1));
}
.prediction-label { flex: 1; }
.label-main {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.label-en {
    font-size: 0.95rem;
    font-weight: 400;
    color: #9ca3af;
    margin-left: 2px;
}
.label-confidence {
    font-size: 0.92rem;
    color: #6b7280;
    margin-top: 4px;
}
.label-confidence strong {
    color: #111827;
    font-size: 1.1rem;
}

/* ===== 置信度柱状图 ===== */
.confidence-bars {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 24px;
}
.bar-row {
    display: flex;
    align-items: center;
    gap: 12px;
}
.bar-label {
    width: 72px;
    font-size: 0.88rem;
    font-weight: 500;
    color: #374151;
    text-align: right;
    flex-shrink: 0;
}
.bar-track {
    flex: 1;
    height: 28px;
    background: #f3f4f6;
    border-radius: 14px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 14px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
    min-width: 42px;
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    box-sizing: border-box;
}
.bar-cat {
    background: linear-gradient(90deg, #fbbf24, #f59e0b);
}
.bar-dog {
    background: linear-gradient(90deg, #60a5fa, #3b82f6);
}

/* ===== 元信息 ===== */
.meta-info {
    display: flex;
    gap: 24px;
    padding-top: 16px;
    border-top: 1px solid #f3f4f6;
}
.meta-item {
    display: flex;
    align-items: center;
    gap: 6px;
}
.meta-icon { font-size: 1.05rem; }
.meta-text {
    font-size: 0.86rem;
    color: #6b7280;
}
.meta-text strong {
    color: #111827;
    font-weight: 600;
}

/* ===== 空状态 ===== */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 320px;
    text-align: center;
}
.empty-emoji { font-size: 2.8rem; margin-bottom: 12px; }
.empty-text { font-size: 1rem; color: #374151; line-height: 1.6; margin-bottom: 8px; }
.empty-hint { font-size: 0.82rem; color: #9ca3af; }

/* ===== 错误状态 ===== */
.error-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 200px;
    text-align: center;
}
.error-emoji { font-size: 2.4rem; margin-bottom: 10px; }
.error-text { font-size: 0.92rem; color: #ef4444; line-height: 1.5; }

/* ===== 状态行 ===== */
.status-line {
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid #f3f4f6;
    font-size: 0.84rem;
    text-align: center;
}
.status-success { color: #16a34a; }
.status-ready  { color: #6b7280; }
.status-error  { color: #ef4444; }

/* ===== 底部 ===== */
.footer-note {
    text-align: center;
    color: #9ca3af;
    font-size: 0.78rem;
    padding: 8px 0 2px 0;
}

/* ===== 暗色模式 ===== */
@media (prefers-color-scheme: dark) {
    .result-card {
        background: #1f2937;
        border-color: #374151;
    }
    .bar-track { background: #374151; }
    .bar-label { color: #d1d5db; }
    .label-confidence { color: #9ca3af; }
    .label-confidence strong { color: #f9fafb; }
    .meta-text { color: #9ca3af; }
    .meta-text strong { color: #f9fafb; }
    .meta-info { border-top-color: #374151; }
    .empty-text { color: #d1d5db; }
    .status-line { border-top-color: #374151; }
}
"""


# ============================================================================
# UI 构建
# ============================================================================

def build_ui():
    """构建 Gradio Blocks 界面 — 左右双栏布局。"""
    with gr.Blocks(title="🐱 Cat vs Dog · AI 图像分类器") as demo:

        # ---- 标题区 ----
        gr.HTML("""
        <div class="app-header">
          <h1>🐱 Cat vs Dog &middot; AI 图像分类器 🐶</h1>
          <p>基于 SE-ResNet 深度学习模型 &middot; 支持 SEResNet18 / SEResNet34 切换</p>
        </div>
        """)

        # ---- 左右双栏主体 ----
        with gr.Row(elem_classes="main-row", equal_height=True):

            # ============ 左栏：输入区 ============
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="📷 上传图片",
                    image_mode="RGB",
                    height=340,
                    sources=["upload", "clipboard"],
                )

                model_selector = gr.Dropdown(
                    choices=[
                        ("SE-ResNet18 — 轻量快速", "seresnet18"),
                        ("SE-ResNet34 — 更高精度", "seresnet34"),
                    ],
                    value="seresnet18",
                    label="🧠 模型选择",
                    info="切换后自动重新预测",
                    interactive=True,
                )

            # ============ 右栏：结果区 ============
            with gr.Column(scale=1):
                result_html = gr.HTML(value=_render_empty(), show_label=False)

        # ---- 底部 ----
        gr.HTML("""
        <div class="footer-note">
          SE-ResNet 从头训练于 Kaggle Dogs vs. Cats 数据集 · 推理设备自动选择 GPU / CPU
        </div>
        """)

        # ---- 事件绑定 ----

        image_input.change(
            fn=predict,
            inputs=[image_input, model_selector],
            outputs=[result_html],
        )

        model_selector.change(
            fn=predict,
            inputs=[image_input, model_selector],
            outputs=[result_html],
        )

    return demo


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="amber",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    )
