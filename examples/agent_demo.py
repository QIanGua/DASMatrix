#!/usr/bin/env python3
"""DASMatrix AI Agent 集成示例。

本示例展示如何将 DASMatrix 工具集成到 AI Agent 工作流中。

运行方式:
    uv run python examples/agent_demo.py

环境变量:
    ANTHROPIC_API_KEY: Anthropic API 密钥 (可选，用于真实 API 调用)
    OPENAI_API_KEY: OpenAI API 密钥 (可选)
"""

import json
from pathlib import Path

import numpy as np


def create_demo_data() -> Path:
    """创建用于演示的合成 DAS 数据。"""
    import h5py

    # 生成合成数据
    fs = 1000  # 采样率 1kHz
    duration = 5  # 5 秒
    n_channels = 100
    n_samples = fs * duration

    t = np.arange(n_samples) / fs

    # 创建合成信号: 10Hz + 50Hz + 噪声
    data = np.zeros((n_samples, n_channels))
    for ch in range(n_channels):
        signal = (
            np.sin(2 * np.pi * 10 * t)  # 10 Hz 成分
            + 0.5 * np.sin(2 * np.pi * 50 * t + ch * 0.1)  # 50 Hz 成分 (带相移)
            + 0.1 * np.random.randn(n_samples)  # 噪声
        )
        # 在某些通道添加 "事件"
        if 40 <= ch <= 60:
            event_start = int(2.5 * fs)  # 2.5 秒处
            event_duration = int(0.2 * fs)  # 0.2 秒
            signal[event_start : event_start + event_duration] += 2 * np.sin(2 * np.pi * 100 * t[:event_duration])
        data[:, ch] = signal

    # 保存为 HDF5
    demo_path = Path("/tmp/das_demo_data.h5")
    with h5py.File(demo_path, "w") as f:
        f.create_dataset("data", data=data)
        f.attrs["sampling_rate"] = fs
        f.attrs["n_channels"] = n_channels
        f.attrs["channel_spacing"] = 1.0

    print(f"✅ 创建演示数据: {demo_path}")
    print(f"   形状: {data.shape}, 采样率: {fs} Hz, 时长: {duration} 秒")

    return demo_path


def demo_tool_execution():
    """演示工具函数的直接调用。"""
    print("\n" + "=" * 60)
    print("🔧 DASMatrix Agent 工具演示 - 直接调用模式")
    print("=" * 60)

    from DASMatrix.agent import DASAgentTools

    # 创建演示数据
    demo_path = create_demo_data()

    # 初始化工具集
    tools = DASAgentTools()

    # 1. 读取数据
    print("\n📂 步骤 1: 读取 DAS 数据")
    result = tools.read_das_data(str(demo_path))
    print(f"   结果: {json.dumps(result, indent=2)}")
    data_id = result["id"]

    # 2. 获取统计信息
    print("\n📊 步骤 2: 获取数据统计")
    stats = tools.get_data_stats(data_id)
    print(f"   均值: {stats['statistics']['mean']:.4f}")
    print(f"   标准差: {stats['statistics']['std']:.4f}")
    print(f"   RMS: {stats['statistics']['rms']:.4f}")

    # 3. 信号处理
    print("\n🔬 步骤 3: 应用信号处理流水线")
    processed = tools.process_signal(
        data_id,
        operations=[
            {"op": "detrend"},
            {"op": "bandpass", "low": 5, "high": 80},
            {"op": "normalize", "method": "zscore"},
        ],
        output_name="filtered_data",
    )
    print(f"   处理后数据 ID: {processed['id']}")
    print(f"   应用操作: {processed['pipeline']}")

    # 4. 频谱分析
    print("\n📈 步骤 4: 频谱分析")
    spectrum = tools.compute_spectrum(data_id, channel=50)
    print(f"   频率范围: {spectrum['frequency_range']} Hz")
    print(f"   主导频率: {spectrum['dominant_frequency_hz']:.2f} Hz")
    print(f"   峰值频率: {[p['frequency_hz'] for p in spectrum['peak_frequencies']]}")

    # 5. 事件检测
    print("\n🔍 步骤 5: 事件检测")
    events = tools.detect_events(data_id, threshold_db=-20)
    print(f"   检测到事件数: {events['events_detected']}")
    for i, event in enumerate(events["events"][:3]):
        print(
            f"   事件 {i + 1}: {event['start_time_s']:.3f}s - {event['end_time_s']:.3f}s ({event['duration_ms']:.1f}ms)"
        )

    # 6. 可视化
    print("\n🎨 步骤 6: 生成可视化")
    viz = tools.create_visualization(data_id, plot_type="waterfall", output_path="/tmp/das_demo_waterfall.png")
    print(f"   图表类型: {viz['plot_type']}")
    print(f"   保存路径: {viz['output_path']}")

    # 7. 会话管理
    print("\n📋 步骤 7: 查看会话对象")
    session_info = tools.list_session_objects()
    print(f"   当前对象数: {session_info['count']}")
    for obj_id, info in session_info["objects"].items():
        print(f"   - {obj_id}: {info['type']}, 形状: {info.get('shape')}")

    print("\n" + "=" * 60)
    print("✅ 演示完成!")
    print("=" * 60)


def demo_simulated_agent_conversation():
    """模拟 AI Agent 对话流程。"""
    print("\n" + "=" * 60)
    print("🤖 DASMatrix Agent 工具演示 - 模拟对话模式")
    print("=" * 60)

    from DASMatrix.agent import DASAgentTools, get_openai_tools

    # 创建演示数据
    demo_path = create_demo_data()

    # 初始化工具集
    tools = DASAgentTools()

    # 获取工具 schema
    tool_schemas = get_openai_tools()
    print(f"\n📚 已注册 {len(tool_schemas)} 个工具:")
    for schema in tool_schemas:
        print(f"   - {schema['function']['name']}: {schema['function']['description'][:50]}...")

    # 模拟用户对话
    conversations = [
        {
            "user": "请帮我读取 /tmp/das_demo_data.h5 文件",
            "tool_call": {
                "name": "read_das_data",
                "arguments": {"path": str(demo_path)},
            },
        },
        {
            "user": "对数据做 10-60Hz 的带通滤波",
            "tool_call": {
                "name": "process_signal",
                "arguments": {
                    "data_id": "<DATA_ID>",  # 会被替换
                    "operations": [
                        {"op": "detrend"},
                        {"op": "bandpass", "low": 10, "high": 60},
                    ],
                },
            },
        },
        {
            "user": "分析一下主要的频率成分",
            "tool_call": {
                "name": "compute_spectrum",
                "arguments": {"data_id": "<DATA_ID>"},
            },
        },
        {
            "user": "生成一个瀑布图",
            "tool_call": {
                "name": "create_visualization",
                "arguments": {"data_id": "<DATA_ID>", "plot_type": "waterfall"},
            },
        },
    ]

    data_id = None

    print("\n--- 对话开始 ---\n")

    for conv in conversations:
        print(f"👤 用户: {conv['user']}")

        # 替换 DATA_ID 占位符
        # 替换 DATA_ID 占位符
        tool_call = conv["tool_call"]
        if not isinstance(tool_call, dict) or "arguments" not in tool_call or "name" not in tool_call:
            continue

        args = tool_call["arguments"]
        if isinstance(args, dict):
            args = args.copy()
            if "data_id" in args and args["data_id"] == "<DATA_ID>":
                args["data_id"] = data_id

        # 执行工具调用
        tool_name = tool_call["name"]
        if not isinstance(tool_name, str):
            continue

        if not hasattr(tools, tool_name):
            continue

        method = getattr(tools, tool_name)
        if isinstance(args, dict):
            result = method(**args)
        else:
            result = method()

        # 保存 data_id 供后续使用
        if "id" in result and tool_name == "read_das_data":
            data_id = result["id"]

        print(f"🔧 工具调用: {tool_name}")
        print(f"   参数: {json.dumps(args, ensure_ascii=False)}")
        print(f"   结果: {json.dumps(result, indent=2, ensure_ascii=False)[:200]}...")

        if tool_name == "read_das_data":
            print(f"🤖 Agent: 已成功读取数据，共 {result['n_channels']} 个通道，时长 {result['duration']:.1f} 秒。")
        elif tool_name == "process_signal":
            print(f"🤖 Agent: 已完成滤波处理，应用了 {result['operations_applied']} 个操作。")
        elif tool_name == "compute_spectrum":
            print(f"🤖 Agent: 频谱分析完成，主导频率为 {result['dominant_frequency_hz']:.1f} Hz。")
        elif tool_name == "create_visualization":
            print(f"🤖 Agent: 已生成{result['plot_type']}图，保存在 {result['output_path']}")

        print()

    print("--- 对话结束 ---\n")


def print_tool_schemas():
    """打印工具 Schema (用于复制到 API 调用)。"""
    from DASMatrix.agent.schemas import get_anthropic_tools, get_openai_tools

    print("\n" + "=" * 60)
    print("📋 OpenAI Function Calling Schema")
    print("=" * 60)
    print(json.dumps(get_openai_tools(), indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("📋 Anthropic Tool Use Schema")
    print("=" * 60)
    print(json.dumps(get_anthropic_tools(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DASMatrix AI Agent 集成示例")
    parser.add_argument(
        "--mode",
        choices=["direct", "conversation", "schema"],
        default="direct",
        help="演示模式: direct(直接调用), conversation(模拟对话), schema(打印Schema)",
    )
    args = parser.parse_args()

    if args.mode == "direct":
        demo_tool_execution()
    elif args.mode == "conversation":
        demo_simulated_agent_conversation()
    elif args.mode == "schema":
        print_tool_schemas()
