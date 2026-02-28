from DASMatrix.agent.schemas import get_anthropic_tools, get_openai_tools


def test_openai_tools_shape():
    tools = get_openai_tools()
    assert isinstance(tools, list)
    names = {tool["function"]["name"] for tool in tools}
    assert "read_das_data" in names
    assert "process_signal" in names
    assert "create_visualization" in names


def test_anthropic_tools_shape():
    tools = get_anthropic_tools()
    assert isinstance(tools, list)
    names = {tool["name"] for tool in tools}
    assert "read_das_data" in names
    assert "process_signal" in names
    assert "create_visualization" in names
