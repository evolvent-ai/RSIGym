import json
from rsiwatch.trajectory import MiniSweReader
from rsiwatch.scanner import Trial


def test_mini_snapshot_tools_outputs_and_partial_write(tmp_path):
    agent = tmp_path/'agent'
    agent.mkdir()
    path = agent/'mini-swe-agent.trajectory.json'
    data = {'messages': [
        {'role': 'assistant', 'reasoning_content': 'inspect', 'content': None,
         'tool_calls': [{'id': 'one', 'function': {'name': 'bash', 'arguments': '{"command":"ls"}'}}],
         'extra': {'response': {'model': 'flint', 'usage': {'prompt_tokens': 20, 'completion_tokens': 3, 'prompt_tokens_details': {'cached_tokens': 8}}}}},
        {'role': 'tool', 'tool_call_id': 'one', 'content': '{"returncode":1,"output":"failed"}'},
    ]}
    path.write_text(json.dumps(data))
    (tmp_path/'config.json').write_text(json.dumps({'agent': {'name': 'mini-swe-agent'}}))
    trial = Trial(name='trial',job_name='job',path=tmp_path)
    trial.refresh()
    reader = trial.reader
    assert isinstance(reader, MiniSweReader)
    assert trial.status() == 'running'
    assert reader.tool_counts == {'bash': 1}
    assert reader.tool_errors == {'bash': 1}
    assert reader.usage.total == 23
    assert reader.usage.cache_read == 8
    assert reader.usage.input == 12
    assert reader.events[-1].body == 'failed'
    assert not reader.poll()
    path.write_text('{')
    assert not reader.poll()
    assert reader.usage.total == 23
    data['messages'].append({'role': 'assistant', 'content': 'continue'})
    path.write_text(json.dumps(data))
    assert reader.poll()
    assert reader.tool_counts == {'bash': 1}
    assert reader.usage.total == 23


def test_successful_exit_is_not_error(tmp_path):
    path = tmp_path/'mini.json'
    path.write_text(json.dumps({'messages': [
        {'role': 'exit', 'content': 'done', 'extra': {'exit_status': 'Submitted'}}
    ]}))
    reader = MiniSweReader(path)
    assert reader.poll()
    assert not reader.events[-1].is_error
