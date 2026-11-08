import json
from rsiwatch.trajectory import OmpReader


def test_omp_does_not_duplicate_turn_end_usage_and_handles_partial_line(tmp_path):
    path=tmp_path/'omp.txt'
    message={'role':'assistant','timestamp':1794100999344,'responseId':'one','model':'flint',
             'content':[{'type':'thinking','thinking':'inspect'},
                        {'type':'toolCall','id':'call','name':'bash','arguments':{'command':'ls'}}],
             'usage':{'input':10,'output':3,'cacheRead':20,'cost':{'total':0}}}
    records=[{'type':'message_end','message':message},
             {'type':'turn_end','message':message},
             {'type':'agent_end','messages':[message]},
             {'type':'message_end','message':message}]
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    reader=OmpReader(path)
    assert reader.poll()
    assert reader.usage.total==33
    assert reader.tool_counts=={'bash':1}
    assert reader.first_timestamp.startswith('2026-')
    result={'type':'message_end','message':{'role':'toolResult','timestamp':1794101000000,
            'toolCallId':'call','toolName':'bash','isError':False,
            'content':[{'type':'text','text':'files'}]}}
    line=json.dumps(result)
    with path.open('a') as f:f.write(line[:20])
    reader.poll()
    assert len(reader.events)==2
    with path.open('a') as f:f.write(line[20:]+'\n')
    reader.poll()
    assert reader.events[-1].kind=='tool_result'
    assert reader.usage.total==33
