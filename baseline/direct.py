import json
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


llm2 = ChatOpenAI(
    model="",
    temperature=0.0,
    max_tokens=8192,
    openai_api_base="",
    openai_api_key="",
    streaming=True,
    response_format={
        'type': 'json_object'
    },
    callbacks=[StreamingStdOutCallbackHandler()]
)


with open(r"../dataset/The Beer Murder/role.json", 'r', encoding='utf-8') as f:
    role = json.load(f)
with open(r"../dataset/The Beer Murder/evidence.json", 'r', encoding='utf-8') as f:
    evidence = json.load(f)
with open(r"../dataset/The Beer Murder/event.json", 'r', encoding='utf-8') as f:
    event = json.load(f)


case_info = f""" -角色信息:{role} -物品信息: {evidence}  -事件信息(完全正确): {event}"""

def reasoning():
    prompt = f"""
    任务：找出真相
    ---------------------------------
    案件信息：{case_info}
    ----------------------------------
    找出真相按照下面的 JSON 格式输出：
    {{
        "suspect":"<凶手>"
        "motive":"<直接导火索(对应原文中的段落，必须是嫌疑人能直接感觉到的，私下的不算)以及深层驱动因素>"
        "modus_operandi":"<作案手法与过程>"
        "key_clue": [
        {{
          "clue_description": "<案件中所有疑点，有一个都不能漏>",
          "implication": "<这条线索在推理中的意义>"
        }},
        // 更多关键线索
      ]
    }}
    """
    llm_output = llm2.invoke([HumanMessage(content=prompt)]).content
    parsed = json.loads(llm_output)
    return parsed

if __name__ == "__main__":
    res1=reasoning()












